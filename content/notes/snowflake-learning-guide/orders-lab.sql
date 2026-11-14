-- Snowflake Notes: a small, synthetic order analytics lab.
-- Run sections separately, in order, in a personal sandbox. Use the same session.
-- Stop if either CREATE below reports that the name is already in use.

-- 1. Create the lab and check the session context.
USE ROLE SYSADMIN;

CREATE WAREHOUSE sf_learn_orders_wh
  WAREHOUSE_TYPE = 'STANDARD'
  WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE
  STATEMENT_TIMEOUT_IN_SECONDS = 120;

CREATE DATABASE sf_learn_orders DATA_RETENTION_TIME_IN_DAYS = 1;

USE WAREHOUSE sf_learn_orders_wh;
USE DATABASE sf_learn_orders;
CREATE SCHEMA raw;
CREATE SCHEMA core;
CREATE SCHEMA mart;

SELECT CURRENT_ROLE() AS role_name, CURRENT_WAREHOUSE() AS warehouse_name,
       CURRENT_DATABASE() AS database_name;

-- 2. Load batch A. The raw table keeps fields as received, including bad values.
CREATE TABLE raw.order_events (
  ingest_id NUMBER(10, 0) NOT NULL,
  order_id_text VARCHAR,
  order_date_text VARCHAR,
  status_text VARCHAR,
  amount_text VARCHAR,
  source_version_text VARCHAR
);

INSERT INTO raw.order_events VALUES
  (1, 'O1001', '2026-11-01', 'PAID',     '120.00', '2'),
  (2, 'O1002', '2026-11-01', 'PAID',      '75.50', '1'),
  (3, 'O1003', '2026-11-01', 'PAID',      '50.00', '1'),
  (4, 'O1001', '2026-11-01', 'PAID',     '100.00', '1'),
  (5, 'O1003', '2026-11-01', 'REFUNDED',  '50.00', '2'),
  (6, 'O1004', '2026-11-02', 'PAID',      '80.00', '1'),
  (7, 'O1004', '2026-11-02', 'PAID',      '80.00', '1'),
  (8, 'O1005', '2026-11-02', 'PAID',     'oops',   '1'),
  (9, 'O1006', '2026-11-02', 'PENDING',   '40.00', '1');

CREATE VIEW raw.order_events_checked AS
WITH typed AS (
  SELECT ingest_id,
         NULLIF(TRIM(order_id_text), '') AS order_id,
         TRY_TO_DATE(order_date_text, 'YYYY-MM-DD') AS order_date,
         UPPER(TRIM(status_text)) AS status,
         amount_text,
         IFF(REGEXP_LIKE(TRIM(amount_text), '[0-9]+([.][0-9]{1,2})?'),
             TRY_TO_DECIMAL(TRIM(amount_text), 12, 2), NULL) AS amount,
         IFF(REGEXP_LIKE(TRIM(source_version_text), '[0-9]+'),
             TRY_TO_NUMBER(TRIM(source_version_text), 10, 0), NULL) AS source_version
  FROM raw.order_events
)
SELECT typed.*,
       CASE
         WHEN order_id IS NULL THEN 'MISSING_ORDER_ID'
         WHEN order_date IS NULL THEN 'BAD_DATE'
         WHEN status IS NULL OR status NOT IN ('PAID', 'REFUNDED', 'PENDING') THEN 'BAD_STATUS'
         WHEN amount IS NULL THEN 'BAD_AMOUNT'
         WHEN source_version IS NULL OR source_version < 1 THEN 'BAD_VERSION'
       END AS reject_reason
FROM typed;

SELECT ingest_id, order_id, amount_text, reject_reason
FROM raw.order_events_checked
WHERE reject_reason IS NOT NULL
ORDER BY ingest_id;
-- Expect exactly (8, 'O1005', 'oops', 'BAD_AMOUNT').

CREATE TABLE core.orders (
  order_id VARCHAR NOT NULL,
  order_date DATE NOT NULL,
  status VARCHAR NOT NULL,
  amount NUMBER(12, 2) NOT NULL,
  source_version NUMBER(10, 0) NOT NULL
);

CREATE VIEW mart.daily_paid_sales AS
SELECT order_date, COUNT(*) AS paid_orders, SUM(amount) AS paid_sales
FROM core.orders
WHERE status = 'PAID'
GROUP BY order_date;

-- 3. Apply batch A, then run the checks below.
SET batch_first = 1;
SET batch_last = 9;

-- Start of the reusable MERGE block.
MERGE INTO core.orders AS t
USING (
  SELECT order_id, order_date, status, amount, source_version
  FROM raw.order_events_checked
  WHERE reject_reason IS NULL AND ingest_id BETWEEN $batch_first AND $batch_last
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY order_id ORDER BY source_version DESC, ingest_id DESC
  ) = 1
) AS s ON t.order_id = s.order_id
WHEN MATCHED AND s.source_version > t.source_version THEN UPDATE SET
  t.order_date = s.order_date,
  t.status = s.status,
  t.amount = s.amount,
  t.source_version = s.source_version
WHEN NOT MATCHED THEN INSERT (order_id, order_date, status, amount, source_version)
  VALUES (s.order_id, s.order_date, s.status, s.amount, s.source_version);
-- End of the reusable MERGE block.

SELECT COUNT(*) AS order_rows FROM core.orders;
SELECT order_id, COUNT(*) AS n FROM core.orders GROUP BY order_id HAVING COUNT(*) > 1;
SELECT * FROM mart.daily_paid_sales ORDER BY order_date;
-- Expect 5 orders, no duplicate keys, and sales rows (2, 195.50), (1, 80.00).

-- 4. Optional: insert batch B once. Then run only the MERGE block above, twice.
INSERT INTO raw.order_events VALUES
  (10, 'O1002', '2026-11-01', 'PAID', '70.50',  '2'),
  (11, 'O1001', '2026-11-01', 'PAID', '100.00', '1'),
  (12, 'O1007', '2026-11-02', 'PAID', '25.00',  '1');

SET batch_first = 10;
SET batch_last = 12;
-- Run the MERGE block from section 3 now. The first run updates O1002 and inserts
-- O1007. The second run should report zero updates and inserts.

SELECT order_id, amount, source_version
FROM core.orders WHERE order_id IN ('O1001', 'O1002', 'O1007') ORDER BY order_id;
SELECT * FROM mart.daily_paid_sales ORDER BY order_date;
-- After either MERGE: O1001 remains 120.00 at v2. Sales: (2, 190.50), (2, 105.00).

-- 5. Optional: expand a JSON array into one row per line item.
WITH orders AS (
  SELECT PARSE_JSON('{
    "order_id": "O1001",
    "items": [{"sku":"A","qty":2,"price":40.00}, {"sku":"B","qty":1,"price":40.00}]
  }') AS payload
)
SELECT o.payload:order_id::VARCHAR AS order_id,
       f.value:sku::VARCHAR AS sku,
       f.value:qty::NUMBER(10, 0) * f.value:price::NUMBER(12, 2) AS line_amount
FROM orders AS o, LATERAL FLATTEN(INPUT => o.payload:items) AS f
ORDER BY sku;
-- Expect O1001 A 80.00 and O1001 B 40.00.
