-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Databricks Notes: payment pipeline
-- MAGIC
-- MAGIC Replace `workspace` with a catalog you can write to. Use a dedicated sandbox.
-- MAGIC Run the cells in order. Record the Silver version from DESCRIBE HISTORY before the Bronze append.
-- MAGIC This exercise overwrites its own Bronze and Silver tables if you rerun the seed cells.

-- COMMAND ----------

SHOW CATALOGS;

-- COMMAND ----------

USE CATALOG `workspace`;

-- COMMAND ----------

CREATE SCHEMA IF NOT EXISTS databricks_notes_lab;

-- COMMAND ----------

USE SCHEMA databricks_notes_lab;

-- COMMAND ----------

SELECT current_catalog() AS catalog_name, current_schema() AS schema_name;

-- COMMAND ----------

CREATE OR REPLACE TABLE payments_bronze USING DELTA AS
SELECT * FROM VALUES
  (1, 1, 'p01', 1, DATE '2026-11-01', 'web',   'approved', '2499'),
  (1, 2, 'p01', 1, DATE '2026-11-01', 'web',   'approved', '2499'),
  (1, 3, 'p02', 1, DATE '2026-11-01', 'store', 'declined', '1800'),
  (1, 4, 'p03', 2, DATE '2026-11-02', 'web',   'approved', '3100'),
  (1, 5, 'p04', 1, DATE '2026-11-02', 'store', 'approved', '2400'),
  (1, 6, 'p_bad', 1, DATE '2026-11-02', 'web', 'approved', 'oops')
AS v(batch_id, delivery_id, payment_id, event_version, event_day,
     channel, status, amount_raw);

-- COMMAND ----------

SELECT * FROM payments_bronze ORDER BY delivery_id;

-- COMMAND ----------

CREATE OR REPLACE VIEW payments_checked AS
WITH typed AS (
  SELECT *, TRY_CAST(amount_raw AS BIGINT) AS amount_cents
  FROM payments_bronze
)
SELECT *,
  CASE
    WHEN payment_id IS NULL OR TRIM(payment_id) = ''
      OR event_version IS NULL OR event_version < 1
      OR event_day IS NULL THEN 'invalid_key_version_or_day'
    WHEN channel IS NULL OR channel NOT IN ('web', 'store', 'mobile')
      THEN 'invalid_channel'
    WHEN status IS NULL OR status NOT IN ('approved', 'declined')
      THEN 'invalid_status'
    WHEN amount_cents IS NULL OR amount_cents <= 0
      THEN 'invalid_amount'
  END AS reject_reason
FROM typed;

-- COMMAND ----------

CREATE OR REPLACE VIEW payments_rejects AS
SELECT * FROM payments_checked WHERE reject_reason IS NOT NULL;

-- COMMAND ----------

SELECT payment_id, amount_raw, reject_reason FROM payments_rejects;

-- COMMAND ----------

CREATE OR REPLACE VIEW payments_batch_latest AS
SELECT batch_id, payment_id, event_version, event_day, channel, status, amount_cents
FROM (
  SELECT *, ROW_NUMBER() OVER (
    PARTITION BY batch_id, payment_id
    ORDER BY event_version DESC, delivery_id DESC
  ) AS row_num
  FROM payments_checked
  WHERE reject_reason IS NULL
) AS ranked
WHERE row_num = 1;

-- COMMAND ----------

CREATE OR REPLACE TABLE payments_silver USING DELTA AS
SELECT payment_id, event_version, event_day, channel, status, amount_cents
FROM payments_batch_latest
WHERE batch_id = 1;

-- COMMAND ----------

DESCRIBE HISTORY payments_silver;

-- COMMAND ----------

CREATE OR REPLACE VIEW payments_daily AS
SELECT
  event_day,
  channel,
  COUNT(*) AS payment_count,
  COUNT_IF(status = 'approved') AS approved_count,
  SUM(CASE WHEN status = 'approved' THEN amount_cents ELSE 0 END) AS approved_cents
FROM payments_silver
GROUP BY event_day, channel;

-- COMMAND ----------

SELECT * FROM payments_daily ORDER BY event_day, channel;

-- COMMAND ----------

SELECT SUM(payment_count) AS payments,
       SUM(approved_count) AS approved,
       SUM(approved_cents) AS approved_cents
FROM payments_daily;

-- COMMAND ----------

INSERT INTO payments_bronze VALUES
  (2, 7,  'p02', 2, DATE '2026-11-01', 'store',  'approved', '2100'),
  (2, 8,  'p04', 2, DATE '2026-11-02', 'store',  'declined', '2400'),
  (2, 9,  'p03', 1, DATE '2026-11-02', 'web',    'declined', '3000'),
  (2, 10, 'p05', 1, DATE '2026-11-02', 'mobile', 'declined', '1200');

-- COMMAND ----------

MERGE INTO payments_silver AS target
USING (SELECT * FROM payments_batch_latest WHERE batch_id = 2) AS incoming
ON target.payment_id = incoming.payment_id
WHEN MATCHED AND incoming.event_version > target.event_version THEN
  UPDATE SET
    event_version = incoming.event_version,
    event_day = incoming.event_day,
    channel = incoming.channel,
    status = incoming.status,
    amount_cents = incoming.amount_cents
WHEN NOT MATCHED THEN
  INSERT (payment_id, event_version, event_day, channel, status, amount_cents)
  VALUES (incoming.payment_id, incoming.event_version, incoming.event_day,
          incoming.channel, incoming.status, incoming.amount_cents);

-- COMMAND ----------

SELECT * FROM payments_daily ORDER BY event_day, channel;

-- COMMAND ----------

SELECT SUM(payment_count) AS payments,
       SUM(approved_count) AS approved,
       SUM(approved_cents) AS approved_cents
FROM payments_daily;

-- COMMAND ----------

SELECT payment_id, event_version, status, amount_cents
FROM payments_silver ORDER BY payment_id;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Replace 0 in the next cell with the version you recorded. For a replay test, rerun just the MERGE and compare the outputs.

-- COMMAND ----------

SELECT COUNT(*) AS payments,
       COUNT_IF(status = 'approved') AS approved,
       SUM(CASE WHEN status = 'approved' THEN amount_cents ELSE 0 END) AS approved_cents
FROM payments_silver VERSION AS OF 0;
