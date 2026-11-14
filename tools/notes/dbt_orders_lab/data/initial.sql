-- Resets the two synthetic input tables only.
begin transaction;
create schema if not exists raw;

create or replace table raw.orders (
    order_id varchar primary key,
    ordered_on date not null,
    channel varchar not null,
    order_total_cents bigint not null,
    order_status varchar not null,
    updated_at timestamp not null
);

insert into raw.orders values
    ('o100', date '2026-01-01', 'web',   2500, 'open', timestamp '2026-01-01 09:00:00'),
    ('o101', date '2026-01-01', 'web',   1800, 'open', timestamp '2026-01-01 10:00:00'),
    ('o102', date '2026-01-02', 'store', 3200, 'open', timestamp '2026-01-02 11:00:00'),
    ('o103', date '2026-01-02', 'web',   4100, 'open', timestamp '2026-01-02 12:00:00');

create or replace table raw.payment_events (
    delivery_id varchar primary key,
    payment_id varchar not null,
    order_id varchar not null,
    revision integer not null,
    payment_status varchar not null,
    amount_cents bigint not null,
    event_at timestamp not null,
    loaded_at timestamp not null
);

insert into raw.payment_events values
    ('d001', 'p01', 'o100', 1, 'captured',   1000, '2026-01-01 09:01:00', '2026-01-01 09:02:00'),
    ('d002', 'p02', 'o100', 1, ' Captured ', 1500, '2026-01-01 09:05:00', '2026-01-01 09:06:00'),
    ('d003', 'p03', 'o101', 1, 'failed',     1800, '2026-01-01 10:10:00', '2026-01-01 10:11:00'),
    ('d004', 'p04', 'o102', 2, 'captured',   3200, '2026-01-02 11:55:00', '2026-01-02 12:00:00'),
    ('d005', 'p02', 'o100', 1, 'captured',   1500, '2026-01-01 09:05:00', '2026-01-04 19:00:00');
commit;
