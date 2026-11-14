-- Later arrivals for old business dates. Replaying this file is safe.
insert into raw.payment_events values
    ('d006', 'p02', 'o100', 2, 'captured', 1200, '2026-01-01 09:05:00', '2026-01-05 09:00:00'),
    ('d007', 'p04', 'o102', 1, 'failed',   3200, '2026-01-02 11:54:00', '2026-01-05 09:01:00'),
    ('d008', 'p05', 'o101', 1, 'captured', 1800, '2026-01-01 10:15:00', '2026-01-05 09:02:00')
on conflict (delivery_id) do nothing;
