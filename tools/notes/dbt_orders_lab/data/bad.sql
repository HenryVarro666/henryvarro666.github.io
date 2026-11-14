-- A new upstream status the lookup table does not recognize.
insert into raw.payment_events values
    ('d_bad', 'p_bad', 'o103', 1, 'unknown', 999, '2026-01-02 12:01:00', '2026-01-05 10:00:00')
on conflict (delivery_id) do nothing;
