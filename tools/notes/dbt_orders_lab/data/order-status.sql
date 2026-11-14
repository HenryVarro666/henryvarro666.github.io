update raw.orders
set order_status = 'complete', updated_at = timestamp '2026-01-06 10:00:00'
where order_id = 'o100';
