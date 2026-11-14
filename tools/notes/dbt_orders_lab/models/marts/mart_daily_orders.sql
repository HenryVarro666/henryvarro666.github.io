select
    ordered_on,
    channel,
    count(*) as order_count,
    sum(case when captured_cents > 0 then 1 else 0 end) as paid_order_count,
    sum(order_total_cents) as ordered_cents,
    sum(captured_cents) as captured_cents
from {{ ref('fct_orders') }}
group by ordered_on, channel
