select
    o.order_id,
    o.ordered_on,
    o.channel,
    o.order_total_cents,
    coalesce(p.captured_cents, 0) as captured_cents
from {{ ref('stg_orders') }} as o
left join {{ ref('int_payments_by_order') }} as p on o.order_id = p.order_id
