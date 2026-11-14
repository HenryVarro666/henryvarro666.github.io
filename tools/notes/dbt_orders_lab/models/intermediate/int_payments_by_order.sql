select
    p.order_id,
    sum(case when s.counts_as_capture then p.amount_cents else 0 end) as captured_cents
from {{ ref('int_payments_current') }} as p
inner join {{ ref('payment_statuses') }} as s
    on p.payment_status = s.payment_status
group by p.order_id
