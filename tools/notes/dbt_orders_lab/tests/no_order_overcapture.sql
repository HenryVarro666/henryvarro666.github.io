-- In this shop, amounts are in USD cents and orders cannot be overcaptured.
select order_id, order_total_cents, captured_cents
from {{ ref('fct_orders') }}
where captured_cents < 0 or captured_cents > order_total_cents
