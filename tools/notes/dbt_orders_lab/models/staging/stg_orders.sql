select
    order_id,
    ordered_on,
    lower(trim(channel)) as channel,
    order_total_cents
from {{ source('shop', 'orders') }}
