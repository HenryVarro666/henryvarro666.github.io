select
    delivery_id,
    payment_id,
    order_id,
    revision,
    lower(trim(payment_status)) as payment_status,
    amount_cents,
    event_at,
    loaded_at
from {{ source('shop', 'payment_events') }}
