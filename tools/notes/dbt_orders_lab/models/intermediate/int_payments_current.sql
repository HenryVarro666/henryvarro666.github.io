{{ config(materialized='incremental', incremental_strategy='merge',
          unique_key='payment_id') }}

with deliveries as (
    select * from {{ ref('stg_payments') }}
),

{% if is_incremental() %}
changed_keys as (
    select distinct payment_id
    from deliveries
    where loaded_at >= (
        select coalesce(max(loaded_at), timestamp '1900-01-01')
        from {{ this }}
    ) - interval '2 days'
),
{% endif %}

ranked as (
    select
        d.*,
        row_number() over (
            partition by d.payment_id
            order by d.revision desc, d.loaded_at desc, d.delivery_id desc
        ) as revision_rank
    from deliveries as d
    {% if is_incremental() %}
    inner join changed_keys as k on d.payment_id = k.payment_id
    {% endif %}
)

select
    payment_id, order_id, revision, payment_status, amount_cents, event_at, loaded_at
from ranked
where revision_rank = 1
