-- Remove the bad *synthetic fixture* from the source.
delete from raw.payment_events where delivery_id = 'd_bad';
