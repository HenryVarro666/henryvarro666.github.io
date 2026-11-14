# dbt orders lab

The two input tables contain fictional orders and payment deliveries. All writes
go to `orders_lab.duckdb` beside this file. Run the commands from this project
directory; the connection profile uses a relative path.

Use Python 3.12 (the tested version) and a fresh virtual environment:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
dbt --version
python load_data.py initial
dbt debug --profiles-dir .
dbt build --profiles-dir .
dbt show --profiles-dir . --inline "select * from {{ ref('mart_daily_orders') }} order by ordered_on, channel"
```

On Windows PowerShell, use `py -3.12 -m venv .venv` and
`.\.venv\Scripts\Activate.ps1`.

- Initial January 1 web row: 2 orders, 1 paid, 4,300 ordered cents, 2,500 captured cents.
- Run `python load_data.py late`, then
  `dbt build --profiles-dir . --select 'int_payments_current+'`. That row
  now has 2 paid orders and 4,000 captured cents. Replay the input and rebuild
  to check that the total holds.
- Run `python load_data.py bad` and then the full `dbt build --profiles-dir .`.
  Expect the payment-status relationship test to fail and the daily mart to be
  skipped. Recover with `python load_data.py fix`, then
  `dbt build --profiles-dir . --select 'int_payments_current+' --full-refresh`.
- Run `python load_data.py order-status` and
  `dbt snapshot --profiles-dir . --select orders_history`. Query
  `analytics_dev.orders_history` for both versions of `o100`.
- Run `dbt docs generate --profiles-dir .` and `dbt docs serve` for local
  lineage and documentation.

To repeat from scratch, extract a fresh copy into a new directory. Reloading the
initial fixture replaces only the raw tables; it leaves existing dbt outputs and
snapshot history in the database.
