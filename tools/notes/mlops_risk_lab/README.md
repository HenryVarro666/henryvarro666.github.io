# A small, local MLOps risk lab

Train a toy binary classifier, freeze its evidence, evaluate it, activate a version,
serve the same pipeline, detect a shifted batch, and roll back. Everything runs on
a CPU laptop with Python 3.12. No account, external dataset, GPU, or cloud service
is needed. The synthetic process is deliberately simple. Its probabilities,
metrics, thresholds, and gates are **not estimates of real fraud risk**.

Validated on 2026-11-18 with Python 3.12.16, macOS arm64. Main versions:
scikit-learn 1.7.2, pandas 2.3.3, numpy 2.3.4, scipy 1.16.3, joblib 1.5.2,
FastAPI 0.121.1, uvicorn 0.38.0, pytest 8.4.2. `requirements.lock` pins the
tested transitive dependencies; it is a version lock, not a hash-verified lock.

## Run the complete demo

Run commands from this directory. On macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python lab.py generate
python lab.py train --run-id baseline
python lab.py evaluate --run-id baseline
python lab.py promote --run-id baseline
python lab.py predict --json '{"amount":120.0,"account_age_days":90.0,"attempts_1h":2,"channel":"web"}'
python -m pytest -q
```

Expected: `generate` writes 6,000 reference and 1,500 shifted rows. After the
two-hour label delay purge, splits contain 3,481 / 1,081 / 1,200 rows.
`train` chooses a threshold of `0.20` on validation F1. `evaluate` reports the
held-out test, with average precision about `0.4350`, ROC AUC `0.7562`, precision
`0.3974`, recall `0.6456`, F1 `0.4920`, and `gate_passed: true`. The sample
prediction has probability about `0.3113`, `flagged: true`, and
`model_version: "baseline"`. Small numerical differences across platforms are
possible. The core suite reports **21 passed, 1 skipped**; installing the optional
tracking dependencies enables the remaining test (**22 passed**).

The default output root is `workspace/`. To repeat from scratch without deleting
evidence, put the global option **before** each subcommand:

```bash
python lab.py --root /tmp/risk-demo-2 generate
python lab.py --root /tmp/risk-demo-2 train --run-id baseline
```

Use that same `--root` for all subsequent CLI commands. Run IDs and evaluation
reports cannot be overwritten. Use a new ID to change model parameters or gates.

## Serve and reject a bad request

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --no-access-log
```

In another terminal:

```bash
curl -sS http://127.0.0.1:8000/healthz
curl -sS http://127.0.0.1:8000/readyz
curl -sS -H 'Content-Type: application/json' \
  -d '{"amount":120.0,"account_age_days":90.0,"attempts_1h":2,"channel":"web"}' \
  http://127.0.0.1:8000/predict
curl -i -H 'Content-Type: application/json' \
  -d '{"amount":-1,"account_age_days":90.0,"attempts_1h":2,"channel":"web"}' \
  http://127.0.0.1:8000/predict
```

The last request returns **HTTP 422**. Numeric strings, unknown channels, extra
fields such as `label`, non-finite values, and fractional `attempts_1h` are also
rejected. Features are `amount` (0..100,000), `account_age_days` (0..36,500),
`attempts_1h` (integer 0..1,000), and `channel` (`web`, `mobile`, or `store`).
Request features and raw identifiers are not written to prediction logs.

`/healthz` measures process liveness. `/readyz` returns 503 until a compatible
active bundle exists. A successful prediction includes `model_version`,
`bundle_sha256`, `risk_probability`, `threshold`, `flagged`, and `synthetic_only`.
The process loads one immutable bundle at startup. **Restart uvicorn after
promotion or rollback**; changing the alias does not reload running processes.
For another root, set `MLOPS_LAB_ROOT=/tmp/risk-demo-2` when starting uvicorn.

## Monitor, promote another version, and roll back

```bash
python lab.py monitor
python lab.py train --run-id candidate --C 0.2
python lab.py evaluate --run-id candidate
python lab.py promote --run-id candidate
python lab.py rollback
```

`monitor` reads `workspace/data/shifted.csv` by default. The generated batch has
deliberate feature **and** concept changes: data drift, score drift, and labeled
performance degradation alerts are all true. Its AP is about `0.1947`, ROC AUC
`0.3595`. These are distinct signals. Feature drift alone does not establish
degradation. A batch with changed inputs and stable outcomes could drift without
losing predictive quality.

`--data PATH` accepts another CSV with the same feature and event metadata columns.
Unlabeled monitoring inputs omit both `label` and `label_ready_time`. Reports use
aggregate standardized mean differences, category total variation, mean score
shift, and labeled metrics. These simple fixed heuristics are not statistical
drift tests. Performance comparisons need at least 100 mature rows and both
classes. Status is `available`, `insufficient_mature_labels`, `one_class_labels`,
or `unavailable_without_mature_labels`; unavailable performance is `null`.

`--as-of` defaults to the synthetic snapshot `2025-02-01T00:00:00Z`, not the wall
clock. Timestamps normalize to UTC, future events are excluded, and performance
uses only rows whose `label_ready_time <= as_of`. For example:

```bash
python lab.py monitor --as-of 2025-01-10
```

That snapshot contains one observed event and no mature labels, so it cannot
report performance. Reports are new `workspace/reports/monitor-*.json` files.
No row-level predictions or identifiers are stored in reports.

Candidate `C=0.2` also passes the toy gates. It uses threshold about `0.17`, with
test AP `0.4351`, precision `0.3516`, recall `0.7046`, and F1 `0.4691`. This
demonstrates version changes; it does not establish candidate superiority.
`rollback` restores the compatible previous bundle by atomically swapping the
local alias. A second rollback swaps the two versions again. Restart the API to
observe either change. The alias file is a **teaching registry**, separate from
MLflow Model Registry; it has no authentication, approval, or multi-writer lock.

To see a release refusal:

```bash
python lab.py train --run-id blocked
python lab.py evaluate --run-id blocked --min-average-precision 0.999
python lab.py promote --run-id blocked
```

Evaluation writes `gate_passed: false` and exits 0 because producing the report
succeeded. Promotion refuses with `error: Promotion gate refused this bundle`
and exit code 2. `ci.example.yml` explicitly asserts the report gate after
evaluation; the example is not installed as a repository workflow. Its GitHub
execution has not been tested. The optional Dockerfile example below is also
unbuilt and untested.

## What the artifacts prove

```text
workspace/
  data/{reference,shifted}.csv
  runs/<run-id>/
    model.joblib                 # preprocessing + estimator together
    manifest.json                # threshold, schema, validation metrics, hashes
    environment.json             # Python, platform, core package versions
    lab.py, api.py                # exact code snapshots
    requirements.lock
  evaluations/<run-id>.json       # held-out test + fixed gate, separate from activation
  registry/production.json        # current + previous run IDs and bundle digests
  reports/monitor-*.json          # aggregate batch checks
```

The generator uses seed 42 and independent event IDs. Data ingestion rejects
duplicate IDs, unknown/leaking columns, invalid times, and malformed features.
Splits are chronological 60/20/20; delayed labels unavailable at the next decision
boundary are purged. IDs are disjoint across splits. `StandardScaler` and
`OneHotEncoder` live inside one sklearn `Pipeline`, fit only on training data.
Threshold selection uses validation only; the test is read for metrics only by
the separate evaluation command. Evaluation verifies the original data hash,
split policy, snapshot time, split ID hashes and counts before scoring. Test-set
reuse for repeated candidate selection would still bias conclusions; this tiny
demo does not provide an untouched final audit set for a long tuning campaign.

The run manifest hashes data, code snapshots, environment, lock, and serialized
model. Loading verifies the manifest and file digests plus input schema and core
runtime versions before deserialization. The alias points to a specific bundle
digest. Promotion requires a matching passing evaluation and rollback checks the
previous bundle's compatibility before updating the alias. The built-in release
gate is AP >= .25, ROC AUC >= .65, recall >= .45; the CLI can raise the AP minimum.
These pedagogical gates do not encode real loss, capacity, fairness, calibration,
or business approval requirements.

Immutability here means commands refuse to overwrite runs/evaluations and check
digests. Local files are not an authenticated artifact store, and checksums do
not protect against an attacker replacing both files and manifests. `joblib`
uses pickle semantics: load only bundles created locally by this lab. Alias
replacement is atomic on one filesystem and assumes one writer; the lab does
not provide distributed transactions or durable release history.

## Optional MLflow tracking

```bash
python -m pip install -r requirements-mlflow.txt
python lab.py train --run-id tracked --mlflow
python -m pytest -q
```

Tested with `mlflow-skinny==3.6.0`, `SQLAlchemy==2.0.44`, and
`alembic==1.17.1`. SQLite needs the latter two extras. The command explicitly
uses `sqlite:///.../workspace/mlflow.db` and keeps artifact copies under
`workspace/mlartifacts/`; `workspace/tracking/tracked.json` links the local run
to its MLflow run ID. It logs parameters, validation metrics, and the bundle.
There is no server or account in this path. The skinny package is sufficient for
this tested client workflow; a browser UI/server is not part of this lab.
Core commands do not import MLflow. MLflow tracking does not activate a model.

## Optional container example (unverified)

The Dockerfile is a starting example; its build and runtime have **not** been
executed. It uses a Python 3.12 image and the same core lock. A separate named
volume lets training and serving use the container environment throughout:

```bash
docker build -t mlops-risk-lab .
docker run --rm -v risk-lab-demo:/app/workspace mlops-risk-lab python lab.py generate
docker run --rm -v risk-lab-demo:/app/workspace mlops-risk-lab python lab.py train --run-id baseline
docker run --rm -v risk-lab-demo:/app/workspace mlops-risk-lab python lab.py evaluate --run-id baseline
docker run --rm -v risk-lab-demo:/app/workspace mlops-risk-lab python lab.py promote --run-id baseline
docker run --rm -p 127.0.0.1:8000:8000 -v risk-lab-demo:/app/workspace mlops-risk-lab
```

Use a fresh volume name when repeating generation. Restart the serving container
after changing the alias. This image is a learning example and includes the
core test dependencies; it has no production deployment or hardening claim.

Relevant official references:
[sklearn leakage guidance](https://scikit-learn.org/1.7/common_pitfalls.html),
[FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/),
[FastAPI lifespan testing](https://fastapi.tiangolo.com/advanced/testing-events/),
[MLflow local database tracking](https://mlflow.org/docs/latest/ml/tracking/tutorials/local-database/),
[sklearn model persistence](https://scikit-learn.org/1.7/model_persistence.html).
