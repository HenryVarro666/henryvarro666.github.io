---
title: "MLOps learning notes: build, deploy, monitor, and roll back a model"
date: 2026-11-18T00:00:00-06:00
description: "A practical beginner's guide to MLOps, with a downloadable CPU-only lab covering data contracts, leakage, evaluation, lineage, APIs, tests, monitoring, releases, and rollback, plus imaging and RAG applications."
topics: [ml-systems, data-science]
---

A good notebook score answers one question about a prediction system. Can the next data batch run through it? Are production amounts expressed in the same units? Which model and threshold handled this request? Can the previous release be restored? **MLOps turns these questions into recorded, testable operating procedures.**

This guide starts with a small risk-prediction system, includes a downloadable local lab, and then develops the production decisions around it. All lab data are synthetic. Illustrative latency targets, costs, traffic, and business assumptions are labeled separately from measured lab results. The exercise does not estimate real fraud risk or clinical performance. References and version notes were checked on **November 18, 2026**.

## Choose a reading route {#reading-routes}

You should be comfortable writing Python functions, using a terminal and virtual environment, and distinguishing features, labels, training data, and test data. If those concepts are new, work through classification, overfitting, and evaluation in the [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course) first. No GPU, cloud account, Kubernetes, or distributed-training knowledge is required here.

| Your starting point | Suggested route | What to produce |
|---|---|---|
| New to MLOps | Concepts → data and time → lineage → evaluation → lab | Explain what one prediction depends on and restore a model version. |
| Already training models | Start at [local setup](#setup), then study serving, tests, releases, and monitoring | Allow roughly 60–90 minutes after installation for the main exercise. This is a study estimate. |
| Working on a project | Data contracts → releases → monitoring → applications → troubleshooting | A release checklist, alert response, and explicit rollback boundary. |
| Planning structured study | Finish the lab, then follow the [six-stage route](#learning-path) | One inspectable artifact at each stage. |

Build one complete small system, then add tools when a specific problem justifies them. The learning sequence and checklists below are teaching designs, not universal standards.

## 1. Operate the prediction system, including its changing dependencies {#mental-model}

Imagine a payment-risk service. Before authorization, it receives an amount, account age, recent attempt count, and channel. It estimates risk, and a separate rule decides whether the payment enters review. Review capacity is limited, outcomes arrive later, the service has a response deadline, and future releases may change the model or threshold.

Application code is one source of behavior. Training data, feature definitions, label policies, fitted parameters, and thresholds are others. An upstream switch from dollars to cents can break predictions without changing the inference code. Versioning and validation must cover those dependencies.

![A data contract leads through training, evaluation, release approval, and decisions; observed outcomes return to investigation and later work.](mlops-lifecycle.svg "Figure 1: The top row is the delivery path. The lower path represents operational feedback. An alert starts an investigation before anyone decides to retrain or change the system.")

Google's reference architecture describes levels 0, 1, and 2 as manual operation, automated ML pipelines, and CI/CD for pipelines. This is an explanatory maturity model, last reviewed August 28, 2024, rather than a requirement to install every component in its diagram.[Google MLOps architecture](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

### CI, CD, and CT change different things {#ci-cd-ct}

| Term | Responsibility | Example output |
|---|---|---|
| Continuous Integration, CI | Check implementation changes and relevant contracts before integration | Tested feature code and a small pipeline smoke test. |
| Continuous Delivery / Deployment, CD | Prepare or release an identified implementation, runtime, and model package | A release that can be inspected and restored. |
| Continuous Training, CT | Repeat training when its trigger is satisfied | A candidate with evaluation evidence, awaiting release gates. |

Continuous means repeatable automation. It does not mean training must run without interruption or every completed run must replace production. Manual approval can remain while the surrounding work becomes automated.

## 2. Attach the vocabulary to actual objects {#glossary}

| Term | Meaning | In this lab |
|---|---|---|
| Feature / label | Prediction-time input / outcome to predict | Four input fields / a synthetic risk outcome. |
| Run | One execution with inputs, configuration, and results | A training run named `baseline`. |
| Experiment | Related runs addressing a question | Comparing regularization settings. |
| Artifact | A produced file worth retaining | Model, manifest, or evaluation report. |
| Pipeline | Ordered processing steps | Scaling, encoding, classification; the term can also mean the whole training workflow. |
| Lineage | Which inputs and processes produced an output | Data, source, and environment → run → release. |
| Registry | A place that organizes versions and release references | A teaching JSON alias file. |
| Training-serving skew | Differences in feature computation or input distributions between training and serving | Dollars in training, cents at serving. |
| SLI / SLO | A measured service indicator / its chosen target | Valid response ratio / a target for that ratio. |

A training pipeline may read labels and fit transformations. An inference pipeline loads already-fitted transformations and predicts on new inputs. Fitting a new encoder inside a prediction request changes the meaning of its features.

## 3. Define the decision before the model {#requirements}

Write a one-page decision description: who needs the result, at what moment, what action follows, what errors cost, when the result becomes useless, and what happens on failure. Keep a rule or simple model as a baseline so that extra complexity has a comparison point.[Google Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)

All numbers in this table are **illustrative business assumptions**.

| Requirement | Example | Design consequence |
|---|---|---|
| Prediction opportunity | Before each payment authorization | Use information available before that decision. |
| Output and action | Probability, followed by a review threshold | Version the decision rule separately from the score. |
| Response budget | Valid gateway response within 250 ms | Include queueing, features, network, and serialization. |
| Review capacity | At most 500 cases per day | Higher recall cannot assume unlimited reviewers. |
| Failure path | A previously approved fallback rule | Business risk determines whether to allow, deny, or defer. |
| Success evidence | Mature-label quality, loss, false declines, review volume | Offline metrics alone do not establish useful outcomes. |

A model can improve recall while overwhelming reviewers. Keep optimization objectives, predictive metrics, service targets, and business outcomes distinct in the release report.

## 4. A data contract includes units and meaning {#data-contract}

A row represents one prediction opportunity. Training ingestion and API validation use the same four-field contract.

| Field | Accepted values | Meaning |
|---|---|---|
| `amount` | Finite number, 0–100,000 | US dollars. |
| `account_age_days` | Finite number, 0–36,500 | Account age at prediction, in days. |
| `attempts_1h` | Integer, 0–1,000 | Known attempts in the preceding hour. |
| `channel` | `web`, `mobile`, or `store` | Fixed teaching categories. |

A numeric JSON schema cannot detect a dollar-to-cent conversion. Document missing-value policy, time zone, window boundaries, duplicate handling, acceptable delay, and the upstream owner. A fixed example can test both feature values and prediction behavior.

The lab rejects unknown categories, numeric strings, extra fields, and non-finite values. A real application can explicitly map unfamiliar categories to `unknown`, provided training and serving agree and the rate is monitored. Silently dropping unexpected columns makes upstream changes harder to investigate.

## 5. Prevent leakage by asking what was known then {#time-leakage}

Data leakage occurs when model development uses information unavailable at prediction time. Split before fitting scaling, imputation, feature selection, or similar transformations. A sklearn `Pipeline` helps keep fitting inside the training boundary; it cannot repair a feature calculated from the future.[scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)

![Event occurrence, decision time, late data, and outcomes appear on a timeline; training, validation, and test have separate roles.](time-and-leakage.svg "Figure 2: A past event can still be unavailable at the decision. Chronological splitting also needs label-readiness checks and, where relevant, entity separation.")

Distinguish **event time**, when something happened; **available time**, when the system could read it; and **label maturity time**, when observation is complete enough for evaluation. An event at 09:55 that arrives at 10:05 cannot support a reconstructed 10:00 decision. An outcome not yet observed is not automatically negative.

### Match the split to its intended use {#splits}

| Intended generalization | Starting split | Remaining checks |
|---|---|---|
| Similar new samples | Random split | Duplicates, repeated entities, preprocessing leakage. |
| A later period | Chronological or rolling evaluation | Label maturity, availability, seasonality. |
| New patients or customers | Group-disjoint split | Repeated scans or events stay together. |
| New sites or devices | Held-out site or external evaluation | Representativeness and acquisition conditions. |

`GroupKFold` separates groups; `TimeSeriesSplit` constructs future-facing windows. Neither automatically handles all grouping, time, delayed-label, and representation requirements.[scikit-learn cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)

The lab uses chronological 60% / 20% / 20% splits, purging training and validation rows whose labels are not ready at the next boundary. Its **two-hour label delay** is a generator rule, not a general fraud-industry delay.

### Check availability with a few rows {#leakage-exercise}

For a separate synthetic application submitted at time t, consider reported income at submission, income corrected after review, trailing behavior counts, and today's customer status. The first is a candidate input; the correction is future information; trailing counts require a correct window and arrival time; today's status needs historical versions.

Write an earliest-available time for every field. Then trace five evaluation rows back to their raw records. Confirm that each input existed at the corresponding decision. This small audit can expose retrospective corrections and query mistakes that training code alone does not reveal.

## 6. Reproducibility begins with the whole dependency chain {#lineage}

A model file or screenshot cannot identify a release completely. Connect fitted preprocessing, model, input contract, threshold, and evaluation evidence. This also defines what must be restored together.

![Data, code, and runtime identities connect to a complete release bundle; a movable champion alias points to it and the serving process returns a concrete version.](release-lineage.svg "Figure 3: The bundle includes preprocessing, parameters, contracts, thresholds, and evaluation identity. Record the resolved version because champion can move.")

The following is an **illustrative manifest design**, not the lab's exact JSON format:

~~~yaml
run_id: baseline
data:
  snapshot: reference.csv
  sha256: "..."
  split: chronological-60-20-20
  label_delay_hours: 2
implementation:
  git_commit: "..."
  entrypoint: lab.py
  configuration: {C: 1.0, seed: 42}
environment:
  python: "3.12.x"
  dependency_lock: requirements.lock
prediction:
  pipeline_sha256: "..."
  schema_version: "risk-v1"
  threshold: "selected on validation, then frozen"
evaluation:
  report: evaluations/baseline.json
  metric: average_precision_score
~~~

Provenance means knowing where an artifact came from. Rerunnability means the procedure executes again. Reproducibility means results match within a defined tolerance. Bitwise equality is stronger. A seed helps, but does not guarantee equality across framework versions, platforms, or CPU/GPU execution.[PyTorch reproducibility](https://docs.pytorch.org/docs/2.15/notes/randomness.html)

A dataset name or digest is not a data backup. Start with retained, non-overwritten files and checksums. DVC becomes useful when large changing data must move with Git-based work: Git stores pointers, while cache and remote storage hold the data. Pushing Git does not push those objects.[DVC getting started](https://doc.dvc.org/start)

## 7. Evaluate ranking, decisions, and probabilities separately {#evaluation}

A classifier estimates a score; a threshold turns it into an action. Select the threshold on validation data, freeze it, and reserve final test data for evidence independent of that selection.[scikit-learn threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html)

| Question | Useful measures | Common mistake |
|---|---|---|
| Are positives ranked higher? | ROC-AUC, Average Precision (AP) | Good ranking does not select an operating threshold. |
| What happens at this threshold? | Precision, recall, confusion matrix, alert volume | Accuracy can hide missed positives. |
| Are probabilities meaningful? | Brier score, calibration plots | Ranking can stay unchanged while probabilities miscalibrate. |
| Which groups struggle? | Channel, site, or other relevant slices | Small slices need counts and uncertainty. |
| Is the action useful? | Cost, capacity, user outcomes | Offline association does not prove intervention benefit. |

Precision = TP / (TP + FP), and recall = TP / (TP + FN). TP is a correctly identified positive, FP a false alert, and FN a missed positive.

This lab reports AP from `average_precision_score`, which weights precision by recall increments. Trapezoidal integration of a precision-recall curve is a different calculation. Name the computation rather than treating every “PR-AUC” as interchangeable. Also distinguish metrics requiring scores from those requiring hard predictions.[scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)

State the positive class, population, window, sample count, label coverage, and threshold. A score without those conditions is difficult to compare. See [precision and recall]({{< relref path="/notes/precision-vs-recall" lang="en" >}}) and [ROC-AUC versus PR-AUC]({{< relref path="/notes/roc-auc-vs-pr-auc" lang="en" >}}).

### Explain a threshold using a decision table {#threshold-exercise}

These are **invented numbers**, separate from the lab: 1,000 cases, including 50 positives; each miss costs \$10 and each false review costs \$1.

| Rule | TP / FP / FN / TN | Precision / recall | Reviews | Assumed cost |
|---|---|---|---:|---:|
| Higher threshold | 30 / 20 / 20 / 930 | 0.60 / 0.60 | 50 | \$220 |
| Lower threshold | 40 / 80 / 10 / 870 | 0.333 / 0.80 | 120 | \$180 |

The lower threshold reduces assumed cost, but fails a capacity limit of 50 reviews. Actual costs may depend on amount, user, and intervention. Write the assumptions before comparing rules.

Calibration asks another question: among 100 cases assigned probabilities near 0.8, are roughly 80 positive? That is an illustration, not a guarantee for an individual. Calibration bins and slices need sufficient data; report their counts.

## 8. Run the local system first {#setup}

[Download the complete MLOps risk lab ZIP](/notes/mlops-learning-guide/mlops-risk-lab.zip), extract it, and enter `mlops-risk-lab`. You need Python 3.12, a CPU, and local storage. Installation needs internet access; execution needs no account or external dataset.

~~~bash
cd mlops-risk-lab
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python lab.py generate
python lab.py train --run-id baseline
python lab.py evaluate --run-id baseline
python lab.py promote --run-id baseline
python lab.py predict --json '{"amount":120.0,"account_age_days":90.0,"attempts_1h":2,"channel":"web"}'
python -m pytest -q
~~~

In Windows PowerShell, create the environment with `py -3.12 -m venv .venv` and activate it using `.\.venv\Scripts\Activate.ps1`. Calling the environment's Python directly is another option when activation policy prevents scripts.

The measured environment was **macOS arm64, Python 3.12.16**. Main pins are sklearn 1.7.2, pandas 2.3.3, NumPy 2.3.4, SciPy 1.16.3, FastAPI 0.121.1, Uvicorn 0.38.0, and pytest 8.4.2. The lock pins transitive versions, not installation artifact hashes. These are tested versions, not claims about the newest releases. Linux and Windows execution was not verified.

Outputs default to `workspace/`. Put the global option before each subcommand to use another root: `python lab.py --root workspace-two generate`. Keep that root consistent for all later commands.

## 9. Inspect artifacts, then release a second candidate {#lab}

| Command | Responsibility |
|---|---|
| `generate` | Create reference and shifted synthetic data with event and label-availability metadata. |
| `train` | Fit preprocessing and classifier, choose a validation threshold, save a run bundle. |
| `evaluate` | Produce fixed evaluation evidence and a release-gate result. |
| `promote` | Point production at a specific passing bundle. |
| `predict` | Return probability, threshold, version, and bundle digest. |
| `monitor` | Compare inputs, scores, and mature-label quality. |
| `rollback` | Restore the recorded previous release. |

~~~text
workspace/
  data/reference.csv
  data/shifted.csv
  runs/baseline/
    model.joblib
    manifest.json
    environment.json
    lab.py
    api.py
    requirements.lock
  evaluations/baseline.json
  registry/production.json
  reports/monitor-....json
~~~

Offline prediction and the API load the same fitted sklearn Pipeline. Commands refuse to overwrite runs and evaluation reports. This is a program convention, not enforced immutable storage. SHA-256 detects content changes; it does not authenticate a publisher.

The JSON registry assumes one trusted writer. It has no multi-writer coordination, permissions, approval system, or durable release history. Evaluation verifies data and split identity, including test ID hashes and counts: hashing the dataset alone would not freeze a changed split policy.

The training core can be reduced to the following. The complete program additionally validates data, constructs historical splits, chooses the threshold, and writes evidence.

~~~python
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

features = ["amount", "account_age_days", "attempts_1h", "channel"]
preprocess = ColumnTransformer([
    ("numeric", StandardScaler(), features[:3]),
    ("categorical", OneHotEncoder(handle_unknown="error"), ["channel"]),
])
pipeline = Pipeline([
    ("features", preprocess),
    ("classifier", LogisticRegression(C=1.0, max_iter=500, random_state=42)),
])
pipeline.fit(training[features], training["label"])
probabilities = pipeline.predict_proba(validation[features])[:, 1]
~~~

Here, `training` and `validation` are already-split DataFrames. Only training calls `fit`; later stages reuse the fitted object. Under this binary label convention, column 1 contains positive-class probabilities.

### Measured reference results {#measured-results}

These are rounded measurements from the locked lab, with synthetic positive prevalence near 20%.

| Measure | Baseline |
|---|---:|
| Training rows after label purge | 3,481 |
| Validation rows after label purge | 1,081 |
| Test rows | 1,200 |
| Test prevalence | 0.1975 |
| Validation-selected threshold | 0.20 |
| Test AP | 0.435033 |
| Test ROC-AUC | 0.756225 |
| Precision / recall | 0.397403 / 0.645570 |
| F1 / Brier score | 0.491961 / 0.136691 |
| TN / FP / FN / TP | 731 / 232 / 84 / 153 |
| Teaching release gate | Pass |

The sample input returns probability about **0.311307**, so `flagged` is true at threshold 0.20. Responses include `model_version`, `bundle_sha256`, `threshold`, and `synthetic_only`. Inspect your own files; numerical differences require checking dependencies, data, configuration, and splits before changing gates.

### Promote and restore another version {#rollback-lab}

~~~bash
python lab.py train --run-id candidate --C 0.2
python lab.py evaluate --run-id candidate
python lab.py promote --run-id candidate
python lab.py predict --json '{"amount":120.0,"account_age_days":90.0,"attempts_1h":2,"channel":"web"}'
python lab.py rollback
python lab.py predict --json '{"amount":120.0,"account_age_days":90.0,"attempts_1h":2,"channel":"web"}'
~~~

For logistic regression, smaller `C` means stronger regularization. The candidate selects threshold about 0.17. Its reference AP is 0.435109 and recall 0.704641, while F1 is 0.469101, below baseline's 0.491961. This rehearses release mechanics, not overall superiority.

Both candidates inspect the same test set for this synthetic workflow exercise. They are not fresh independent final estimates at every iteration. Real selection belongs on validation or rolling evaluation, with an untouched final holdout. One report per run does not prevent test-based tuning across many runs.

Compare returned versions and digests after rollback. A running API retains its loaded model until restart.

To observe refusal:

~~~bash
python lab.py train --run-id blocked
python lab.py evaluate --run-id blocked --min-average-precision 0.999
python lab.py promote --run-id blocked
~~~

Promotion should exit 2. Evaluation exits 0 when report creation succeeds even if `gate_passed` is false; automation must inspect that field. Default AP ≥ 0.25, ROC-AUC ≥ 0.65, and recall ≥ 0.45 are teaching gates, without validated business meaning.

## 10. Tracking records experiments; a registry references versions {#tracking}

Tracking records runs, parameters, metrics, and artifacts. A Model Registry manages registered versions and release references. Recording an experiment does not activate its model.

The core exercise has no MLflow dependency. Add the optional tracking path with:

~~~bash
python -m pip install -r requirements-mlflow.txt
python lab.py train --run-id tracked --mlflow
~~~

The tested extras pin **mlflow-skinny 3.6.0, SQLAlchemy 2.0.44, and Alembic 1.17.1**, using an explicit SQLite URI. New local defaults changed to SQLite in MLflow 3.7.0; older tutorials and existing file stores can behave differently.[MLflow self-hosting](https://mlflow.org/docs/latest/self-hosting/)

Inspect local records without a server:

~~~python
from pathlib import Path
from mlflow import MlflowClient

database = (Path("workspace") / "mlflow.db").resolve()
if not database.exists():
    raise FileNotFoundError("Train with --mlflow and check the output root")
client = MlflowClient(tracking_uri=f"sqlite:///{database}")
for experiment in client.search_experiments():
    runs = client.search_runs([experiment.experiment_id])
    print(experiment.name, [(run.info.run_id, run.data.metrics) for run in runs])
~~~

Keep the working directory and root consistent. Artifact copies live under `workspace/mlartifacts/`; a receipt links the local run to MLflow. Dataset metadata and digests do not archive every data byte. This client workflow includes no UI, server, or MLflow Model Registry.

For a later registry integration, `models:/risk-model/7` identifies a version and `models:/risk-model@champion` resolves an alias. Reassigning an alias affects the next load, not an already-loaded object. Record the resolved identity. Model Stages have been deprecated since 2.9.0; use current alias workflows.[MLflow Registry workflow](https://mlflow.org/docs/latest/ml/model-registry/workflow/)

## 11. An API needs input, output, and loading contracts {#serving}

After promotion, start the service:

~~~bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --no-access-log
~~~

From another terminal in the same project and environment:

~~~bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"amount":120.0,"account_age_days":90.0,"attempts_1h":2,"channel":"web"}'
~~~

Liveness asks whether the process responds. Readiness asks whether a compatible active bundle has loaded. Prediction validates the request and uses that loaded object. None establishes predictive quality by itself.

FastAPI lifespan supports loading shared resources at startup, instead of reading the model on every request.[FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/)

Measured checks returned 200 for readiness and valid prediction, and 422 for a negative amount. Without an active bundle, liveness remains 200 while readiness and prediction return 503. After promotion or rollback, restart and check the concrete returned version.

For a custom root, start with `MLOPS_LAB_ROOT=/tmp/risk-demo-2 python -m uvicorn api:app --host 127.0.0.1 --port 8000 --no-access-log`. In PowerShell, set the variable using `$env:MLOPS_LAB_ROOT`.

Production also needs request-size limits, timeouts, authentication, concurrency policy, and logging rules. `async def` does not make CPU work automatically parallel. Workers generally have separate model memory; measure memory and throughput before increasing their count.

## 12. Organize tests by risk and cost {#testing}

The following layers are a teaching recommendation. The original 2017 ML Test Score paper proposed 28 production-oriented testing and monitoring needs, supplementing ordinary software tests. Its score is not model accuracy or a certification.[ML Test Score paper](https://storage.googleapis.com/gweb-research2023-media/pubtools/4156.pdf)

| Layer | Failures to catch | When |
|---|---|---|
| Fast code and contracts | Missing fields, units, invalid categories, non-finite values | Every relevant change. |
| Small integration | Train/save/load/predict failures and offline/API disagreement | Every pipeline change. |
| Candidate evaluation | Failed quality gates, poor slices, excessive review volume | Before promotion. |
| Deployment smoke tests | Missing bundles, failed readiness, wrong version | Each deployment. |
| Operational checks | Stale inputs, latency, mature-label decline, broken recovery | Continuously or in scheduled drills. |

Run `python -m pytest -q`. The measured core result was **21 passed, 1 skipped**; the skipped test needs optional MLflow dependencies. With those installed, **22 passed**.

Ask which mistake a test prevents. A file-existence test is weaker than a load-and-predict check. HTTP 200 is weaker than verified identity and prediction parity. Prioritize temporal boundaries, invalid inputs, serialization, feature consistency, and recovery rather than mechanical assertions to raise a coverage number. Deliberately break a contract once and confirm that its test fails.

## 13. Package the runtime and preserve delivery identity {#packaging}

Docker packages an application and runtime dependencies. It does not evaluate predictive quality or automatically supply scaling and operations. The download includes an **optional, unbuilt and untested Dockerfile**. It installs dependencies and copies named source files; training happens separately in the container environment.

These are extension commands for the reader to verify, not measured results:

~~~bash
docker build -t mlops-risk-lab .
docker run --rm -v risk-lab-demo:/app/workspace mlops-risk-lab python lab.py generate
docker run --rm -v risk-lab-demo:/app/workspace mlops-risk-lab python lab.py train --run-id baseline
docker run --rm -v risk-lab-demo:/app/workspace mlops-risk-lab python lab.py evaluate --run-id baseline
docker run --rm -v risk-lab-demo:/app/workspace mlops-risk-lab python lab.py promote --run-id baseline
docker run --rm -p 127.0.0.1:8000:8000 -v risk-lab-demo:/app/workspace mlops-risk-lab
~~~

Use a fresh volume name to repeat generation, and restart serving after alias changes. This image includes test dependencies and makes no production-hardening claim.

Copy dependency files before application code for caching and use exec-form `CMD` for signal handling. FastAPI's container guide builds from a Python image; its old prebuilt FastAPI/Gunicorn base image is deprecated.[FastAPI containers](https://fastapi.tiangolo.com/deployment/docker/)

CI should check an identified change and produce an identified package that deployment reuses. The download's inactive `ci.example.yml` has not run on GitHub; its constituent local checks were executed. It explicitly checks `gate_passed`. Do not retrain casually on the deployment host and assume the result is the reviewed artifact.

Workflow artifacts and container-registry images are different objects. GitHub's v4 workflow artifacts are immutable, but an automatic download digest mismatch currently produces a warning, not a hard failure. Enforce a failing comparison if integrity is a release gate.[GitHub workflow artifacts](https://docs.github.com/en/actions/tutorials/store-and-share-data)

## 14. Choose serving patterns from decision timing {#serving-patterns}

These patterns can coexist; they are not a maturity ranking.

| Pattern | Example | Operational contract |
|---|---|---|
| Batch inference | Overnight review list | Deadline, completeness, freshness, safe reruns. |
| Synchronous API | A payment flow waits for a score | End-to-end timeout, tail latency, fallback. |
| Asynchronous inference | A large 3-D processing job | Job ID, queue age, retries, idempotency. |
| Event-stream inference | Continuous transaction-window features | Event time, lateness, deduplication, state, lag. |

Online serving does not imply online learning. A fixed model trained yesterday can answer each request. Event streaming is also different from LLM token streaming: one handles ongoing events and state; the other progressively returns one answer.

Retrying a prediction can repeat associated side effects. Use a durable idempotency key where work writes a payment, notification, or review action. Exactly-once stream guarantees have a defined boundary and do not automatically cover arbitrary external APIs.[Kafka Streams concepts](https://kafka.apache.org/44/streams/core-concepts/)

## 15. Release through gates and restore a compatible whole {#release}

Offline improvement is one release condition. Check input compatibility, runtime, system performance, business limits, and recovery. Retain the predecessor and specify approval, expansion, and stop conditions before exposing traffic.

![A candidate passes release gates and staged rollout before expansion; an anomaly restores the previous compatible package and routing.](safe-release.svg "Figure 4: Passing offline checks and expanding live traffic are separate decisions. Shadow, canary, and A/B answer different questions; failure recovery restores the whole compatible release.")

| Method | Does the candidate affect users? | Main question | Limitation |
|---|---|---|---|
| Shadow | Its response is ignored | Can it serve live-shaped requests, and how does it differ? | No candidate treatment effect is observed merely by mirroring. |
| Canary | Yes, for limited exposure | Is expansion safe? | Short windows can miss rare or delayed failures. |
| A/B experiment | Yes, by assignment | Does a declared outcome improve? | Routing weights alone do not establish valid causal evidence. |

Shadow systems must suppress real side effects. Canary rollout keeps a recovery path when operational alarms fire. A/B evaluation also needs appropriate assignment, exposure records, duration, sample size, and guardrails.[AWS shadow tests](https://docs.aws.amazon.com/sagemaker/latest/dg/shadow-tests.html) · [AWS canary rollout](https://docs.aws.amazon.com/sagemaker/latest/dg/deployment-guardrails-blue-green-canary.html) · [A/B testing notes]({{< relref path="/notes/ab-testing" lang="en" >}})

### Write an SLO that can be evaluated {#slo}

An **illustrative** target is: over 28 days, 99.5% of eligible requests return a valid gateway response within 250 ms. Define eligible, valid, timeout, and fallback treatment first. The SLI is the observed ratio; the SLO is its target. A ratio target's error budget is one minus the target.[Google SRE: implementing SLOs](https://sre.google/workbook/implementing-slos/)

p95 means approximately 95% of observed latencies fall at or below that value in the chosen window. Report traffic, input mix, warm/cold state, and errors. One timed request cannot establish sustainable QPS. Include queueing, features, preprocessing, execution, and the response path.

Rollback restores the compatible **model, preprocessing, schema/features, threshold, and runtime**. An alias cannot undo a destructive feature migration. After restoration, verify readiness, version, representative requests, and operating signals.

### Measure capacity and cost within declared boundaries {#capacity}

As an extension, fix the input set and compare cold start, warm single-request behavior, sustained concurrency, and bursts. Record successes, errors, duration, successful-request latency percentiles, queue delay, and peak memory. No production QPS or latency claim is provided by this lab.

If feature retrieval dominates, more inference workers may not help. A growing queue can invalidate a short successful test. Locate the measured bottleneck before changing hardware or concurrency.

For **illustrative arithmetic**, \$2 per hour divided across 40,000 usable results is \$0.05 per 1,000. If only half finish usefully, that becomes \$0.10. State whether idle capacity, feature queries, storage, network, retries, and shadow work are included.

## 16. Monitor five distinct layers {#monitoring}

![Immediate service signals appear above request-data signals and, later, mature-label quality and cost.](monitoring-layers.svg "Figure 5: Moving downward generally requires more time before evidence is complete. Input changes and quality deterioration can occur independently.")

| Layer | Indicators | First question |
|---|---|---|
| System | Traffic, errors, timeouts, percentiles, queue age, saturation | Can users obtain timely usable results? |
| Data contract | Missingness, ranges, categories, units, freshness, completeness | Did upstream deliver the intended inputs? |
| Prediction behavior | Score distribution, positive/abstention rates, slices | Did model actions change? |
| Labeled quality | AP, recall, calibration, slices, coverage, maturity | Is performance worse, and for whom? |
| Business and economics | Loss, false declines, review volume, completion, cost | Is the system achieving its purpose? |

Latency, traffic, errors, and saturation are a useful starting point for service monitoring. Separate successful/error latency and inspect tails rather than relying on a mean.[Google SRE monitoring](https://sre.google/sre-book/monitoring-distributed-systems/)

Record a stable `prediction_id`, prediction time, resolved model, threshold, and feature identity, then join later labels. Evaluate cohorts by prediction time and maturity, rather than treating labels arriving this week as this week's model performance. Monitor join coverage: selectively available labels can bias results.

Under an **invented seven-day policy**, Monday's predictions become mature next Monday. Friday's unknown outcomes stay unknown. Separate recent service/input reports from mature-cohort quality and retain label versions when corrections arrive.

Each alert needs a window, threshold, owner, and first action. Unknown-category alerts should lead to source records and a decision about affected traffic.

The lab's `python lab.py monitor` defaults to `workspace/data/shifted.csv`. The reference batch triggers data, score, and quality alerts; AP is about **0.194730** and ROC-AUC **0.359483**. Aggregate heuristics are not formal drift hypothesis tests. The generator intentionally changes both inputs and their relationship to outcomes, so this demonstration does not show that input drift caused degradation.

Monitoring defaults to a fixed synthetic `--as-of 2025-02-01T00:00:00Z`. It normalizes UTC, excludes future events, and uses labels ready by that snapshot. Fewer than 100 mature rows, only one class, or missing labels produce an explicit unavailable status and null performance. Try `python lab.py monitor --as-of 2025-01-10`: one observed event and no mature labels cannot establish quality.

## 17. Drift is evidence for investigation {#drift}

Terminology varies. Here, input drift means a change in `P(X)`, concept change concerns `P(Y|X)`, and prior change concerns `P(Y)`. Input statistics cannot substitute for measured labeled performance.

Consider two **mathematical constructions**, with binary x and fixed model `f(x)=x`:

| Change | Accuracy |
|---|---|
| Keep `y=x`, change `P(x=1)` from 0.5 to 0.9 | Still 1 despite input drift. |
| Keep `P(x)`, change the relationship to `y=1-x` | Falls from 1 to 0 without input drift. |

Input drift is therefore neither sufficient nor necessary for deterioration. PSI, KS, and distribution-distance thresholds need window, sample-size, seasonality, segment, and false-alarm policies.

Investigate instrumentation and freshness, then schema and pipeline incidents, then composition and mature labels. The response might be input repair, rollback, an approved rule change, candidate training, additional labeling, or observation. A holiday mix change and a cents/dollars bug require different actions.

## 18. Retraining produces a candidate; promotion is another decision {#retraining}

Triggers can include mature labels arriving, a schedule, investigated deterioration, or feature/label revisions. Record the trigger, snapshot, and evaluation population. An upstream column becoming empty should not automatically teach the model a supposed new reality.

Make the steps explicit: await inputs → validate → snapshot → train → evaluate → release checks → approve or reject → controlled deployment. Completed computation does not establish a useful candidate.

Before automation, rehearse interrupted training, rerunning a batch without duplicate release, and failed evaluation while the current model continues serving. Retain failed evidence so that the run history does not conceal rejected work.

## 19. Add orchestration and feature storage when their problems appear {#orchestration}

An orchestrator helps when recurring jobs depend on several upstream tasks and need retries, backfills, deadlines, and execution records. Give each task inputs, outputs, idempotency rules, and failure behavior. Scheduling does not define correct labels or remove leakage. Compare [Airflow]({{< relref path="/notes/airflow" lang="en" >}}) and the [dbt lab]({{< relref path="/notes/dbt-learning-guide" lang="en" >}}).

A feature store becomes useful when shared historical and online retrieval are difficult to keep consistent. A point-in-time join retrieves values valid for a historical prediction. A 10:00 sample must not use the latest 11:00 account total. Feast's historical retrieval looks backward from each entity timestamp and applies a TTL.[Feast point-in-time joins](https://docs.feast.dev/getting-started/concepts/point-in-time-joins)

Availability still matters. A 09:55 event arriving at 10:10 was unavailable at 10:00. Incorrect or overwritten timestamps cannot be repaired by adopting a tool. Define semantics, materialization freshness, and compatible rollback paths first.

## 20. Apply the lifecycle to different domains {#applications}

### Tabular fraud: feedback is delayed and selected {#fraud}

Fraud research describes rapid investigator feedback on selected alerts alongside broader delayed outcomes. Their selection processes differ.[Dal Pozzolo et al., delayed supervised information](https://boracchi.faculty.polimi.it/docs/2015_04_Credit_Card_Fraud_Detection_DalPozzolo_Boracchi_Caelen_Alippi_Bontempi.pdf)

Separate recent review precision from mature-cohort quality. Record thresholds and capacity, define when an unconfirmed outcome becomes negative, and recognize that blocked transactions can hide counterfactual outcomes. The synthetic lab teaches mechanics without modeling that complete causal setting.

### Imaging: preserve patient, acquisition, and preprocessing boundaries {#imaging}

A 3-D MRI segmentation research pipeline should identify sequences, orientation, spacing, resampling, intensity processing, cropping, label definitions, and postprocessing. Split by patient before distributing scans or slices. Add site, device, time, or external validation according to the intended question.

Operational checks include file integrity, dimensions, preprocessing failures, and queue-to-result time. Quality needs labeled cases and task-appropriate metrics. Voxel accuracy can hide poor small-structure segmentation; define Dice or surface measures, case aggregation, and reference-label quality. Use representativeness, independent testing, relevant conditions, and ongoing monitoring as design questions, then read the FDA introduction linking to the final 2025 IMDRF principles.[FDA GMLP overview and IMDRF document](https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles)

This is a research-pipeline example. It does not establish clinical suitability, intended-use validation, workflow integration, or change authorization.

### LLMOps and RAG: version knowledge and prompts too {#llmops}

A RAG release includes the generator, prompt, parsing/chunking, embedding model, corpus/index snapshot, reranker, access policy, and evaluation set. Changing embedding spaces can require rebuilding a compatible index. Restoring only the prompt may not restore previous behavior.

Separate failures: parsing lost a table; retrieval chose the wrong source; truncation removed evidence; generation produced an unsupported answer. Tracing intermediate request steps helps locate these failures.[MLflow tracing](https://mlflow.org/docs/latest/genai/tracing/)

Evaluate retrieval, source correctness and freshness, answer support, and task completion separately. Groundedness asks whether provided context supports the answer; it does not prove that context is true, current, or authorized. Calibrate LLM judges against human examples and record their configuration.[MLflow RetrievalGroundedness](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/rag/groundedness)

Monitor total usable-answer time, time to first token, errors/retries, tokens, cost, and sampled quality. Fast first tokens do not establish fast completion. Filter sensitive traces. The [LangChain and LangGraph notes]({{< relref path="/notes/langchain-langgraph-guide" lang="en" >}}) offer a related application route.

## 21. Make security constraints concrete {#security}

Load Python-serialized models only from trusted sources. `pickle` and `joblib` loading can execute code; a checksum identifies bytes but does not prove safety.[scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html) Use controlled build sources and least-necessary runtime access.

Inputs, identifiers, prompts, retrieved documents, and labels can be sensitive. Avoid broad logging and artifact uploads. Prefer request IDs, versions, necessary aggregates, and authorized sanitized samples, with explicit access and retention policies.

Give CI tokens minimal permissions, prefer short-lived deployment identity, and keep secrets out of source. Treat privileged triggers and untrusted code carefully. Log redaction is not guaranteed; verify and pin workflow dependencies.[GitHub Actions secure use](https://docs.github.com/en/actions/reference/security/secure-use)

## 22. Diagnose the layer that evidence identifies {#troubleshooting}

| Symptom | Inspect first | Likely action |
|---|---|---|
| Good offline results, strange API output | Same-row fields, units, transforms, bundle identity | Restore the shared fitted pipeline or fix the contract. |
| Readiness fails | Promotion, root, artifact completeness | Evaluate/promote, then correct paths and permissions. |
| Old version after promotion | Startup time, returned version, alias | Restart or perform a controlled rollout. |
| Cross-machine score differences | Dependencies, data, randomness, hardware | Restore the environment and state tolerances. |
| Drift alert, stable quality | Mature labels, seasonality, slices, window size | Recalibrate monitoring or observe. |
| Stable inputs, worse quality | Label policy, conditional relationship, join coverage | Investigate concept or label changes. |
| Successful training, worsening production | Gates, test reuse, thresholds | Repair evaluation and promotion policy. |
| Idle compute, slow results | Queue, feature calls, serialization, retries | Optimize the measured bottleneck. |
| Successful batch, missing results | Input manifest, row/key coverage, publication | Add completeness and safe-commit checks. |
| Rollback still fails | Feature/schema, preprocessing, threshold, runtime | Restore the compatible whole release. |

Keep a short incident record: symptom, impact, evidence, cause, action, and the new check. The next reader should understand why the test exists.

## 23. Organize study around six deliverables {#learning-path}

This is a flexible **four-to-six-week study estimate**, not a completion or employment promise. Combine stages if you know ML; stretch them if weekly time is limited.

| Stage | Focus | Exit artifact |
|---|---|---|
| 1. Data and model basics | Labels, baselines, splits, leakage, metrics | A contract, split diagram, and AP/threshold explanation. |
| 2. Repeatable local training | CLI, lock, digests, pipeline | A fresh-directory run and saved-model check. |
| 3. Comparison and lineage | Candidates, evaluation, tracking | A versioned report with a selection rationale. |
| 4. Inference and tests | API or batch, validation, parity | Valid/invalid requests and a deliberately failing test. |
| 5. Release and recovery | Gates, versions, controlled rollout | Promotion and rollback with verified outputs. |
| 6. Operations and extension | Monitoring, labels, scheduling, domain risks | An injected-incident report and your own design. |

Then replace synthetic data with a suitably licensed public dataset while preserving a baseline and frozen evaluation. Choose one justified serving pattern and one actionable alert before adding cloud infrastructure or a cluster.

## 24. Choose resources by the exercise they enable {#resources}

Public course material does not imply free cloud compute. Older lessons can teach useful design while their package APIs need adaptation. Choose the resource that addresses your current stage.

| Resource | Best use | Exercise |
|---|---|---|
| [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course) | Classification, overfitting, evaluation | Explain a confusion matrix and train a baseline. |
| [Google MLOps architecture](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) | Implementation delivery versus data-driven training | Draw manual steps and approval points. |
| [sklearn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) | Preprocessing consistency and leakage | Repair an all-data scaler fit. |
| [ML Test Score](https://storage.googleapis.com/gweb-research2023-media/pubtools/4156.pdf) | Production-oriented checks from the 2017 paper | Choose three relevant checks and implement one. |
| [DVC getting started](https://doc.dvc.org/start) | Pointers, cache, remote storage | Restore matching older data and code. |
| [Made With ML](https://madewithml.com/) | Connected design, data, testing, production | Turn a notebook path into scripts and tests. |
| [MLOps Zoomcamp](https://datatalksclub.github.io/docs/courses/mlops-zoomcamp/) | Tracking, deployment, orchestration, monitoring | Reproduce one module, then change the dataset. |
| [Full Stack Deep Learning 2022](https://fullstackdeeplearning.com/course/2022/) | DL data, tests, and delivery tradeoffs | Port a testing or deployment lab deliberately. |
| [Stanford CS329S](https://stanford-cs329s.github.io/) | Requirements-led system design | Write a one-page design with failure tradeoffs. |
| [Designing Machine Learning Systems, author's page](https://huyenchip.com/books/) | Data, retraining, and monitoring decisions; commercial book | Review your design using one chapter's questions. |

As checked on November 18, 2026, Zoomcamp is self-paced with no live cohort planned for 2026; historical certificate information is not a current certificate promise. FSDL is the 2022 course. Public CS329S material does not confer enrollment, credit, or a certificate.

For a final exercise, use a fresh directory, regenerate and train, submit an invalid request, promote and restore a version, identify the actual loaded bundle, and explain the shifted-batch alerts. Save the commands, outputs, and reasoning as your own experiment record.
