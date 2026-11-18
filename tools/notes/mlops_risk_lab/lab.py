"""A local, synthetic MLOps teaching lab. No real fraud estimates or services."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
from typing import Literal
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["amount", "account_age_days", "attempts_1h"]
FEATURES = NUMERIC + ["channel"]
CHANNELS = ["web", "mobile", "store"]
SCHEMA_VERSION = "risk-v1"
SPLIT_POLICY_VERSION = "chronological-60-20-20-label-ready-v1"
SNAPSHOT_AS_OF = "2025-02-01T00:00:00Z"
RUNTIME_PACKAGES = ["scikit-learn", "numpy", "scipy", "pandas", "joblib", "pydantic"]


class Features(BaseModel):
    """The same strict contract guards CSV ingestion, CLI, and HTTP inputs."""
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    amount: float = Field(ge=0, le=100_000)
    account_age_days: float = Field(ge=0, le=36_500)
    attempts_1h: int = Field(ge=0, le=1000)
    channel: Literal["web", "mobile", "store"]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Atomic replacement on one local filesystem; this lab assumes one writer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES}}


def generate(root):
    """Known toy process; IDs are independent synthetic event keys, not people."""
    root = Path(root)
    rng = np.random.default_rng(42)
    paths = []
    for name, size, shift, start in [("reference", 6000, False, "2025-01-01"),
                                      ("shifted", 1500, True, "2025-01-10")]:
        amount = np.round(np.clip(rng.lognormal(4.2 + 0.9 * shift, 0.8, size), 1, 10_000), 2)
        age = np.round(rng.exponential(300 if not shift else 90, size), 2)
        attempts = rng.poisson(1.5 + 2 * shift, size)
        channel = rng.choice(CHANNELS, size, p=[0.45, 0.45, 0.10] if not shift else [0.8, 0.15, 0.05])
        logit = -3.0 + 0.8 * np.log1p(amount / 50) - 0.002 * age + 0.55 * attempts
        logit += 0.5 * (channel == "web")
        # The second batch also has deliberate concept change, independent of drift.
        logit = logit if not shift else -1.0 - 0.4 * logit
        label = rng.binomial(1, 1 / (1 + np.exp(-logit)))
        times = pd.date_range(start, periods=size, freq="min", tz="UTC")
        data = pd.DataFrame({"event_id": [f"{name}-{i:06d}" for i in range(size)],
                             "event_time": times,
                             "label_ready_time": times + pd.Timedelta(hours=2),
                             "amount": amount, "account_age_days": age,
                             "attempts_1h": attempts, "channel": channel, "label": label})
        path = root / "data" / f"{name}.csv"
        if path.exists():
            raise ValueError(f"Refusing to overwrite {path}; use a fresh --root")
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)
        paths.append(str(path))
    return {"seed": 42, "files": paths, "synthetic_only": True}


def read_data(path, *, allow_unlabeled=False):
    data = pd.read_csv(path)
    metadata = {"event_id", "event_time"}
    labeled = "label" in data.columns
    expected = set(FEATURES) | metadata | ({"label", "label_ready_time"} if labeled else set())
    if set(data.columns) != expected or (not labeled and not allow_unlabeled):
        raise ValueError("Unexpected/missing columns; future outcomes must never become features")
    if len(data) < 2 or data.event_id.isna().any() or not data.event_id.is_unique:
        raise ValueError("Rows need unique non-null event IDs; duplicate IDs risk split overlap")
    data["event_time"] = pd.to_datetime(data.event_time, utc=True, errors="raise")
    if data.event_time.isna().any() or not data.event_time.is_unique or not data.event_time.is_monotonic_increasing:
        raise ValueError("event_time must be unique, non-null, and chronologically sorted")
    if labeled:
        if data.label.isna().any() or not data.label.isin([0, 1]).all():
            raise ValueError("label must be binary and complete")
        data["label_ready_time"] = pd.to_datetime(data.label_ready_time, utc=True, errors="raise")
        if data.label_ready_time.isna().any() or (data.label_ready_time < data.event_time).any():
            raise ValueError("A label cannot be ready before its event")
    for row in data[FEATURES].to_dict(orient="records"):
        try:
            Features.model_validate(row)
        except ValidationError as error:
            raise ValueError("Feature schema violation") from error
    return data


def split_data(data):
    if len(data) < 1000:
        raise ValueError("At least 1000 rows required for this toy chronological split")
    first, second = int(0.6 * len(data)), int(0.8 * len(data))
    train = data.iloc[:first]
    validation = data.iloc[first:second]
    test = data.iloc[second:]
    # Fit/tune only with labels available at each historical decision boundary.
    train = train[train.label_ready_time <= validation.event_time.min()]
    validation = validation[validation.label_ready_time <= test.event_time.min()]
    test = test[test.label_ready_time <= pd.Timestamp(SNAPSHOT_AS_OF)]
    splits = (train, validation, test)
    if any(len(part) < 100 or part.label.nunique() < 2 for part in splits):
        raise ValueError("Too few label-ready rows or both classes missing")
    for earlier, later in [(train, validation), (validation, test), (train, test)]:
        if not set(earlier.event_id).isdisjoint(later.event_id):
            raise ValueError("Split overlap")
        if earlier.event_time.max() >= later.event_time.min():
            raise ValueError("Temporal split overlap")
        if earlier.label_ready_time.max() > later.event_time.min():
            raise ValueError("Future label leakage")
    return splits


def metrics(labels, probabilities, threshold):
    predicted = probabilities >= threshold
    return {"n": int(len(labels)), "prevalence": float(np.mean(labels)),
            "average_precision": float(average_precision_score(labels, probabilities)),
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "precision": float(precision_score(labels, predicted, zero_division=0)),
            "recall": float(recall_score(labels, predicted, zero_division=0)),
            "f1": float(f1_score(labels, predicted, zero_division=0)),
            "brier": float(brier_score_loss(labels, probabilities)),
            "confusion_matrix_tn_fp_fn_tp": confusion_matrix(labels, predicted, labels=[0, 1]).ravel().tolist()}


def reference_summary(data, probabilities):
    return {"numeric": {name: {"mean": float(data[name].mean()), "std": float(data[name].std(ddof=0))}
                         for name in NUMERIC},
            "channel_frequency": {name: float((data.channel == name).mean()) for name in CHANNELS},
            "score_mean": float(np.mean(probabilities))}


def run_path(root, run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ValueError("run-id must be 1-64 letters/digits/underscores/hyphens")
    return Path(root) / "runs" / run_id


def train(root, run_id, regularization=1.0, use_mlflow=False):
    root = Path(root)
    destination = run_path(root, run_id)
    if destination.exists():
        raise ValueError("Run IDs are immutable; choose a new run-id")
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("C must be finite and positive")
    tracker = configure_mlflow(root) if use_mlflow else None
    data_path = root / "data/reference.csv"
    data = read_data(data_path)
    training, validation, test = split_data(data)
    pipeline = Pipeline([
        ("features", ColumnTransformer([
            ("numeric", StandardScaler(), NUMERIC),
            ("categorical", OneHotEncoder(categories=[CHANNELS], handle_unknown="error"), ["channel"])])),
        ("classifier", LogisticRegression(C=regularization, max_iter=500, random_state=42))])
    pipeline.fit(training[FEATURES], training.label)
    validation_probabilities = pipeline.predict_proba(validation[FEATURES])[:, 1]
    thresholds = np.linspace(0.05, 0.9, 86)
    threshold = float(max(thresholds, key=lambda value: (f1_score(validation.label, validation_probabilities >= value), value)))
    # Test features/labels are not used for fit, threshold selection, or metrics here.
    validation_metrics = metrics(validation.label, validation_probabilities, threshold)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}-", dir=destination.parent))
    try:
        joblib.dump(pipeline, temporary / "model.joblib")
        write_json(temporary / "environment.json", environment())
        for filename in ["lab.py", "api.py", "requirements.lock"]:
            shutil.copy2(Path(__file__).parent / filename, temporary / filename)
        files = {p.name: digest(p) for p in sorted(temporary.iterdir())}
        manifest = {"run_id": run_id, "schema_version": SCHEMA_VERSION,
                    "input_schema": Features.model_json_schema(), "feature_columns": FEATURES,
                    "data_sha256": digest(data_path), "files_sha256": files,
                    "threshold": threshold, "threshold_policy": "max validation F1; ties choose higher threshold",
                    "parameters": {"C": regularization, "seed": 42}, "validation": validation_metrics,
                    "snapshot_as_of": SNAPSHOT_AS_OF, "split_policy_version": SPLIT_POLICY_VERSION,
                    "split_rows": dict(zip(["train", "validation", "test"], map(len, [training, validation, test]))),
                    "split_id_sha256": {name: hashlib.sha256(canonical(part.event_id.tolist())).hexdigest()
                                        for name, part in zip(["train", "validation", "test"], [training, validation, test])},
                    "reference": reference_summary(training, pipeline.predict_proba(training[FEATURES])[:, 1]),
                    "synthetic_only": True}
        manifest["bundle_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
        write_json(temporary / "manifest.json", manifest)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    if use_mlflow:
        log_mlflow(tracker, root, destination, manifest)
    return manifest


def configure_mlflow(root):
    """Preflight the optional SQLite backend before creating an immutable run."""
    try:
        import mlflow
        import alembic  # noqa: F401 - mlflow-skinny leaves database extras optional.
        import sqlalchemy  # noqa: F401
    except ImportError as error:
        raise ValueError("Install requirements-mlflow.txt before using --mlflow") from error
    mlflow.set_tracking_uri("sqlite:///" + str((root / "mlflow.db").resolve()))
    experiment = mlflow.get_experiment_by_name("synthetic-risk-lab")
    experiment_id = experiment.experiment_id if experiment else mlflow.create_experiment(
        "synthetic-risk-lab", artifact_location=(root / "mlartifacts").resolve().as_uri())
    mlflow.set_experiment(experiment_id=experiment_id)
    return mlflow


def log_mlflow(mlflow, root, destination, manifest):
    """Optional tracking only. Our teaching alias below is not MLflow Registry."""
    with mlflow.start_run(run_name=manifest["run_id"]) as run:
        mlflow.log_params(manifest["parameters"])
        mlflow.log_param("threshold", manifest["threshold"])
        mlflow.log_metrics({f"validation_{key}": value for key, value in manifest["validation"].items()
                            if isinstance(value, (int, float))})
        mlflow.set_tag("bundle_sha256", manifest["bundle_sha256"])
        mlflow.log_artifacts(str(destination), artifact_path="bundle")
        write_json(root / "tracking" / f"{manifest['run_id']}.json", {"mlflow_run_id": run.info.run_id})


def load_bundle(root, run_id):
    path = run_path(root, run_id)
    manifest = json.loads((path / "manifest.json").read_text())
    expected = manifest.pop("bundle_sha256")
    if hashlib.sha256(canonical(manifest)).hexdigest() != expected:
        raise ValueError("Manifest digest mismatch")
    manifest["bundle_sha256"] = expected
    if manifest["run_id"] != run_id or manifest["schema_version"] != SCHEMA_VERSION or manifest["input_schema"] != Features.model_json_schema():
        raise ValueError("Incompatible model/input schema")
    for filename, expected_hash in manifest["files_sha256"].items():
        if Path(filename).name != filename or digest(path / filename) != expected_hash:
            raise ValueError("Bundle file digest mismatch")
    recorded = json.loads((path / "environment.json").read_text())
    current = environment()
    if recorded["python"].split(".")[:2] != current["python"].split(".")[:2] or recorded["packages"] != current["packages"]:
        raise ValueError("Runtime dependency mismatch; use the recorded environment")
    # joblib/pickle can execute code. Load only these locally created trusted bundles.
    return joblib.load(path / "model.joblib"), manifest


def evaluate(root, run_id, min_average_precision=0.25):
    root = Path(root)
    output = root / "evaluations" / f"{run_id}.json"
    if output.exists():
        raise ValueError("Evaluation is immutable; create another run to change a gate")
    pipeline, manifest = load_bundle(root, run_id)
    data_path = root / "data/reference.csv"
    if digest(data_path) != manifest["data_sha256"]:
        raise ValueError("Evaluation data digest differs from the training snapshot")
    if manifest["snapshot_as_of"] != SNAPSHOT_AS_OF or manifest["split_policy_version"] != SPLIT_POLICY_VERSION:
        raise ValueError("Evaluation split identity policy changed; train a new run")
    splits = split_data(read_data(data_path))
    for name, part in zip(["train", "validation", "test"], splits):
        id_hash = hashlib.sha256(canonical(part.event_id.tolist())).hexdigest()
        if id_hash != manifest["split_id_sha256"][name] or len(part) != manifest["split_rows"][name]:
            raise ValueError("Evaluation split identity differs from the training manifest")
    test = splits[2]
    result = metrics(test.label, pipeline.predict_proba(test[FEATURES])[:, 1], manifest["threshold"])
    gates = {"average_precision_min": max(0.25, min_average_precision), "roc_auc_min": 0.65, "recall_min": 0.45}
    passed = (result["average_precision"] >= gates["average_precision_min"] and
              result["roc_auc"] >= gates["roc_auc_min"] and result["recall"] >= gates["recall_min"])
    report = {"run_id": run_id, "bundle_sha256": manifest["bundle_sha256"],
              "data_sha256": manifest["data_sha256"], "split": "held-out chronological test",
              "split_id_sha256": manifest["split_id_sha256"]["test"],
              "evaluation_code_sha256": digest(Path(__file__)),
              "metrics": result, "gate_policy": gates, "gate_passed": bool(passed),
              "warning": "Toy release gate, not a real fraud or business acceptance criterion"}
    write_json(output, report)
    return report


def check_release(root, run_id):
    _, manifest = load_bundle(root, run_id)
    path = Path(root) / "evaluations" / f"{run_id}.json"
    if not path.exists():
        raise ValueError("Run evaluate before promotion")
    report = json.loads(path.read_text())
    if not report["gate_passed"] or report["bundle_sha256"] != manifest["bundle_sha256"]:
        raise ValueError("Promotion gate refused this bundle")
    return manifest


def promote(root, run_id):
    root = Path(root)
    manifest = check_release(root, run_id)
    path = root / "registry/production.json"
    previous = json.loads(path.read_text()) if path.exists() else None
    if previous and previous["current"] == run_id:
        raise ValueError("Already active")
    release = {"current": run_id, "current_sha256": manifest["bundle_sha256"],
               "previous": previous["current"] if previous else None,
               "previous_sha256": previous["current_sha256"] if previous else None}
    write_json(path, release)
    return release


def rollback(root):
    path = Path(root) / "registry/production.json"
    release = json.loads(path.read_text())
    if not release["previous"]:
        raise ValueError("No previous release to roll back to")
    previous = check_release(root, release["previous"])
    if previous["bundle_sha256"] != release["previous_sha256"]:
        raise ValueError("Previous release digest mismatch")
    restored = {"current": release["previous"], "current_sha256": release["previous_sha256"],
                "previous": release["current"], "previous_sha256": release["current_sha256"]}
    write_json(path, restored)
    return restored


def load_active(root):
    release = json.loads((Path(root) / "registry/production.json").read_text())
    pipeline, manifest = load_bundle(root, release["current"])
    if release["current_sha256"] != manifest["bundle_sha256"]:
        raise ValueError("Active release digest mismatch")
    return pipeline, manifest


def predict_loaded(pipeline, manifest, row):
    validated = Features.model_validate(row).model_dump()
    probability = float(pipeline.predict_proba(pd.DataFrame([validated])[FEATURES])[0, 1])
    return {"model_version": manifest["run_id"], "bundle_sha256": manifest["bundle_sha256"],
            "risk_probability": probability, "threshold": manifest["threshold"],
            "flagged": bool(probability >= manifest["threshold"]), "synthetic_only": True}


def monitor(root, data_path, as_of=SNAPSHOT_AS_OF):
    root = Path(root)
    pipeline, manifest = load_active(root)
    data = read_data(data_path, allow_unlabeled=True)
    as_of = pd.to_datetime(as_of, utc=True, errors="raise")
    if pd.isna(as_of):
        raise ValueError("as-of must be a valid UTC timestamp")
    data = data[data.event_time <= as_of]
    if data.empty:
        raise ValueError("No batch events have occurred by as-of")
    probabilities = pipeline.predict_proba(data[FEATURES])[:, 1]
    reference = manifest["reference"]
    shifts = {name: float(abs(data[name].mean() - reference["numeric"][name]["mean"]) /
                          max(reference["numeric"][name]["std"], 1e-9)) for name in NUMERIC}
    tvd = float(sum(abs((data.channel == name).mean() - reference["channel_frequency"][name]) for name in CHANNELS) / 2)
    score_shift = float(abs(probabilities.mean() - reference["score_mean"]))
    performance = None
    degradation = None
    mature_count = 0
    performance_status = "unavailable_without_mature_labels"
    if "label" in data:
        mature = data.label_ready_time <= as_of
        mature_count = int(mature.sum())
        performance_status = "insufficient_mature_labels" if mature_count < 100 else "one_class_labels"
        if mature_count >= 100 and data.loc[mature, "label"].nunique() == 2:
            performance_status = "available"
            performance = metrics(data.loc[mature, "label"], probabilities[mature], manifest["threshold"])
            baseline = json.loads((root / "evaluations" / f"{manifest['run_id']}.json").read_text())["metrics"]
            degradation = (performance["average_precision"] < baseline["average_precision"] - 0.10 or
                           performance["recall"] < baseline["recall"] - 0.15)
    report = {"model_version": manifest["run_id"], "batch_rows": len(data), "batch_sha256": digest(data_path),
              "as_of": str(as_of), "mature_label_rows": mature_count,
              "standardized_mean_shift": shifts, "channel_total_variation": tvd,
              "score_mean_absolute_shift": score_shift,
              "data_drift_alert": bool(any(value > 0.5 for value in shifts.values()) or tvd > 0.15),
              "score_drift_alert": bool(score_shift > 0.10), "performance": performance,
              "performance_status": performance_status,
              "performance_degradation_alert": bool(degradation) if degradation is not None else None,
              "warning": "Aggregate teaching heuristics; drift is not proof of model degradation"}
    write_json(root / "reports" / f"monitor-{uuid4().hex[:12]}.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("workspace"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("generate")
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--run-id", required=True)
    train_parser.add_argument("--C", type=float, default=1.0)
    train_parser.add_argument("--mlflow", action="store_true")
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--run-id", required=True)
    evaluate_parser.add_argument("--min-average-precision", type=float, default=0.25,
                                 help="Only raises the built-in 0.25 toy AP gate")
    promote_parser = commands.add_parser("promote")
    promote_parser.add_argument("--run-id", required=True)
    predict_parser = commands.add_parser("predict")
    predict_parser.add_argument("--json", required=True)
    monitor_parser = commands.add_parser("monitor")
    monitor_parser.add_argument("--data", type=Path)
    monitor_parser.add_argument("--as-of", default=SNAPSHOT_AS_OF)
    commands.add_parser("rollback")
    args = parser.parse_args()
    try:
        if args.command == "generate":
            result = generate(args.root)
        elif args.command == "train":
            manifest = train(args.root, args.run_id, args.C, args.mlflow)
            result = {key: manifest[key] for key in ["run_id", "bundle_sha256", "threshold", "split_rows", "validation"]}
        elif args.command == "evaluate":
            result = evaluate(args.root, args.run_id, args.min_average_precision)
        elif args.command == "promote":
            result = promote(args.root, args.run_id)
        elif args.command == "rollback":
            result = rollback(args.root)
        elif args.command == "predict":
            result = predict_loaded(*load_active(args.root), json.loads(args.json))
        else:
            result = monitor(args.root, args.data or args.root / "data/shifted.csv", args.as_of)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
