"""Safety contracts, exercised against real files, models, and the API."""
import json
import hashlib

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import lab
from api import create_app


ROW = {"amount": 120.0, "account_age_days": 90.0, "attempts_1h": 2, "channel": "web"}


@pytest.fixture
def trained(tmp_path):
    lab.generate(tmp_path)
    lab.train(tmp_path, "baseline")
    return tmp_path


def test_chronology_label_readiness_and_disjoint_ids(tmp_path):
    lab.generate(tmp_path)
    frame = lab.read_data(tmp_path / "data/reference.csv")
    train, validation, test = lab.split_data(frame)
    assert len(train) == 3481
    assert len(validation) == 1081
    assert len(test) == 1200
    assert train.label_ready_time.max() <= validation.event_time.min()
    assert validation.label_ready_time.max() <= test.event_time.min()
    assert set(train.event_id).isdisjoint(validation.event_id)
    assert set(train.event_id).isdisjoint(test.event_id)


@pytest.mark.parametrize("damage", ["duplicate_id", "future_label", "target_feature", "missing", "infinity"])
def test_malformed_training_data_is_refused(tmp_path, damage):
    lab.generate(tmp_path)
    path = tmp_path / "data/reference.csv"
    data = pd.read_csv(path)
    if damage == "duplicate_id":
        data.loc[2, "event_id"] = data.loc[1, "event_id"]
    elif damage == "future_label":
        data.loc[1, "label_ready_time"] = "2024-01-01T00:00:00Z"
    elif damage == "target_feature":
        data["chargeback_after_event"] = data.label
    elif damage == "missing":
        data.loc[1, "amount"] = np.nan
    else:
        data.loc[1, "amount"] = np.inf
    data.to_csv(path, index=False)
    with pytest.raises(ValueError):
        lab.train(tmp_path, "bad")
    assert not (tmp_path / "runs/bad").exists()


def test_pipeline_fit_only_on_train_and_serving_uses_same_pipeline(trained):
    pipeline, manifest = lab.load_bundle(trained, "baseline")
    train, _, _ = lab.split_data(lab.read_data(trained / "data/reference.csv"))
    scaler = pipeline.named_steps["features"].named_transformers_["numeric"]
    np.testing.assert_allclose(scaler.mean_, train[lab.NUMERIC].mean().to_numpy())
    lab.evaluate(trained, "baseline")
    lab.promote(trained, "baseline")
    expected = pipeline.predict_proba(pd.DataFrame([ROW])[lab.FEATURES])[0, 1]
    with TestClient(create_app(trained)) as client:
        result = client.post("/predict", json=ROW)
        assert result.status_code == 200
        assert result.json()["risk_probability"] == pytest.approx(expected)
        assert result.json()["model_version"] == "baseline"
        assert client.get("/readyz").status_code == 200
    assert manifest["data_sha256"]


@pytest.mark.parametrize("bad", [dict(ROW, amount=-1), dict(ROW, amount="120"),
                                    dict(ROW, channel="unknown"), dict(ROW, label=1),
                                    dict(ROW, attempts_1h=1.5)])
def test_api_rejects_schema_violations(trained, bad):
    lab.evaluate(trained, "baseline")
    lab.promote(trained, "baseline")
    with TestClient(create_app(trained)) as client:
        assert client.post("/predict", json=bad).status_code == 422


def test_unevaluated_and_failed_candidates_cannot_activate(trained):
    with pytest.raises(ValueError, match="evaluate"):
        lab.promote(trained, "baseline")
    report = lab.evaluate(trained, "baseline", min_average_precision=0.999)
    assert not report["gate_passed"]
    with pytest.raises(ValueError, match="gate"):
        lab.promote(trained, "baseline")
    assert not (trained / "registry/production.json").exists()


def test_tampered_bundle_rejected_before_deserializing(trained):
    path = trained / "runs/baseline/model.joblib"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="digest"):
        lab.load_bundle(trained, "baseline")


def test_rollback_and_process_pinning(trained):
    lab.evaluate(trained, "baseline")
    lab.promote(trained, "baseline")
    lab.train(trained, "candidate", regularization=0.2)
    lab.evaluate(trained, "candidate")
    with TestClient(create_app(trained)) as client:
        lab.promote(trained, "candidate")
        assert client.post("/predict", json=ROW).json()["model_version"] == "baseline"
    with TestClient(create_app(trained)) as client:
        assert client.post("/predict", json=ROW).json()["model_version"] == "candidate"
    assert lab.rollback(trained)["current"] == "baseline"
    assert lab.load_active(trained)[1]["run_id"] == "baseline"


def test_rollback_refuses_incompatible_previous_release(trained):
    lab.evaluate(trained, "baseline")
    lab.promote(trained, "baseline")
    lab.train(trained, "candidate", regularization=0.2)
    lab.evaluate(trained, "candidate")
    lab.promote(trained, "candidate")
    path = trained / "runs/baseline/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["schema_version"] = "incompatible"
    manifest.pop("bundle_sha256")
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    manifest["bundle_sha256"] = hashlib.sha256(encoded).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Incompatible"):
        lab.rollback(trained)
    assert json.loads((trained / "registry/production.json").read_text())["current"] == "candidate"


def test_shifted_batch_alerts_and_unlabeled_batch_has_no_performance(trained):
    lab.evaluate(trained, "baseline")
    lab.promote(trained, "baseline")
    report = lab.monitor(trained, trained / "data/shifted.csv")
    assert report["data_drift_alert"]
    assert report["performance"] is not None
    data = pd.read_csv(trained / "data/shifted.csv").drop(columns=["label", "label_ready_time"])
    path = trained / "unlabeled.csv"
    data.to_csv(path, index=False)
    report = lab.monitor(trained, path)
    assert report["performance"] is None
    assert report["performance_status"] == "unavailable_without_mature_labels"


def test_not_ready_is_explicit(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503
        assert client.post("/predict", json=ROW).status_code == 503


def test_evaluation_refuses_changed_split_policy(trained, monkeypatch):
    original = lab.split_data

    def changed_policy(data):
        train, validation, test = original(data)
        return train, validation, test.iloc[600:]

    monkeypatch.setattr(lab, "split_data", changed_policy)
    with pytest.raises(ValueError, match="split identity"):
        lab.evaluate(trained, "baseline")
    assert not (trained / "evaluations/baseline.json").exists()


def test_monitor_normalizes_utc_and_ignores_events_after_as_of(trained):
    lab.evaluate(trained, "baseline")
    lab.promote(trained, "baseline")
    report = lab.monitor(trained, trained / "data/shifted.csv", as_of="2025-01-10")
    assert report["batch_rows"] == 1
    assert report["mature_label_rows"] == 0
    assert report["performance"] is None
    assert report["performance_status"] == "insufficient_mature_labels"


def test_monitor_explains_one_class_labels(trained):
    lab.evaluate(trained, "baseline")
    lab.promote(trained, "baseline")
    data = pd.read_csv(trained / "data/shifted.csv")
    data["label"] = 0
    path = trained / "one-class.csv"
    data.to_csv(path, index=False)
    report = lab.monitor(trained, path)
    assert report["mature_label_rows"] == 1500
    assert report["performance"] is None
    assert report["performance_status"] == "one_class_labels"


def test_optional_mlflow_sqlite_tracking_stays_under_lab_root(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("alembic")
    lab.generate(tmp_path)
    lab.train(tmp_path, "tracked", use_mlflow=True)
    assert (tmp_path / "mlflow.db").is_file()
    receipt = json.loads((tmp_path / "tracking/tracked.json").read_text())
    run = mlflow.get_run(receipt["mlflow_run_id"])
    assert run.data.params["C"] == "1.0"
    assert run.data.metrics["validation_roc_auc"] > 0.7
    assert str(tmp_path) in run.info.artifact_uri
