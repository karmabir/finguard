import asyncio
import sqlite3
from types import SimpleNamespace

import joblib
import pytest
from fastapi.testclient import TestClient

from finguard import data, model, tools
from finguard import agent
from finguard.agent import _parse_response
from finguard.policy import PolicyIndex


@pytest.fixture()
def local_pipeline(tmp_path, monkeypatch):
    frame = data.synthetic(n=3000, fraud_rate=0.05, seed=7)
    db = tmp_path / "finguard.db"
    with sqlite3.connect(db) as con:
        frame.to_sql("transactions", con, index=False)
    clf, info = model.train_eval(df=frame)
    model_path = tmp_path / "model.joblib"
    training = model.temporal_split(frame)[0]
    joblib.dump({"clf": clf, "threshold": info["threshold"], "feature_set": "pre_txn", "info": info,
                 "drift_reference": model.make_drift_reference(training)}, model_path)
    monkeypatch.setattr(tools, "DB", db)
    monkeypatch.setattr(tools, "AUDIT", tmp_path / "audit.log")
    monkeypatch.setattr(model, "MODEL_PATH", model_path)
    return frame


def test_model_comparison_and_legacy_upgrade(local_pipeline, monkeypatch):
    from finguard.api import app

    monkeypatch.setenv("FINGUARD_API_KEYS", "local-test")
    client = TestClient(app)
    headers = {"X-API-Key": "local-test"}
    path = "/transactions/45"
    default_score = tools.score_transaction(45)
    assert tools.compare_scores(45) == [default_score]
    monkeypatch.setattr(model, "load_frame", lambda: local_pipeline)
    metrics = model.train_models()
    expected_sizes = tuple(map(len, model.temporal_split(local_pipeline)))
    for info in metrics.values():
        assert (info["n_train"], info["n_val"], info["n_test"]) == expected_sizes
    assert client.get(path + "/scores").status_code == 401
    comparison = client.get(path + "/scores", headers=headers).json()
    assert comparison["default_model_id"] == model.DEFAULT_MODEL
    assert {row["model_id"] for row in comparison["scores"]} == set(model.MODEL_NAMES)
    for row in comparison["scores"]:
        model_id = row["model_id"]
        single = client.get(path + "/score", params={"model_id": model_id}, headers=headers)
        assert row == single.json()
        assert row["flag_threshold"] == round(metrics[model_id]["threshold"], 6)
        assert row["features_used"] == model.FEATURE_SETS["pre_txn"]
    assert client.get(path + "/score", headers=headers).json() == default_score
    assert client.get(path + "/score?model_id=unknown", headers=headers).status_code == 422
    assert client.get("/transactions/999999/scores", headers=headers).status_code == 404
    summary = client.get("/dashboard/summary", headers=headers).json()
    assert {row["id"] for row in summary["models"]} == set(metrics)


def test_synthetic_pipeline_is_complete(local_pipeline):
    assert data.REQUIRED_COLUMNS | {"txn_id"} <= set(local_pipeline.columns)
    transaction = tools.get_transaction(0)
    assert transaction["txn_id"] == 0
    assert "isFraud" not in transaction
    score = tools.score_transaction(0)
    assert 0 <= score["model_score"] <= 1


def test_select_guard_and_row_cap(local_pipeline):
    assert tools.run_select("select count(*) as n from transactions")[0]["n"] == len(local_pipeline)
    assert len(tools.run_select("select txn_id from transactions")) == tools.MAX_ROWS
    for query in ["drop table transactions", "select 1; delete from transactions",
                  "update transactions set isFraud=0", "select isFraud from transactions",
                  "select * from transactions", "select name from sqlite_master"]:
        with pytest.raises((ValueError, sqlite3.DatabaseError)):
            tools.run_select(query)


def test_missing_transaction(local_pipeline):
    assert tools.get_transaction(999_999) is None
    assert tools.score_transaction(999_999) is None
    with pytest.raises(LookupError):
        tools.label_for_eval(999_999)


def test_policy_search():
    assert PolicyIndex().search("account emptying transfer")[0]["policy_id"] == "P-02"


def test_agent_output_is_fail_closed():
    assert _parse_response('{"decision":"clear","summary":"ok","evidence":[]}')["decision"] == "clear"
    assert _parse_response('{"decision":"approve","summary":"ok","evidence":[]}')["decision"] == "review"
    assert _parse_response("not json")["decision"] == "review"


def test_mcp_exposes_agent_tools():
    from finguard.mcp_server import mcp
    listed = asyncio.run(mcp.list_tools())
    names = {tool.name for tool in listed}
    assert names == {"get_transaction", "run_select", "score_transaction", "search_policy"}
    assert all(tool.annotations.read_only_hint for tool in listed)
    assert all(tool.annotations.open_world_hint is False for tool in listed)


def test_openai_tool_loop(monkeypatch):
    responses = [
        SimpleNamespace(
            output=[SimpleNamespace(type="function_call", name="get_transaction",
                                    arguments='{"txn_id": 7}', call_id="call-1")],
            output_text="", usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        ),
        SimpleNamespace(
            output=[], output_text='{"decision":"clear","summary":"low risk","evidence":[]}',
            usage=SimpleNamespace(input_tokens=20, output_tokens=8),
        ),
    ]

    class FakeResponses:
        def __init__(self):
            self.inputs = []

        def create(self, **kwargs):
            self.inputs.append(kwargs["input"].copy())
            return responses.pop(0)

    fake = SimpleNamespace(responses=FakeResponses())
    monkeypatch.setattr(agent, "_run_tool", lambda name, args: {"txn_id": args["txn_id"]})
    result = agent.investigate(7, client=fake)
    assert result["decision"] == "clear"
    assert result["model"] == agent.MODEL
    assert result["input_tokens"] == 30
    assert fake.responses.inputs[1][-1]["type"] == "function_call_output"


def test_api_authentication_metrics_and_drift(local_pipeline, monkeypatch):
    monkeypatch.setenv("FINGUARD_API_KEYS", "test-secret")
    from finguard.api import app
    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert "FinGuard" in client.get("/").text
    assert client.get("/transactions/0").status_code == 401
    headers = {"X-API-Key": "test-secret"}
    summary = client.get("/dashboard/summary", headers=headers)
    assert summary.status_code == 200
    assert summary.json()["transactions"] == len(local_pipeline)
    assert client.get("/transactions/0", headers=headers).status_code == 200
    drift = client.get("/monitoring/drift?sample_size=500", headers=headers)
    assert drift.status_code == 200
    assert set(drift.json()["features"]) == set(model.DRIFT_FEATURES)
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "finguard_http_requests_total" in metrics.text
