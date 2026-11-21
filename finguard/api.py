"""Authenticated HTTP API for FinGuard. Run with `python -m finguard serve`."""
import hmac
import os
import time
from pathlib import Path

from dotenv import dotenv_values
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import tools
from .data import DB
from .model import DEFAULT_MODEL, MODEL_NAMES, MODEL_PATH, load_model, model_catalog
from .monitoring import drift_report, prometheus_metrics, record_request
from .policy import PolicyIndex

app = FastAPI(title="FinGuard", version="1.0.0")
policy = PolicyIndex()
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


def _configured_keys():
    raw = os.environ.get("FINGUARD_API_KEYS") or dotenv_values(
        Path(__file__).resolve().parents[1] / ".env").get("FINGUARD_API_KEYS", "")
    return [key.strip() for key in raw.split(",") if key.strip()]


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)):
    keys = _configured_keys()
    if not keys:
        if request.client and request.client.host in {"127.0.0.1", "::1"}:
            return
        raise HTTPException(status_code=503, detail="API authentication is not configured")
    if x_api_key is None or not any(hmac.compare_digest(x_api_key, key) for key in keys):
        raise HTTPException(status_code=401, detail="invalid API key", headers={"WWW-Authenticate": "ApiKey"})


protected = [Depends(require_api_key)]


@app.middleware("http")
async def request_metrics(request, call_next):
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        record_request(request.method, path, status, time.perf_counter() - started)


class InvestigationRequest(BaseModel):
    txn_id: int = Field(ge=0)


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    ready = DB.exists() and MODEL_PATH.exists()
    return {"status": "ready" if ready else "setup_required", "database": DB.exists(), "model": MODEL_PATH.exists(),
            "authentication": bool(_configured_keys())}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return prometheus_metrics()


@app.get("/dashboard/summary", dependencies=protected)
def dashboard_summary():
    con = tools._connect()
    try:
        total = con.execute("select count(*) from transactions").fetchone()[0]
        by_type = {row[0]: row[1] for row in con.execute(
            "select type, count(*) from transactions group by type order by count(*) desc")}
    finally:
        con.close()
    artifact = load_model()
    info = artifact.get("info", {})
    return {
        "transactions": total,
        "by_type": by_type,
        "default_model_id": DEFAULT_MODEL,
        "models": [
            {"id": key, "name": MODEL_NAMES[key], "threshold": item["threshold"], **item.get("info", {})}
            for key, item in model_catalog(artifact).items()
        ],
        "model": {
            "name": type(artifact["clf"]).__name__,
            "feature_set": artifact["feature_set"],
            "threshold": artifact["threshold"],
            "pr_auc": info.get("test_pr_auc"),
            "precision": info.get("test_precision"),
            "recall": info.get("test_recall"),
        },
    }


@app.get("/transactions/{txn_id}", dependencies=protected)
def transaction(txn_id: int):
    row = tools.get_transaction(txn_id)
    if row is None:
        raise HTTPException(status_code=404, detail="transaction not found")
    return row


@app.get("/transactions/{txn_id}/score", dependencies=protected)
def score(txn_id: int, model_id: str = DEFAULT_MODEL):
    try:
        result = tools.score_transaction(txn_id, model_id)
    except tools.ModelUnavailable as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="transaction not found")
    return result


@app.get("/transactions/{txn_id}/scores", dependencies=protected)
def scores(txn_id: int):
    result = tools.compare_scores(txn_id)
    if result is None:
        raise HTTPException(status_code=404, detail="transaction not found")
    return {"default_model_id": DEFAULT_MODEL, "scores": result}


@app.get("/policies/search", dependencies=protected)
def search_policy(q: str, k: int = 3):
    if not 1 <= k <= 10:
        raise HTTPException(status_code=422, detail="k must be between 1 and 10")
    return policy.search(q, k=k)


@app.get("/monitoring/drift", dependencies=protected)
def model_drift(sample_size: int = 10000):
    try:
        return drift_report(sample_size)
    except (RuntimeError, FileNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/investigations", dependencies=protected)
def investigate(request: InvestigationRequest):
    from .agent import investigate as run
    if tools.get_transaction(request.txn_id) is None:
        raise HTTPException(status_code=404, detail="transaction not found")
    return run(request.txn_id)
