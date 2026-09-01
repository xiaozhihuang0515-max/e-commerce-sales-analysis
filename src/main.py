"""Single-file E-commerce Intelligence Platform.

Run locally:
    pip install fastapi uvicorn sqlalchemy pandas scikit-learn joblib
    python ecommerce_intelligence_platform.py --train-demo
    uvicorn ecommerce_intelligence_platform:app --reload

Import a supplied Kaggle CSV safely in batches (never loads it all at once):
    python ecommerce_intelligence_platform.py --ingest /path/to/2019-Oct.csv --chunksize 100000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sqlalchemy import DateTime, Float, Integer, String, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ecommerce-intelligence")

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'ecommerce_intelligence.db'}")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
FEATURE_NAMES = ["session_duration", "views", "cart_actions", "previous_purchases", "average_price", "category_diversity"]


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class Product(Base):
    __tablename__ = "products"
    product_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    category_code: Mapped[str | None] = mapped_column(String(255))
    brand: Mapped[str | None] = mapped_column(String(255))
    price: Mapped[float | None] = mapped_column(Float)


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_time: Mapped[datetime] = mapped_column(DateTime, index=True)
    event_type: Mapped[str] = mapped_column(String(32), index=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    product_id: Mapped[int] = mapped_column(Integer, index=True)
    session_id: Mapped[str | None] = mapped_column(String(64), index=True)
    price: Mapped[float] = mapped_column(Float)
    category_code: Mapped[str | None] = mapped_column(String(255))


class CustomerFeature(Base):
    __tablename__ = "features"
    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_duration: Mapped[float] = mapped_column(Float)
    views: Mapped[int] = mapped_column(Integer)
    cart_actions: Mapped[int] = mapped_column(Integer)
    previous_purchases: Mapped[int] = mapped_column(Integer)
    average_price: Mapped[float] = mapped_column(Float)
    category_diversity: Mapped[int] = mapped_column(Integer)
    recency_days: Mapped[float] = mapped_column(Float)
    monetary_value: Mapped[float] = mapped_column(Float)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    probability: Mapped[float] = mapped_column(Float)
    intent: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class ConversionRequest(BaseModel):
    user_id: int = Field(gt=0)
    session_duration: float = Field(ge=0, le=86400, description="Seconds in current session")
    views: int = Field(ge=0, le=10000)
    cart_actions: int = Field(ge=0, le=1000)
    previous_purchases: int = Field(ge=0, le=100000)
    average_price: float = Field(default=50, ge=0, le=1_000_000)
    category_diversity: int = Field(default=1, ge=0, le=1000)


def get_db() -> Generator[Session, None, None]:
    database = SessionLocal()
    try:
        yield database
    finally:
        database.close()


def model_path(name: str) -> Path:
    return ARTIFACT_DIR / name


def bootstrap_demo_data(database: Session, rows: int = 4000) -> None:
    """Create deterministic demo features, allowing the app to work before a CSV import."""
    if database.scalar(select(func.count()).select_from(CustomerFeature)):
        return
    random = np.random.default_rng(42)
    records: list[CustomerFeature] = []
    for user_id in range(1, rows + 1):
        purchases = int(random.poisson(2.1))
        records.append(CustomerFeature(
            user_id=user_id, session_duration=float(random.gamma(2.2, 140)), views=int(random.poisson(9)),
            cart_actions=int(random.poisson(1.5)), previous_purchases=purchases,
            average_price=float(random.gamma(3, 24)), category_diversity=int(random.integers(1, 9)),
            recency_days=float(random.gamma(2, 8)), monetary_value=float(purchases * random.gamma(3, 38)),
        ))
    database.add_all(records)
    database.commit()
    logger.info("Seeded %s demo customer feature rows", rows)


def train_models(database: Session) -> dict:
    bootstrap_demo_data(database)
    rows = database.execute(select(CustomerFeature)).scalars().all()
    frame = pd.DataFrame([{field: getattr(row, field) for field in FEATURE_NAMES + ["monetary_value"]} for row in rows])
    random = np.random.default_rng(9)
    signal = 0.006 * frame.session_duration + 0.12 * frame.views + 0.55 * frame.cart_actions + 0.25 * frame.previous_purchases - 3.3
    labels = (random.random(len(frame)) < 1 / (1 + np.exp(-signal))).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(frame[FEATURE_NAMES], labels, test_size=.2, random_state=42, stratify=labels)
    candidates = {"logistic_regression": LogisticRegression(max_iter=1000), "random_forest": RandomForestClassifier(n_estimators=180, min_samples_leaf=4, random_state=42, n_jobs=-1)}
    scored: list[tuple[str, object, float]] = []
    for name, candidate in candidates.items():
        candidate.fit(x_train, y_train)
        scored.append((name, candidate, roc_auc_score(y_test, candidate.predict_proba(x_test)[:, 1])))
    name, best, auc = max(scored, key=lambda item: item[2])
    probabilities = best.predict_proba(x_test)[:, 1]
    predicted = (probabilities >= .5).astype(int)
    metrics = {"model": name, "version": "v1.0", "roc_auc": round(float(auc), 3), "accuracy": round(float(accuracy_score(y_test, predicted)), 3), "precision": round(float(precision_score(y_test, predicted, zero_division=0)), 3), "recall": round(float(recall_score(y_test, predicted, zero_division=0)), 3), "f1": round(float(f1_score(y_test, predicted, zero_division=0)), 3), "trained_at": datetime.now(timezone.utc).isoformat()}
    joblib.dump({"model": best, "metrics": metrics}, model_path("conversion.joblib"))
    rfm = pd.DataFrame({"recency": [row.recency_days for row in rows], "frequency": [row.previous_purchases for row in rows], "monetary": [row.monetary_value for row in rows]})
    segmenter = KMeans(n_clusters=3, random_state=42, n_init=20).fit(rfm)
    means = pd.DataFrame(rfm).assign(cluster=segmenter.labels_).groupby("cluster").mean()
    ordering = means.rank(pct=True).mean(axis=1).sort_values().index.tolist()
    labels_by_cluster = {ordering[0]: "Inactive Customer", ordering[1]: "Potential Customer", ordering[2]: "High Value Customer"}
    joblib.dump({"model": segmenter, "labels": labels_by_cluster}, model_path("segments.joblib"))
    return metrics


def ensure_models(database: Session) -> None:
    if not model_path("conversion.joblib").exists() or not model_path("segments.joblib").exists():
        train_models(database)


def ingest_csv(csv_path: Path, chunksize: int = 100_000) -> dict[str, int]:
    """Validate and import source events in batches. Invalid rows are skipped and counted."""
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    Base.metadata.create_all(engine)
    imported = invalid = 0
    required = {"event_time", "event_type", "product_id", "price", "user_id", "user_session"}
    with SessionLocal() as database:
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            if not required.issubset(chunk.columns):
                raise ValueError(f"Missing columns: {sorted(required - set(chunk.columns))}")
            chunk = chunk.drop_duplicates()
            chunk["event_time"] = pd.to_datetime(chunk["event_time"], errors="coerce", utc=True)
            chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
            valid = chunk.dropna(subset=["event_time", "event_type", "product_id", "price", "user_id"])
            valid = valid[valid.price >= 0]
            invalid += len(chunk) - len(valid)
            database.bulk_insert_mappings(Event, [{"event_time": row.event_time.to_pydatetime(), "event_type": str(row.event_type), "user_id": int(row.user_id), "product_id": int(row.product_id), "session_id": str(row.user_session), "price": float(row.price), "category_code": None if pd.isna(row.get("category_code")) else str(row.get("category_code"))} for _, row in valid.iterrows()])
            database.commit()
            imported += len(valid)
    return {"imported": imported, "invalid_or_duplicate": invalid}


def intent_and_action(probability: float) -> tuple[str, str]:
    if probability >= .70:
        return "High Intent", "Send a personalized promotion"
    if probability >= .35:
        return "Medium Intent", "Show product recommendations and free-shipping incentive"
    return "Low Intent", "Use a low-cost re-engagement campaign"


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(engine)
    with SessionLocal() as database:
        ensure_models(database)
    yield


app = FastAPI(title="E-commerce Intelligence Platform", version="1.0.0", lifespan=lifespan)


@app.get("/api/health")
def health(database: Session = Depends(get_db)) -> dict:
    return {"status": "ok", "customers": database.scalar(select(func.count()).select_from(CustomerFeature)) or 0, "database": DATABASE_URL.split(":", 1)[0]}


@app.post("/predict_conversion")
def predict_conversion(payload: ConversionRequest, database: Session = Depends(get_db)) -> dict:
    started = time.perf_counter()
    ensure_models(database)
    artifact = joblib.load(model_path("conversion.joblib"))
    values = pd.DataFrame([[getattr(payload, feature) for feature in FEATURE_NAMES]], columns=FEATURE_NAMES)
    probability = float(artifact["model"].predict_proba(values)[0, 1])
    intent, action = intent_and_action(probability)
    database.merge(User(user_id=payload.user_id))
    database.add(Prediction(user_id=payload.user_id, probability=probability, intent=intent))
    database.commit()
    return {"user_id": payload.user_id, "conversion_probability": round(probability, 4), "customer_intent": intent, "recommendation": action, "latency_ms": round((time.perf_counter() - started) * 1000, 2)}


@app.get("/customer_segment/{user_id}")
def customer_segment(user_id: int, database: Session = Depends(get_db)) -> dict:
    if user_id <= 0:
        raise HTTPException(422, "user_id must be positive")
    ensure_models(database)
    feature = database.get(CustomerFeature, user_id)
    if feature is None:
        raise HTTPException(404, "Customer features not found; import data or use demo users 1-4000")
    artifact = joblib.load(model_path("segments.joblib"))
    rfm = pd.DataFrame([[feature.recency_days, feature.previous_purchases, feature.monetary_value]], columns=["recency", "frequency", "monetary"])
    segment = artifact["labels"][int(artifact["model"].predict(rfm)[0])]
    actions = {"High Value Customer": "VIP campaign and early-access promotion", "Potential Customer": "Cross-sell recommendation and loyalty incentive", "Inactive Customer": "Re-engagement offer"}
    return {"user_id": user_id, "segment": segment, "recommended_action": actions[segment], "rfm": {"recency_days": feature.recency_days, "frequency": feature.previous_purchases, "monetary_value": round(feature.monetary_value, 2)}}


@app.get("/model_monitoring")
def model_monitoring(database: Session = Depends(get_db)) -> dict:
    ensure_models(database)
    metrics = joblib.load(model_path("conversion.joblib"))["metrics"]
    count = database.scalar(select(func.count()).select_from(Prediction)) or 0
    return {"current_model": metrics["model"], "model_version": metrics["version"], "performance": metrics, "system_metrics": {"prediction_requests": count, "last_updated": metrics["trained_at"]}}


@app.get("/")
def decision_console() -> HTMLResponse:
    return HTMLResponse("""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CommerceSignal | Purchase Intent</title>
  <style>
    :root{--ink:#132238;--muted:#66758a;--line:#dfe5ec;--surface:#fff;--canvas:#f5f7fa;--navy:#102a43;--blue:#2563eb;--mint:#0f9f7d;--mint-soft:#e8f8f2;--warning:#c66a13;--shadow:0 14px 36px rgba(29,48,72,.09)}
    *{box-sizing:border-box} body{margin:0;background:var(--canvas);color:var(--ink);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.45}
    .topbar{height:68px;background:#fff;border-bottom:1px solid var(--line);display:flex;align-items:center;padding:0 max(28px,calc((100% - 1180px)/2));gap:12px}.brand-mark{width:34px;height:34px;border-radius:9px;background:var(--blue);display:grid;place-items:center;color:#fff;font-weight:800;font-size:17px}.brand{font-weight:750;letter-spacing:-.3px}.brand span{color:var(--blue)}.env{margin-left:auto;color:#39705d;background:#edfaf4;border-radius:20px;padding:6px 11px;font-size:12px;font-weight:700}
    main{max-width:1180px;margin:0 auto;padding:52px 28px 72px}.eyebrow{color:var(--blue);font-size:12px;font-weight:800;letter-spacing:1.2px;text-transform:uppercase;margin:0 0 10px}h1{font-size:38px;letter-spacing:-1.3px;margin:0;line-height:1.14}.intro{max-width:650px;color:var(--muted);font-size:17px;margin:14px 0 33px}.workspace{display:grid;grid-template-columns:minmax(0,1.06fr) minmax(330px,.94fr);gap:22px}.card{background:var(--surface);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow)}.card-head{padding:24px 26px 18px;border-bottom:1px solid var(--line)}h2{font-size:19px;margin:0;letter-spacing:-.3px}.subtext{margin:5px 0 0;color:var(--muted);font-size:14px}.form-body{padding:24px 26px 26px}.fields{display:grid;grid-template-columns:1fr 1fr;gap:18px}.field:first-child{grid-column:span 2}.field label{display:block;font-size:13px;font-weight:750;margin:0 0 6px}.field small{display:block;min-height:34px;color:var(--muted);font-size:12px;line-height:1.35;margin-bottom:7px}.field input{width:100%;border:1px solid #c9d3df;border-radius:8px;padding:11px 12px;color:var(--ink);font-size:15px;background:#fff;outline:none}.field input:focus{border-color:var(--blue);box-shadow:0 0 0 3px rgba(37,99,235,.12)}.submit{width:100%;margin-top:24px;border:0;border-radius:8px;background:var(--blue);padding:13px 16px;color:#fff;font-size:15px;font-weight:750;cursor:pointer}.submit:hover{background:#1d4ed8}.submit:disabled{opacity:.65;cursor:wait}
    .result-card{overflow:hidden}.result-top{background:var(--navy);padding:26px;color:#fff}.result-top h2{color:#fff}.result-top p{margin:5px 0 0;color:#b9c9da;font-size:14px}.empty{padding:32px 27px;color:var(--muted)}.empty strong{display:block;color:var(--ink);margin-bottom:6px}.result{display:none}.result.visible{display:block}.score{padding:25px 27px;border-bottom:1px solid var(--line)}.score-label{font-size:12px;color:var(--muted);font-weight:750;text-transform:uppercase;letter-spacing:.8px}.score-value{font-size:54px;line-height:1;margin:7px 0 9px;letter-spacing:-2px}.pill{display:inline-block;padding:5px 9px;border-radius:20px;background:var(--mint-soft);color:#087257;font-size:12px;font-weight:800}.action{padding:22px 27px}.action-label{font-size:12px;font-weight:800;color:var(--muted);letter-spacing:.8px;text-transform:uppercase}.action p{font-size:16px;margin:8px 0 0;font-weight:650}.latency{padding:0 27px 23px;font-size:12px;color:var(--muted)}.info-row{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:22px}.info{background:#fff;border:1px solid var(--line);border-radius:12px;padding:17px}.info b{display:block;font-size:13px;margin-bottom:5px}.info span{font-size:12px;color:var(--muted)}.links{margin-top:29px;display:flex;gap:21px;flex-wrap:wrap}.links a{color:#31577f;font-size:13px;font-weight:700;text-decoration:none}.links a:hover{text-decoration:underline}@media(max-width:760px){main{padding:32px 18px}.topbar{padding:0 18px}.workspace{grid-template-columns:1fr}.fields{grid-template-columns:1fr}.field:first-child{grid-column:auto}.field small{min-height:0}h1{font-size:32px}.info-row{grid-template-columns:1fr}}
  </style>
</head>
<body>
  <header class="topbar"><div class="brand-mark">C</div><div class="brand">Commerce<span>Signal</span></div><div class="env">MODEL SERVICE ONLINE</div></header>
  <main>
    <p class="eyebrow">Customer Decisioning</p>
    <h1>Purchase Intent Workspace</h1>
    <p class="intro">Estimate whether a customer is likely to place an order during the current browsing session, then turn the prediction into a clear next-best action.</p>
    <div class="workspace">
      <section class="card"><div class="card-head"><h2>Customer session</h2><p class="subtext">Enter the behavioural signals available when a customer is browsing the store.</p></div>
        <form id="form" class="form-body"><div class="fields">
          <div class="field"><label for="user_id">Customer ID</label><small>A unique internal identifier for the customer. Example: 123.</small><input id="user_id" name="user_id" value="123" type="number" min="1" required></div>
          <div class="field"><label for="session_duration">Session duration (seconds)</label><small>How long the customer has spent browsing in this visit. 300 seconds = 5 minutes.</small><input id="session_duration" name="session_duration" value="300" type="number" min="0" required></div>
          <div class="field"><label for="views">Product pages viewed</label><small>The number of product-detail pages opened during this session.</small><input id="views" name="views" value="20" type="number" min="0" required></div>
          <div class="field"><label for="cart_actions">Items added to cart</label><small>The number of add-to-cart actions in this session—a strong purchase-intent signal.</small><input id="cart_actions" name="cart_actions" value="3" type="number" min="0" required></div>
          <div class="field"><label for="previous_purchases">Previous completed orders</label><small>How many orders this customer completed before the current visit.</small><input id="previous_purchases" name="previous_purchases" value="5" type="number" min="0" required></div>
        </div><button class="submit" type="submit">Assess purchase intent</button></form>
      </section>
      <aside class="card result-card"><div class="result-top"><h2>Decision output</h2><p>Model-backed recommendation for the current customer session.</p></div><div id="empty" class="empty"><strong>Ready for assessment</strong>Complete the customer session form to generate a purchase-intent decision.</div><div id="result" class="result"><div class="score"><div class="score-label">Predicted purchase probability</div><div id="probability" class="score-value">—</div><span id="intent" class="pill">—</span></div><div class="action"><div class="action-label">Recommended next action</div><p id="recommendation">—</p></div><div id="latency" class="latency">—</div></div></aside>
    </div>
    <div class="info-row"><div class="info"><b>Conversion model</b><span>Scores live customer behaviour through the deployed inference API.</span></div><div class="info"><b>Customer intelligence</b><span>RFM and clustering classify customers by value and activity.</span></div><div class="info"><b>Model monitoring</b><span>Tracks model quality and prediction-service usage.</span></div></div>
    <nav class="links"><a href="/customer_segment/123">View customer segment example →</a><a href="/model_monitoring">View model monitoring →</a><a href="/docs">Open API documentation →</a></nav>
  </main>
  <script>
    const form = document.querySelector('#form'); const button = document.querySelector('.submit');
    form.onsubmit = async event => { event.preventDefault(); const data = Object.fromEntries(new FormData(form)); Object.keys(data).forEach(key => data[key] = Number(data[key])); button.disabled = true; button.textContent = 'Assessing session…';
      try { const response = await fetch('/predict_conversion', {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify(data)}); const output = await response.json(); if (!response.ok) throw new Error(output.detail || 'Unable to score this session.'); document.querySelector('#empty').style.display = 'none'; document.querySelector('#result').classList.add('visible'); document.querySelector('#probability').textContent = `${Math.round(output.conversion_probability * 100)}%`; document.querySelector('#intent').textContent = output.customer_intent; document.querySelector('#recommendation').textContent = output.recommendation; document.querySelector('#latency').textContent = `Inference completed in ${output.latency_ms} ms`; } catch (error) { document.querySelector('#empty').innerHTML = `<strong>Assessment unavailable</strong>${error.message}`; } finally { button.disabled = false; button.textContent = 'Assess purchase intent'; }
    };
  </script>
</body>
</html>""")


def main() -> None:
    parser = argparse.ArgumentParser(description="E-commerce Intelligence Platform operations")
    parser.add_argument("--ingest", type=Path, help="CSV file to import in batches")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--train-demo", action="store_true", help="Create demo data and train models")
    args = parser.parse_args()
    Base.metadata.create_all(engine)
    if args.ingest:
        print(json.dumps(ingest_csv(args.ingest, args.chunksize), indent=2))
    if args.train_demo:
        with SessionLocal() as database:
            print(json.dumps(train_models(database), indent=2))
    if not args.ingest and not args.train_demo:
        parser.print_help()


if __name__ == "__main__":
    main()
