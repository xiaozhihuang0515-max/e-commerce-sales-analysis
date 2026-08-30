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
    return HTMLResponse("""<!doctype html><html><head><meta charset='utf-8'><title>E-commerce Intelligence</title><style>body{font:16px system-ui;background:#07131f;color:#e9f3ff;max-width:880px;margin:50px auto;padding:0 24px}h1{font-size:42px;margin-bottom:6px}p{color:#a9bbcc}section{background:#102333;padding:24px;border-radius:16px;margin:20px 0}input{width:100%;box-sizing:border-box;padding:10px;margin:5px 0;border-radius:7px;border:1px solid #33516a}button{margin-top:12px;background:#43d19e;border:0;border-radius:8px;padding:12px 16px;font-weight:700;cursor:pointer}pre{white-space:pre-wrap;color:#97f4cc}</style></head><body><h1>AI Commerce Decision Console</h1><p>Production-style conversion inference, customer intelligence and model monitoring.</p><section><h2>Purchase prediction</h2><form id='form'><input name='user_id' value='123' type='number' placeholder='Customer ID'><input name='session_duration' value='300' type='number' placeholder='Session duration seconds'><input name='views' value='20' type='number' placeholder='Product views'><input name='cart_actions' value='3' type='number' placeholder='Cart actions'><input name='previous_purchases' value='5' type='number' placeholder='Previous purchases'><button>Predict Purchase Probability</button></form><pre id='result'>Ready.</pre></section><section><h2>Platform APIs</h2><p><a href='/docs'>Interactive API documentation</a> · <a href='/model_monitoring'>Model monitoring</a> · <a href='/customer_segment/123'>Customer segment example</a></p></section><script>document.querySelector('#form').onsubmit=async e=>{e.preventDefault();let x=Object.fromEntries(new FormData(e.target));for(let k in x)x[k]=Number(x[k]);let r=await fetch('/predict_conversion',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(x)});document.querySelector('#result').textContent=JSON.stringify(await r.json(),null,2)}</script></body></html>""")


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
