# E-commerce Intelligence Platform

An end-to-end ML service that turns e-commerce behaviour events into conversion predictions, RFM customer segments, and action recommendations. It is intentionally designed as a backend-and-ML-engineering portfolio project—not a static analytics dashboard.

## Why it matters

Teams need to decide **who to target**, **which customers need retention**, and **how a trained model becomes an application**. This project answers those questions through a batch ETL workflow, relational storage, deployed FastAPI inference, and an AI decision console.

## Data evidence

The supplied Kaggle-style event archives remain outside this repository at `/Users/xiaozhihuang/Desktop/archive/`. To establish a reproducible baseline without loading the 13.7 GB source data into memory, I profiled the first 250,000 rows from each source file (500,000 events total).

| Metric | Observed value |
| --- | ---: |
| Source files | `2019-Oct.csv`, `2019-Nov.csv` |
| Event sample | 500,000 rows |
| Unique users | 94,653 |
| Unique sessions | 120,448 |
| Unique products | 52,812 |
| Views | 483,557 (96.71%) |
| Cart events | 7,314 (1.46%) |
| Purchases | 9,129 (1.83%) |
| View-to-cart session rate | 3.86% |
| Session purchase rate | 6.57% |
| Median product price | 160.57 |
| 95th-percentile price | 1,000.77 |

These are **sample metrics**, not claims about the entire October–November dataset. The ingestion command processes arbitrary source files with `pandas.read_csv(..., chunksize=...)`, so all-data metrics can be regenerated safely.

## Architecture

```text
Raw CSV events -> batch validation/cleaning -> SQL database -> customer features
                                                               |             |
                                                        conversion model   RFM + K-Means
                                                               \             /
                                                                FastAPI inference API
                                                                         |
                                                               AI decision console
```

## Project layout

```text
ecommerce-intelligence-platform/
├── src/main.py             # Application, ETL, ORM, training, APIs, decision UI
├── data/README.md          # Dataset contract and import guidance
├── tests/README.md         # Validation checklist
├── requirements.txt        # Runtime dependencies
├── RESUME_PROJECT.md       # Data-supported portfolio/resume wording
└── README.md               # Project overview and operations guide
```

The deliberately single-source implementation makes this submission easy to review while retaining production-oriented boundaries in code: validation, persistence, feature generation, training, inference and monitoring.

## APIs

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Database and customer-row health check |
| `POST` | `/predict_conversion` | Predict purchase probability and recommended action |
| `GET` | `/customer_segment/{user_id}` | Return RFM-driven customer segment |
| `GET` | `/model_monitoring` | Surface trained model metrics and request count |
| `GET` | `/docs` | OpenAPI interface |

## Run locally

cd ~/Desktop/spotify-eda-project

bash ecommerce-intelligence-platform/start_ecommerce.sh


Then open `http://127.0.0.1:8001` for the API.

## Import real event data

```bash
python src/main.py --ingest /Users/xiaozhihuang/Desktop/archive/2019-Oct.csv --chunksize 100000
```

The import validates required columns, normalizes timestamps and prices, rejects invalid rows, removes duplicates within each batch, and stores events in the database. Online APIs query the database and model artifacts; they never read raw CSV files.

## Deployment notes

Set `DATABASE_URL` to a PostgreSQL SQLAlchemy URL in production, for example `postgresql+psycopg://USER:PASSWORD@HOST:5432/ecommerce`. Persist the database and the generated `artifacts/` directory, then run Uvicorn behind a reverse proxy or container platform. Add migrations, authentication, background retraining, feature-store versioning, and model-drift alerts before production use.
