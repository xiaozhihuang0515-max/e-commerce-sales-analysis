# Verification checklist

Run the following after installing dependencies:

```bash
python src/main.py --train-demo
uvicorn main:app --app-dir src
```

Then verify:

1. `GET /api/health` returns `status: ok`.
2. `POST /predict_conversion` returns a probability, intent, and recommended action for valid numeric input.
3. `GET /customer_segment/123` returns an RFM segment after demo training.
4. `GET /model_monitoring` returns ROC-AUC, accuracy, precision, recall, and F1 metrics.
5. Import a small CSV subset and confirm invalid data is counted rather than accepted.
