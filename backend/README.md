# ReviewSense Python API

The API stores products and reviews in SQLite, trains a TF-IDF + Logistic Regression sentiment classifier from `data/product_reviews_mock_data.csv`, and persists the trained model as `sentiment-model.joblib`. CSV ratings of 1–2 train the negative class; ratings of 4–5 train the positive class; neutral 3-star rows are excluded. The included archive contains 1,000 reviews (798 used for binary training, 202 neutral rows skipped). Curated examples in `model.py` supplement the dataset.

From the project root, install the Python requirements and start the API:

```powershell
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload --port 8000
```

The web app connects to `http://127.0.0.1:8000` by default. Set `NEXT_PUBLIC_REVIEWSENSE_API_URL` to change the API URL. `GET /api/health` reports service status, `GET /api/products` returns products and saved reviews, and `POST /api/products/{product_id}/reviews` classifies and stores a review.

The classifier uses scikit-learn because PySpark and a Java runtime are not available in the current environment. For a PySpark deployment, install Java and PySpark, then replace the classifier implementation in `model.py` with Spark MLlib Logistic Regression.
