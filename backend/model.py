"""Train and load the persisted ReviewSense sentiment model."""

from pathlib import Path
import csv
import hashlib

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_PATH = Path(__file__).parent / "sentiment-model.joblib"
DATASET_PATH = Path(__file__).parent / "data" / "product_reviews_mock_data.csv"


def dataset_fingerprint() -> str:
    if not DATASET_PATH.exists():
        return "no-dataset"
    return hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()

TRAINING_DATA = [
    ("I love this product, it is excellent and works perfectly", "Positive"),
    ("Beautiful design and amazing quality, highly recommend", "Positive"),
    ("Fast, reliable, comfortable, and easy to use", "Positive"),
    ("The battery lasts all day and the screen is gorgeous", "Positive"),
    ("Fantastic sound and wonderful noise cancellation", "Positive"),
    ("Great value, great performance, I am very happy", "Positive"),
    ("Setup was simple and the product works beautifully", "Positive"),
    ("Incredible quality, exceeded my expectations", "Positive"),
    ("A brilliant upgrade, smooth and dependable", "Positive"),
    ("Comfortable fit and excellent battery life", "Positive"),
    ("This is my favorite product, it is perfect", "Positive"),
    ("Impressive features, lovely display, fast delivery", "Positive"),
    ("Works exactly as described, would buy again", "Positive"),
    ("The quality is poor and it broke quickly", "Negative"),
    ("Very disappointed, this stopped working after a week", "Negative"),
    ("Terrible battery life and frustrating controls", "Negative"),
    ("Slow, unreliable, and difficult to set up", "Negative"),
    ("The product feels cheap and the screen is bad", "Negative"),
    ("It keeps disconnecting and customer support was unhelpful", "Negative"),
    ("Poor build quality, already broken, not worth the price", "Negative"),
    ("The app is confusing and the device fails constantly", "Negative"),
    ("Uncomfortable fit and disappointing sound quality", "Negative"),
    ("It stopped charging and now does not work at all", "Negative"),
    ("Awful experience, slow performance, would not recommend", "Negative"),
    ("The battery barely lasts and the product gets very hot", "Negative"),
    ("Missing parts, damaged packaging, and a long delay", "Negative"),
]


def train_model() -> Pipeline:
    training_data = list(TRAINING_DATA)
    if DATASET_PATH.exists():
        with DATASET_PATH.open(encoding="utf-8-sig", newline="") as dataset_file:
            for row in csv.DictReader(dataset_file):
                try:
                    rating = int(row["Rating"])
                    review = row["ReviewText"].strip()
                except (KeyError, TypeError, ValueError):
                    continue
                # Ratings 1–2 are negative and 4–5 are positive. Neutral 3-star
                # reviews are intentionally excluded from this binary model.
                if review and rating <= 2:
                    training_data.append((review, "Negative"))
                elif review and rating >= 4:
                    training_data.append((review, "Positive"))

    texts, labels = zip(*training_data)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    model.fit(list(texts), list(labels))
    model.reviewsense_dataset_fingerprint = dataset_fingerprint()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


def load_model() -> Pipeline:
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        if getattr(model, "reviewsense_dataset_fingerprint", None) == dataset_fingerprint():
            return model
    return train_model()


def predict(model: Pipeline, text: str) -> tuple[str, int]:
    probabilities = model.predict_proba([text])[0]
    index = int(probabilities.argmax())
    return str(model.classes_[index]), round(float(probabilities[index]) * 100)
