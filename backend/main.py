"""ReviewSense API: SQLite-backed product reviews and sentiment analysis."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .model import DATASET_PATH, load_model, predict

DB_PATH = Path(__file__).parent / "reviewsense.sqlite3"
MODEL = None

PRODUCTS = [
    ("macbook", "MacBook Air M3", "Computers", "$1,099", "Light on your bag. Big on everything else.", "4.8"),
    ("headphones", "Sony WH-1000XM5", "Audio", "$349", "Your world, just the way you want to hear it.", "4.6"),
    ("kindle", "Kindle Paperwhite", "Reading", "$159", "A little light for all the books you love.", "4.7"),
    ("airpods", "AirPods Pro (2nd gen)", "Audio", "$249", "A sound experience you have to hear.", "4.5"),
    ("watch", "Apple Watch Series 10", "Wearables", "$399", "Thinner. Smarter. A brilliant way to stay connected.", "4.7"),
    ("bose", "Bose QuietComfort Ultra", "Audio", "$429", "Lose yourself in the music, not the noise.", "4.6"),
    ("ipad", "iPad Air 11-inch", "Tablets", "$599", "A brilliant canvas for work, play, and everything between.", "4.7"),
    ("dyson", "Dyson V15 Detect", "Home", "$749", "Reveals the dust you never knew was there.", "4.5"),
    ("fujifilm", "Fujifilm X100VI", "Cameras", "$1,599", "A little camera for the moments that matter.", "4.8"),
    ("stanley", "Stanley Quencher H2.0", "Lifestyle", "$45", "Your hydration sidekick, wherever the day takes you.", "4.6"),
]

SEED_REVIEWS = {
    "macbook": [("Olivia Bennett", "OB", "Incredibly fast and so light. The battery easily lasts my whole workday."), ("Marcus Chen", "MC", "Beautiful display and a great keyboard. It feels like a real upgrade."), ("Priya Shah", "PS", "Lovely laptop, but I wish it had more ports without carrying a hub."), ("Ethan Brooks", "EB", "Quiet, quick, and the screen is gorgeous. Could not be happier.")],
    "headphones": [("Sofia Martinez", "SM", "Noise cancellation is fantastic on my commute. Super comfortable too."), ("Jordan Lee", "JL", "The sound is lovely, but the touch controls are frustratingly unreliable."), ("Ava Wilson", "AW", "The quality is poor and it broke quickly. Disappointed for the price."), ("Noah Kim", "NK", "Wonderful sound, excellent battery, and I forget I am wearing them.")],
    "kindle": [("Amelia Foster", "AF", "I have finished more books this month than all of last year. Love it."), ("Lucas Moore", "LM", "The screen is easy on the eyes and the adjustable warmth is a dream."), ("Grace Park", "GP", "Battery is amazing. Page turns can lag a little when it is cold out.")],
    "airpods": [("Mia Thompson", "MT", "They pair instantly and the sound quality is genuinely impressive."), ("Daniel Wright", "DW", "The fit is perfect and transparency mode feels almost magical."), ("Chloe Davis", "CD", "One earbud stopped charging after a few months. Not what I expected.")],
    "watch": [("Isabella Clark", "IC", "The health features have helped me take better care of myself."), ("Benjamin Hall", "BH", "Love the brighter display, but the battery barely makes it to bedtime."), ("Harper Allen", "HA", "Fast, elegant, and surprisingly useful. My favorite upgrade in years.")],
    "bose": [("Ella Mitchell", "EM", "The noise cancellation is incredible on long flights. So comfortable."), ("Ryan Patel", "RP", "Great sound, but the app is confusing and the case feels too bulky."), ("Lily Cooper", "LC", "Beautifully balanced audio. These are my favorite headphones.")],
    "ipad": [("Zoe Anderson", "ZA", "Light, fast, and perfect for sketching on the go. Love the display."), ("Oliver Reed", "OR", "The screen is great, but storage fills up much too quickly."), ("Nina Garcia", "NG", "Excellent performance and battery. It replaced my old laptop for travel.")],
    "dyson": [("Mason Turner", "MT", "It picks up everything, even pet hair. Cleaning feels effortless now."), ("Layla Scott", "LS", "Works well but it is heavy, and the battery does not last long enough."), ("Henry Adams", "HA", "The laser shows every bit of dust. Excellent suction and easy to empty.")],
    "fujifilm": [("Ruby Collins", "RC", "The colors are beautiful straight out of camera. I take it everywhere."), ("Theo Martin", "TM", "Incredible image quality, but autofocus can be frustratingly slow."), ("Aria Lewis", "AL", "A joy to use. Compact, sharp, and the film simulations are fantastic.")],
    "stanley": [("Eva Brooks", "EB", "Keeps my water cold all day and fits in my car cup holder. Love it."), ("Jack Wilson", "JW", "The handle is handy, but the lid leaks if it tips over in my bag."), ("Maya Patel", "MP", "Great size and easy to clean. I actually drink enough water now.")],
}


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database() -> None:
    with connect() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS products (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL,
                price TEXT NOT NULL, description TEXT NOT NULL, rating TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT NOT NULL REFERENCES products(id),
                author TEXT NOT NULL, initials TEXT NOT NULL, text TEXT NOT NULL,
                sentiment TEXT NOT NULL CHECK(sentiment IN ('Positive','Negative')),
                confidence INTEGER NOT NULL, created_at TEXT NOT NULL
            );
        """)
        db.executemany("INSERT OR IGNORE INTO products VALUES (?,?,?,?,?,?)", PRODUCTS)
        existing = db.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        if existing == 0:
            now = datetime.now(timezone.utc).isoformat()
            for product_id, reviews in SEED_REVIEWS.items():
                for author, initials, text in reviews:
                    label, confidence = predict(MODEL, text)
                    db.execute("INSERT INTO reviews (product_id,author,initials,text,sentiment,confidence,created_at) VALUES (?,?,?,?,?,?,?)", (product_id, author, initials, text, label, confidence, now))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global MODEL
    MODEL = load_model()
    initialize_database()
    yield


app = FastAPI(title="ReviewSense API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"], allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


class ReviewInput(BaseModel):
    text: str = Field(min_length=3, max_length=500)


def serialize_review(row: sqlite3.Row) -> dict:
    try:
        created = datetime.fromisoformat(row["created_at"])
        age = (datetime.now(timezone.utc) - created).total_seconds()
        display_date = "Just now" if age < 60 else "Today" if age < 86400 else created.strftime("%b %d").replace(" 0", " ")
    except (ValueError, TypeError):
        display_date = "Recently"
    return {"id": row["id"], "author": row["author"], "initials": row["initials"], "date": display_date, "text": row["text"], "sentiment": row["sentiment"], "confidence": row["confidence"]}


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "model": "Logistic Regression", "storage": "SQLite"}


@app.get("/api/products")
def list_products() -> list[dict]:
    with connect() as db:
        products = db.execute("SELECT * FROM products ORDER BY rowid").fetchall()
        result = []
        for product in products:
            reviews = db.execute("SELECT * FROM reviews WHERE product_id=? ORDER BY created_at DESC, id DESC", (product["id"],)).fetchall()
            result.append({**dict(product), "reviews": [serialize_review(review) for review in reviews]})
        return result


@app.get("/api/dataset/reviews")
def list_dataset_reviews(offset: int = Query(0, ge=0), limit: int = Query(12, ge=1, le=100)) -> dict:
    if not DATASET_PATH.exists():
        raise HTTPException(status_code=404, detail="Training dataset is not installed.")
    import csv

    with DATASET_PATH.open(encoding="utf-8-sig", newline="") as dataset_file:
        rows = [
            {
                "id": row["ReviewID"],
                "product_id": row["ProductID"],
                "user_id": row["UserID"],
                "rating": int(row["Rating"]),
                "text": row["ReviewText"],
                "date": row["ReviewDate"],
            }
            for row in csv.DictReader(dataset_file)
            if row.get("ReviewText", "").strip()
        ]
    return {"total": len(rows), "offset": offset, "limit": limit, "reviews": rows[offset:offset + limit]}


@app.post("/api/products/{product_id}/reviews", status_code=201)
def create_review(product_id: str, payload: ReviewInput) -> dict:
    text = payload.text.strip()
    if len(text) < 3:
        raise HTTPException(status_code=422, detail="Write at least 3 characters for your review.")
    label, confidence = predict(MODEL, text)
    if label not in ("Positive", "Negative"):
        label = "Positive"
    created_at = datetime.now(timezone.utc).isoformat()
    with connect() as db:
        if not db.execute("SELECT 1 FROM products WHERE id=?", (product_id,)).fetchone():
            raise HTTPException(status_code=404, detail="Product not found.")
        cursor = db.execute("INSERT INTO reviews (product_id,author,initials,text,sentiment,confidence,created_at) VALUES (?,?,?,?,?,?,?)", (product_id, "You", "YO", text, label, confidence, created_at))
        saved = db.execute("SELECT * FROM reviews WHERE id=?", (cursor.lastrowid,)).fetchone()
        return serialize_review(saved)
