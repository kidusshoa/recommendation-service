import pandas as pd
import joblib
from surprise import SVD

def recommend_for_user(user_id: str, top_n: int = 5):
    model = joblib.load("models/recommender.pkl")
    df = pd.read_csv("data/reviews.csv")
    businesses = pd.read_csv("data/businesses.csv")

    rated = df[df["user_id"] == user_id]["business_id"].unique()
    all_biz = businesses["business_id"].unique()
    candidates = [b for b in all_biz if b not in rated]

    predictions = []
    for biz in candidates:
        pred = model.predict(user_id, biz)
        predictions.append((biz, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_recs = predictions[:top_n]
    
    return {
        "user_id": user_id,
        "recommendations": [{"business_id": b, "predicted_rating": r} for b, r in top_recs]
    }
