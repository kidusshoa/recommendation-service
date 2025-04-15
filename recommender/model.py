import pandas as pd
import joblib
from surprise import SVD
from app.models import RecommendationResponse, Business

def recommend_for_user(user_id: str, top_n: int = 5) -> RecommendationResponse:
    model = joblib.load("models/recommender.pkl")
    df = pd.read_csv("data/reviews.csv")
    businesses = pd.read_csv("data/businesses.csv")

    rated = df[df["user_id"] == user_id]["business_id"].unique()
    all_biz = businesses["business_id"].unique()
    candidates = [b for b in all_biz if b not in rated]

    predictions = []
    for biz in candidates:
        pred = model.predict(user_id, biz)
        biz_data = businesses[businesses["business_id"] == biz].iloc[0]
        predictions.append((
            biz,
            pred.est,
            biz_data["name"] if "name" in biz_data else None,
            biz_data["rating"] if "rating" in biz_data else None
        ))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_recs = predictions[:top_n]
    
    recommendations = [
        Business(
            business_id=biz_id,
            name=name,
            rating=rating,
            predicted_rating=round(pred_rating, 2)
        )
        for biz_id, pred_rating, name, rating in top_recs
    ]

    return RecommendationResponse(
        user_id=user_id,
        recommendations=recommendations
    )
