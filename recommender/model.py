import pandas as pd
import joblib
import os
import logging
from typing import Dict, List, Any, Tuple
from surprise import SVD
from app.models import RecommendationResponse, Business
from recommender.content_based import ContentBasedRecommender, build_user_profile

logger = logging.getLogger(__name__)

def load_collaborative_model(model_path="models/recommender.pkl"):
    """Load the collaborative filtering model"""
    try:
        if not os.path.exists(model_path):
            logger.warning(f"Collaborative model file {model_path} not found")
            return None

        model = joblib.load(model_path)
        logger.info(f"Collaborative model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading collaborative model: {str(e)}")
        return None

def load_content_model(model_path="models/content_recommender.pkl"):
    """Load the content-based model"""
    try:
        model = ContentBasedRecommender.load(model_path)
        if model:
            logger.info(f"Content-based model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading content-based model: {str(e)}")
        return None

def get_collaborative_recommendations(user_id: str, top_n: int = 5) -> List[Tuple[str, float, str, float]]:
    """Get recommendations using collaborative filtering"""
    try:
        # Load model and data
        model = load_collaborative_model()
        if not model:
            logger.warning("Collaborative model not available")
            return []

        df = pd.read_csv("data/reviews.csv")
        businesses = pd.read_csv("data/businesses.csv")

        # Find businesses the user hasn't rated yet
        rated = df[df["user_id"] == user_id]["business_id"].unique()
        all_biz = businesses["business_id"].unique()
        candidates = [b for b in all_biz if b not in rated]

        # Generate predictions
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

        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_n]
    except Exception as e:
        logger.error(f"Error getting collaborative recommendations: {str(e)}")
        return []

def get_content_recommendations(user_id: str, top_n: int = 5) -> List[Tuple[str, float, str, float]]:
    """Get recommendations using content-based filtering"""
    try:
        # Load content model
        content_model = load_content_model()
        if not content_model:
            logger.warning("Content-based model not available")
            return []

        # Load data
        reviews_df = pd.read_csv("data/reviews.csv")
        businesses_df = pd.read_csv("data/businesses.csv")

        # Build user profile
        user_profile = build_user_profile(user_id, reviews_df, businesses_df)

        if not user_profile:
            logger.warning(f"No profile could be built for user {user_id}")
            return []

        # Get content-based recommendations
        cb_recs = content_model.recommend_for_user_profile(user_profile, top_n=top_n)

        # Format recommendations to match collaborative format
        formatted_recs = []
        for biz_id, score in cb_recs:
            biz_data = businesses_df[businesses_df["business_id"] == biz_id]
            if not biz_data.empty:
                formatted_recs.append((
                    biz_id,
                    score,
                    biz_data["name"].iloc[0] if "name" in biz_data else None,
                    biz_data["rating"].iloc[0] if "rating" in biz_data else None
                ))

        return formatted_recs
    except Exception as e:
        logger.error(f"Error getting content recommendations: {str(e)}")
        return []

def recommend_for_user(user_id: str, top_n: int = 5) -> RecommendationResponse:
    """
    Get hybrid recommendations for a user combining collaborative and content-based filtering

    Args:
        user_id: User ID to get recommendations for
        top_n: Number of recommendations to return

    Returns:
        RecommendationResponse with hybrid recommendations
    """
    logger.info(f"Getting recommendations for user {user_id}")

    # Get collaborative filtering recommendations
    cf_recommendations = get_collaborative_recommendations(user_id, top_n=top_n*2)

    # Get content-based recommendations
    cb_recommendations = get_content_recommendations(user_id, top_n=top_n*2)

    # Combine recommendations with a simple ensemble approach
    combined_scores = {}
    business_data = {}

    # Add collaborative filtering recommendations with weight 0.7
    for biz_id, score, name, rating in cf_recommendations:
        combined_scores[biz_id] = 0.7 * score
        business_data[biz_id] = (name, rating)

    # Add content-based recommendations with weight 0.3
    for biz_id, score, name, rating in cb_recommendations:
        if biz_id in combined_scores:
            combined_scores[biz_id] += 0.3 * score
        else:
            combined_scores[biz_id] = 0.3 * score
            business_data[biz_id] = (name, rating)

    # Sort by combined score
    sorted_recommendations = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    # Format recommendations
    recommendations = []
    for biz_id, score in sorted_recommendations:
        name, rating = business_data.get(biz_id, (None, None))
        recommendations.append(
            Business(
                business_id=str(biz_id),
                name=name,
                rating=rating,
                predicted_rating=round(score, 2)
            )
        )

    logger.info(f"Returning {len(recommendations)} hybrid recommendations for user {user_id}")
    return RecommendationResponse(
        user_id=user_id,
        recommendations=recommendations
    )
