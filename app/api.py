from fastapi import APIRouter, HTTPException, Query
from typing import List
from app.models import RecommendationResponse, TrainingResponse, Business
from recommender.model import recommend_for_user
from recommender.train import train_model

router = APIRouter(prefix="/api/v1", tags=["recommendations"])

@router.get("/recommendations/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(
    user_id: str,
    limit: int = Query(default=5, ge=1, le=20, description="Number of recommendations to return")
):
    try:
        recommendations = recommend_for_user(user_id, top_n=limit)
        return recommendations
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model or data files not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/retrain", response_model=TrainingResponse)
async def retrain():
    try:
        train_model()
        return TrainingResponse(
            status="success",
            message="Model retrained successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to train model: {str(e)}"
        )

@router.get("/health")
async def health_check():
    return {"status": "healthy"}
