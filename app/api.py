from fastapi import APIRouter
from recommender.model import recommend_for_user
from recommender.train import train_model

router = APIRouter()

@router.get("/recommendations/{user_id}")
def get_recommendations(user_id: str):
    return recommend_for_user(user_id)

@router.post("/retrain")
def retrain():
    train_model()
    return {"status": "model retrained successfully"}
