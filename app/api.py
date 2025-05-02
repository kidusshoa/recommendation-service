import os
import logging
import shutil
from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from typing import List
from app.models import RecommendationResponse, TrainingResponse, Business
from recommender.model import recommend_for_user
from recommender.train import train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate required files at startup
REQUIRED_FILES = [
    "models/recommender.pkl",
    "data/reviews.csv",
    "data/businesses.csv"
]

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Check if files exist, but don't fail startup if they don't
for file in REQUIRED_FILES:
    if not os.path.exists(file):
        logger.warning(f"Required file '{file}' is missing. Some endpoints may not work until data is uploaded.")

router = APIRouter(prefix="/api/v1", tags=["recommendations"])

@router.get("/recommendations/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(
    user_id: str,
    limit: int = Query(default=5, ge=1, le=20, description="Number of recommendations to return")
):
    try:
        logger.info(f"Fetching recommendations for user_id={user_id} with limit={limit}")
        recommendations = recommend_for_user(user_id, top_n=limit)
        if not recommendations:
            logger.warning(f"No recommendations found for user_id={user_id}")
            raise HTTPException(status_code=404, detail="No recommendations found")
        logger.info(f"Returning {len(recommendations)} recommendations for user_id={user_id}")
        return recommendations
    except FileNotFoundError:
        logger.error("Model or data files not found")
        raise HTTPException(status_code=404, detail="Model or data files not found")
    except Exception as e:
        logger.error(f"Error fetching recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/retrain", response_model=TrainingResponse)
async def retrain():
    try:
        logger.info("Starting model retraining...")
        train_model()
        logger.info("Model retrained successfully.")
        return TrainingResponse(
            status="success",
            message="Model retrained successfully"
        )
    except ValueError as ve:
        logger.error(f"Data validation error during retraining: {str(ve)}")
        raise HTTPException(status_code=400, detail=f"Data validation error: {str(ve)}")
    except Exception as e:
        logger.error(f"Failed to train model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to train model: {str(e)}"
        )

@router.post("/data/upload", response_model=TrainingResponse)
async def upload_data(
    reviews: UploadFile = File(...),
    businesses: UploadFile = File(...)
):
    try:
        logger.info("Receiving data upload...")

        # Save reviews file
        with open("data/reviews.csv", "wb") as f:
            shutil.copyfileobj(reviews.file, f)

        # Save businesses file
        with open("data/businesses.csv", "wb") as f:
            shutil.copyfileobj(businesses.file, f)

        logger.info("Data files uploaded successfully")
        return TrainingResponse(
            status="success",
            message="Data files uploaded successfully"
        )
    except Exception as e:
        logger.error(f"Error uploading data files: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading data files: {str(e)}"
        )

@router.get("/health")
async def health_check():
    try:
        # Check if data files exist
        data_files_exist = all(os.path.exists(file) for file in ["data/reviews.csv", "data/businesses.csv"])

        # Check if model file exists
        model_file_exists = os.path.exists("models/recommender.pkl")

        return {
            "status": "healthy",
            "data_files_exist": data_files_exist,
            "model_file_exists": model_file_exists
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}
