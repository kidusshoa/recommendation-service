import os
import logging
import shutil
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Request
from typing import List
from app.models import RecommendationResponse, TrainingResponse, Business
from recommender.model import recommend_for_user
from recommender.train import train_model
from app.config import MODEL_DIR, DATA_DIR
from app.db import get_mongo_client, get_database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate required files at startup
REQUIRED_FILES = [
    os.path.join(MODEL_DIR, "recommender.pkl"),
    os.path.join(MODEL_DIR, "content_recommender.pkl"),
    os.path.join(DATA_DIR, "reviews.csv"),
    os.path.join(DATA_DIR, "businesses.csv")
]

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Check if files exist, but don't fail startup if they don't
for file in REQUIRED_FILES:
    if not os.path.exists(file):
        logger.warning(f"Required file '{file}' is missing. Some endpoints may not work until data is uploaded.")

# Check MongoDB connection at startup
try:
    client = get_mongo_client()
    db = get_database()
    logger.info(f"Successfully connected to MongoDB database: {db.name}")
except (ConnectionFailure, ServerSelectionTimeoutError) as e:
    logger.warning(f"Failed to connect to MongoDB: {str(e)}. Will use local CSV files if available.")

router = APIRouter(prefix="/api/v1", tags=["recommendations"])

@router.get("/recommendations/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(
    user_id: str,
    limit: int = Query(default=5, ge=1, le=20, description="Number of recommendations to return"),
    method: str = Query(default="hybrid", description="Recommendation method: 'hybrid', 'collaborative', or 'content'"),
    use_mongodb: bool = Query(default=True, description="Whether to use MongoDB data or local CSV files")
):
    try:
        logger.info(f"Fetching {method} recommendations for user_id={user_id} with limit={limit} using MongoDB={use_mongodb}")

        # Check if MongoDB is available when requested
        if use_mongodb:
            try:
                client = get_mongo_client()
                db = get_database()
                logger.info(f"Successfully connected to MongoDB for recommendations")
            except Exception as db_error:
                logger.warning(f"MongoDB connection failed, falling back to CSV files: {str(db_error)}")
                use_mongodb = False

        # Import the appropriate recommendation function based on method
        if method == "collaborative":
            from recommender.model import get_collaborative_recommendations
            raw_recommendations = get_collaborative_recommendations(user_id, top_n=limit, use_mongodb=use_mongodb)
            # Convert to RecommendationResponse format
            businesses = []
            for biz_id, score, name, rating in raw_recommendations:
                businesses.append(
                    Business(
                        business_id=str(biz_id),
                        name=name,
                        rating=rating,
                        predicted_rating=round(score, 2)
                    )
                )
            recommendations = RecommendationResponse(
                user_id=user_id,
                recommendations=businesses
            )
        elif method == "content":
            from recommender.model import get_content_recommendations
            raw_recommendations = get_content_recommendations(user_id, top_n=limit, use_mongodb=use_mongodb)
            # Convert to RecommendationResponse format
            businesses = []
            for biz_id, score, name, rating in raw_recommendations:
                businesses.append(
                    Business(
                        business_id=str(biz_id),
                        name=name,
                        rating=rating,
                        predicted_rating=round(score, 2)
                    )
                )
            recommendations = RecommendationResponse(
                user_id=user_id,
                recommendations=businesses
            )
        else:  # hybrid (default)
            from recommender.model import recommend_for_user
            recommendations = recommend_for_user(user_id, top_n=limit, use_mongodb=use_mongodb)

        if not recommendations or len(recommendations.recommendations) == 0:
            logger.warning(f"No recommendations found for user_id={user_id} using method={method}")
            raise HTTPException(status_code=404, detail="No recommendations found")

        logger.info(f"Returning {len(recommendations.recommendations)} recommendations for user_id={user_id}")
        return recommendations
    except FileNotFoundError:
        logger.error("Model or data files not found")
        raise HTTPException(status_code=404, detail="Model or data files not found")
    except Exception as e:
        logger.error(f"Error fetching recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/retrain", response_model=TrainingResponse)
async def retrain(
    use_mongodb: bool = Query(default=True, description="Whether to use MongoDB data or local CSV files")
):
    try:
        logger.info(f"Starting model retraining using MongoDB={use_mongodb}...")

        # Check if MongoDB is available when requested
        if use_mongodb:
            try:
                _ = get_mongo_client()
                _ = get_database()
                logger.info("Successfully connected to MongoDB for retraining")
            except Exception as db_error:
                logger.warning(f"MongoDB connection failed, falling back to CSV files: {str(db_error)}")
                use_mongodb = False

        train_model(use_mongodb=use_mongodb)
        logger.info("Model retrained successfully.")
        return TrainingResponse(
            status="success",
            message=f"Model retrained successfully using {'MongoDB' if use_mongodb else 'CSV files'}"
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

@router.post("/webhook")
async def handle_webhook(request: Request):
    """
    Handle webhooks for real-time updates
    """
    try:
        # Verify webhook secret
        webhook_secret = request.headers.get("X-Webhook-Secret")
        expected_secret = os.environ.get("WEBHOOK_SECRET", "default-webhook-secret")

        if webhook_secret != expected_secret:
            logger.warning(f"Invalid webhook secret received")
            raise HTTPException(status_code=401, detail="Invalid webhook secret")

        # Parse webhook payload
        payload = await request.json()
        event = payload.get("event")
        data = payload.get("data")

        if not event or not data:
            logger.warning(f"Invalid webhook payload: {payload}")
            raise HTTPException(status_code=400, detail="Invalid webhook payload")

        logger.info(f"Received webhook: {event}")

        # Process different event types
        if event.startswith("review.") or event.startswith("serviceReview."):
            # Schedule model retraining
            # In a production environment, you might want to use a task queue
            # For simplicity, we'll just trigger retraining directly
            logger.info(f"Scheduling model retraining due to webhook: {event}")

            # Run in a separate thread to avoid blocking
            import threading
            thread = threading.Thread(target=lambda: train_model())
            thread.daemon = True
            thread.start()

        return {"status": "success", "message": f"Webhook {event} processed"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing webhook: {str(e)}")

@router.get("/health")
async def health_check():
    try:
        # Check if data files exist
        data_files_exist = all(os.path.exists(file) for file in [
            os.path.join(DATA_DIR, "reviews.csv"),
            os.path.join(DATA_DIR, "businesses.csv")
        ])

        # Check if model files exist
        model_files_exist = all(os.path.exists(file) for file in [
            os.path.join(MODEL_DIR, "recommender.pkl"),
            os.path.join(MODEL_DIR, "content_recommender.pkl")
        ])

        # Check MongoDB connection
        mongodb_connected = False
        try:
            client = get_mongo_client()
            db = get_database()
            # Ping the database
            db.command('ping')
            mongodb_connected = True
        except Exception as db_error:
            logger.warning(f"MongoDB health check failed: {str(db_error)}")
            mongodb_connected = False

        return {
            "status": "healthy",
            "data_files_exist": data_files_exist,
            "model_files_exist": model_files_exist,
            "mongodb_connected": mongodb_connected,
            "version": "1.1.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}
