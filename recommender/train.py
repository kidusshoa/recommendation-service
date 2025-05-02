import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import joblib
import os
import logging
from typing import List, Dict, Tuple
from datetime import datetime
import pickle
from recommender.content_based import ContentBasedRecommender

logger = logging.getLogger(__name__)

def validate_data(file_path: str, required_columns: List[str]) -> bool:
    try:
        df = pd.read_csv(file_path)
        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"Missing required column: {column}")
        return True
    except Exception as e:
        raise RuntimeError(f"Data validation failed for {file_path}: {str(e)}")

def save_model_with_version(model, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"recommender_{timestamp}.pkl")
    latest_model_path = os.path.join(model_dir, "recommender.pkl")

    # Save the model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Save a copy as the main model file
    with open(latest_model_path, "wb") as f:
        pickle.dump(model, f)

def train_model():
    """
    Train both collaborative filtering and content-based models
    """
    logger.info("Starting model training...")

    # Validate data files
    validate_data("data/reviews.csv", ["user_id", "business_id", "rating"])
    validate_data("data/businesses.csv", ["business_id", "name", "category"])

    # Load data
    reviews_df = pd.read_csv("data/reviews.csv")
    businesses_df = pd.read_csv("data/businesses.csv")

    # Train collaborative filtering model (SVD)
    logger.info("Training collaborative filtering model...")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(reviews_df[["user_id", "business_id", "rating"]], reader)

    trainset, _ = train_test_split(data, test_size=0.2)
    cf_model = SVD()
    cf_model.fit(trainset)

    # Save collaborative filtering model
    save_model_with_version(cf_model)
    logger.info("Collaborative filtering model trained and saved")

    # Train content-based model
    logger.info("Training content-based model...")
    cb_model = ContentBasedRecommender()
    cb_model.fit(businesses_df)

    # Save content-based model
    cb_model.save('models/content_recommender.pkl')
    logger.info("Content-based model trained and saved")

    logger.info("Model training completed successfully")
