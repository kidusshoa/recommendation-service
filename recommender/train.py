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
from app.db import fetch_reviews_as_dataframe, fetch_businesses_as_dataframe
from app.config import DATA_DIR, MODEL_DIR

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

def save_model_with_version(model, model_dir=MODEL_DIR):
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

    logger.info(f"Model saved to {model_path} and {latest_model_path}")

def train_model(use_mongodb=True):
    """
    Train both collaborative filtering and content-based models

    Args:
        use_mongodb: Whether to fetch data from MongoDB (True) or use local CSV files (False)
    """
    logger.info("Starting model training...")

    try:
        if use_mongodb:
            logger.info("Fetching data from MongoDB...")
            # Fetch data from MongoDB
            reviews_df = fetch_reviews_as_dataframe()
            businesses_df = fetch_businesses_as_dataframe()

            # Save data to CSV for backup and compatibility
            os.makedirs(DATA_DIR, exist_ok=True)
            reviews_df.to_csv(os.path.join(DATA_DIR, "reviews.csv"), index=False)
            businesses_df.to_csv(os.path.join(DATA_DIR, "businesses.csv"), index=False)
            logger.info(f"Data saved to CSV files in {DATA_DIR}")
        else:
            # Validate data files
            validate_data(os.path.join(DATA_DIR, "reviews.csv"), ["user_id", "business_id", "rating"])
            validate_data(os.path.join(DATA_DIR, "businesses.csv"), ["business_id", "name", "category"])

            # Load data from CSV files
            reviews_df = pd.read_csv(os.path.join(DATA_DIR, "reviews.csv"))
            businesses_df = pd.read_csv(os.path.join(DATA_DIR, "businesses.csv"))
            logger.info("Data loaded from CSV files")

        # Check if we have enough data
        if len(reviews_df) < 10 or len(businesses_df) < 5:
            logger.warning(f"Not enough data for training: {len(reviews_df)} reviews, {len(businesses_df)} businesses")
            return

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
        cb_model.save(os.path.join(MODEL_DIR, 'content_recommender.pkl'))
        logger.info("Content-based model trained and saved")

        logger.info("Model training completed successfully")
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise
