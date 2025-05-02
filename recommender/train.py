import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import joblib
import os
from typing import List
from datetime import datetime
import pickle

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
    latest_model_path = os.path.join(model_dir, "latest_model.pkl")
    
    # Save the model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    # Update symlink
    if os.path.exists(latest_model_path):
        os.remove(latest_model_path)
    os.symlink(model_path, latest_model_path)

def train_model():
    validate_data("data/reviews.csv", ["user_id", "business_id", "rating"])
    validate_data("data/businesses.csv", ["business_id", "name", "category"])

    df = pd.read_csv("data/reviews.csv")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["user_id", "business_id", "rating"]], reader)

    trainset, _ = train_test_split(data, test_size=0.2)
    model = SVD()
    model.fit(trainset)

    save_model_with_version(model)
