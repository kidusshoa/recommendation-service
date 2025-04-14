import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import joblib
import os

def train_model():
    df = pd.read_csv("data/reviews.csv")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["user_id", "business_id", "rating"]], reader)

    trainset, _ = train_test_split(data, test_size=0.2)
    model = SVD()
    model.fit(trainset)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/recommender.pkl")
