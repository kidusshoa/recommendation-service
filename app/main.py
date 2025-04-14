from fastapi import FastAPI
from app.api import router

app = FastAPI(title="Business Recommender API")

app.include_router(router)
