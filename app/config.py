import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "khanut")
MONGODB_REVIEWS_COLLECTION = os.getenv("MONGODB_REVIEWS_COLLECTION", "reviews")
MONGODB_BUSINESSES_COLLECTION = os.getenv("MONGODB_BUSINESSES_COLLECTION", "businesses")
MONGODB_USERS_COLLECTION = os.getenv("MONGODB_USERS_COLLECTION", "users")

# Model Configuration
MODEL_DIR = os.getenv("MODEL_DIR", "models")
DATA_DIR = os.getenv("DATA_DIR", "data")

# API Configuration
API_PREFIX = os.getenv("API_PREFIX", "/api/v1")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "default-webhook-secret")

# Recommendation Configuration
DEFAULT_RECOMMENDATIONS = int(os.getenv("DEFAULT_RECOMMENDATIONS", "5"))
MAX_RECOMMENDATIONS = int(os.getenv("MAX_RECOMMENDATIONS", "20"))

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def get_mongodb_uri():
    """Get MongoDB URI with error handling"""
    uri = MONGODB_URI
    if not uri:
        logger.warning("MongoDB URI not set, using default localhost connection")
        return "mongodb://localhost:27017"
    return uri
