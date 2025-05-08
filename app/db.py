import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import pandas as pd
from app.config import (
    get_mongodb_uri,
    MONGODB_DB,
    MONGODB_REVIEWS_COLLECTION,
    MONGODB_BUSINESSES_COLLECTION,
    MONGODB_USERS_COLLECTION
)

logger = logging.getLogger(__name__)

# MongoDB client instance
_mongo_client = None

def get_mongo_client():
    """Get or create MongoDB client with connection pooling"""
    global _mongo_client
    if _mongo_client is None:
        try:
            uri = get_mongodb_uri()
            logger.info(f"Connecting to MongoDB at {uri}")
            _mongo_client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            
            # Verify connection
            _mongo_client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
    return _mongo_client

def get_database():
    """Get MongoDB database"""
    client = get_mongo_client()
    return client[MONGODB_DB]

def get_reviews_collection():
    """Get reviews collection"""
    db = get_database()
    return db[MONGODB_REVIEWS_COLLECTION]

def get_businesses_collection():
    """Get businesses collection"""
    db = get_database()
    return db[MONGODB_BUSINESSES_COLLECTION]

def get_users_collection():
    """Get users collection"""
    db = get_database()
    return db[MONGODB_USERS_COLLECTION]

def fetch_reviews_as_dataframe():
    """Fetch reviews from MongoDB and convert to DataFrame"""
    try:
        collection = get_reviews_collection()
        reviews = list(collection.find({}, {
            "_id": 1,
            "userId": 1,
            "businessId": 1,
            "rating": 1,
            "comment": 1,
            "createdAt": 1
        }))
        
        if not reviews:
            logger.warning("No reviews found in MongoDB")
            return pd.DataFrame()
        
        # Convert MongoDB documents to DataFrame
        df = pd.DataFrame(reviews)
        
        # Rename fields to match expected format
        df = df.rename(columns={
            "_id": "review_id",
            "userId": "user_id",
            "businessId": "business_id",
            "comment": "text"
        })
        
        logger.info(f"Fetched {len(df)} reviews from MongoDB")
        return df
    except Exception as e:
        logger.error(f"Error fetching reviews from MongoDB: {str(e)}")
        raise

def fetch_businesses_as_dataframe():
    """Fetch businesses from MongoDB and convert to DataFrame"""
    try:
        collection = get_businesses_collection()
        businesses = list(collection.find({}, {
            "_id": 1,
            "name": 1,
            "description": 1,
            "category": 1,
            "location": 1,
            "address": 1,
            "city": 1,
            "rating": 1,
            "businessType": 1
        }))
        
        if not businesses:
            logger.warning("No businesses found in MongoDB")
            return pd.DataFrame()
        
        # Convert MongoDB documents to DataFrame
        df = pd.DataFrame(businesses)
        
        # Rename fields to match expected format
        df = df.rename(columns={
            "_id": "business_id",
            "businessType": "business_type"
        })
        
        # Extract city from address if not present
        if 'city' not in df.columns and 'address' in df.columns:
            df['city'] = df['address'].apply(
                lambda x: x.get('city', '') if isinstance(x, dict) else ''
            )
        
        # Extract category from businessType if not present
        if 'category' not in df.columns and 'business_type' in df.columns:
            df['category'] = df['business_type']
        
        logger.info(f"Fetched {len(df)} businesses from MongoDB")
        return df
    except Exception as e:
        logger.error(f"Error fetching businesses from MongoDB: {str(e)}")
        raise

def fetch_users_as_dataframe():
    """Fetch users from MongoDB and convert to DataFrame"""
    try:
        collection = get_users_collection()
        users = list(collection.find({}, {
            "_id": 1,
            "name": 1,
            "email": 1,
            "preferences": 1
        }))
        
        if not users:
            logger.warning("No users found in MongoDB")
            return pd.DataFrame()
        
        # Convert MongoDB documents to DataFrame
        df = pd.DataFrame(users)
        
        # Rename fields to match expected format
        df = df.rename(columns={
            "_id": "user_id"
        })
        
        logger.info(f"Fetched {len(df)} users from MongoDB")
        return df
    except Exception as e:
        logger.error(f"Error fetching users from MongoDB: {str(e)}")
        raise
