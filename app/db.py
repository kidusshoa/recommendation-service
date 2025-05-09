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
            "authorId": 1,  # For regular reviews
            "customerId": 1, # For service reviews
            "businessId": 1,
            "rating": 1,
            "comment": 1,
            "text": 1,      # Some reviews might use text instead of comment
            "createdAt": 1,
            "status": 1
        }))

        if not reviews:
            logger.warning("No reviews found in MongoDB")
            return pd.DataFrame()

        # Convert MongoDB documents to DataFrame
        df = pd.DataFrame(reviews)

        # Handle different user ID field names
        if 'userId' in df.columns:
            df['user_id'] = df['userId']
        elif 'authorId' in df.columns:
            df['user_id'] = df['authorId']
        elif 'customerId' in df.columns:
            df['user_id'] = df['customerId']
        else:
            # Create a placeholder user ID if none exists
            df['user_id'] = 'unknown_user'

        # Handle business ID
        if 'businessId' in df.columns:
            df['business_id'] = df['businessId']
        else:
            # Create a placeholder business ID if none exists
            df['business_id'] = 'unknown_business'

        # Handle review text
        if 'comment' in df.columns:
            df['text'] = df['comment']
        elif 'text' not in df.columns:
            df['text'] = ''

        # Filter out non-approved reviews if status is available
        if 'status' in df.columns:
            df = df[df['status'] == 'approved']

        # Convert ObjectId to string for user_id and business_id
        df['user_id'] = df['user_id'].astype(str)
        df['business_id'] = df['business_id'].astype(str)

        # Ensure we have the required columns
        required_columns = ['user_id', 'business_id', 'rating']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Required column {col} not found in reviews data")
                return pd.DataFrame()

        # Select only the columns we need
        df = df[['user_id', 'business_id', 'rating', 'text']]

        logger.info(f"Fetched {len(df)} reviews from MongoDB")
        return df
    except Exception as e:
        logger.error(f"Error fetching reviews from MongoDB: {str(e)}")
        raise

def fetch_businesses_as_dataframe():
    """Fetch businesses from MongoDB and convert to DataFrame"""
    try:
        collection = get_businesses_collection()
        businesses = list(collection.find({
            "approved": True  # Only fetch approved businesses
        }, {
            "_id": 1,
            "name": 1,
            "description": 1,
            "category": 1,
            "businessType": 1,
            "location": 1,
            "address": 1,
            "city": 1,
            "rating": 1,
            "approved": 1,
            "status": 1
        }))

        if not businesses:
            logger.warning("No businesses found in MongoDB")
            return pd.DataFrame()

        # Convert MongoDB documents to DataFrame
        df = pd.DataFrame(businesses)

        # Convert ObjectId to string for business_id
        df['_id'] = df['_id'].astype(str)

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
        elif 'city' not in df.columns:
            df['city'] = 'Unknown'

        # Extract category from businessType if not present
        if 'category' not in df.columns and 'business_type' in df.columns:
            df['category'] = df['business_type']
        elif 'category' not in df.columns:
            df['category'] = 'Unknown'

        # Ensure rating is present
        if 'rating' not in df.columns:
            df['rating'] = 0.0

        # Ensure description is present
        if 'description' not in df.columns:
            df['description'] = ''

        # Filter by status if available
        if 'status' in df.columns:
            df = df[df['status'] == 'active']

        # Ensure we have the required columns
        required_columns = ['business_id', 'name', 'category']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Required column {col} not found in businesses data")
                return pd.DataFrame()

        # Select only the columns we need
        df = df[['business_id', 'name', 'category', 'description', 'city', 'rating']]

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
