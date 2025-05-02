import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import logging

logger = logging.getLogger(__name__)

class ContentBasedRecommender:
    """
    Content-based recommender using TF-IDF and cosine similarity
    """
    
    def __init__(self):
        self.business_profiles = None
        self.tfidf_matrix = None
        self.vectorizer = None
        self.business_indices = None
        
    def fit(self, businesses_df):
        """
        Build the content-based recommender model
        
        Args:
            businesses_df: DataFrame with business data including 'business_id', 'name', 
                          'category', 'description', etc.
        """
        logger.info("Building content-based recommender model...")
        
        # Create a copy of the dataframe to avoid modifying the original
        df = businesses_df.copy()
        
        # Fill NaN values
        df['description'] = df['description'].fillna('')
        df['category'] = df['category'].fillna('Unknown')
        df['city'] = df['city'].fillna('Unknown')
        
        # Create a combined text field for TF-IDF
        df['content'] = df['name'] + ' ' + df['category'] + ' ' + df['description'] + ' ' + df['city']
        
        # Create TF-IDF matrix
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(df['content'])
        
        # Create a mapping of business IDs to indices
        self.business_indices = pd.Series(df.index, index=df['business_id'])
        
        # Store the business profiles
        self.business_profiles = df
        
        logger.info(f"Content-based model built with {len(df)} businesses")
        return self
    
    def recommend_similar_businesses(self, business_id, top_n=5):
        """
        Recommend businesses similar to the given business
        
        Args:
            business_id: ID of the business to find similar businesses for
            top_n: Number of recommendations to return
            
        Returns:
            List of tuples (business_id, similarity_score)
        """
        # Check if the business exists in our data
        if business_id not in self.business_indices:
            logger.warning(f"Business ID {business_id} not found in content-based model")
            return []
        
        # Get the index of the business
        idx = self.business_indices[business_id]
        
        # Get the TF-IDF vector for this business
        business_vector = self.tfidf_matrix[idx]
        
        # Calculate cosine similarity between this business and all others
        sim_scores = cosine_similarity(business_vector, self.tfidf_matrix).flatten()
        
        # Get the indices of the top similar businesses (excluding itself)
        similar_indices = sim_scores.argsort()[:-top_n-1:-1]
        similar_indices = [i for i in similar_indices if i != idx][:top_n]
        
        # Map indices back to business IDs and include similarity scores
        similar_businesses = [(self.business_profiles.iloc[i]['business_id'], 
                              sim_scores[i]) for i in similar_indices]
        
        return similar_businesses
    
    def recommend_for_user_profile(self, user_profile, top_n=5):
        """
        Recommend businesses based on user profile
        
        Args:
            user_profile: Dict with user preferences (categories, cities, etc.)
            top_n: Number of recommendations to return
            
        Returns:
            List of tuples (business_id, similarity_score)
        """
        # Create a content string from user profile
        if not user_profile:
            logger.warning("Empty user profile provided")
            return []
        
        content = ' '.join([
            user_profile.get('preferred_categories', ''),
            user_profile.get('preferred_cities', ''),
            user_profile.get('interests', '')
        ])
        
        if not content.strip():
            logger.warning("User profile has no usable content")
            return []
        
        # Transform the user profile into TF-IDF space
        user_vector = self.vectorizer.transform([content])
        
        # Calculate cosine similarity between user profile and all businesses
        sim_scores = cosine_similarity(user_vector, self.tfidf_matrix).flatten()
        
        # Get the indices of the top similar businesses
        similar_indices = sim_scores.argsort()[:-top_n-1:-1]
        
        # Map indices back to business IDs and include similarity scores
        recommendations = [(self.business_profiles.iloc[i]['business_id'], 
                           sim_scores[i]) for i in similar_indices]
        
        return recommendations
    
    def save(self, filepath='models/content_recommender.pkl'):
        """Save the model to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Content-based recommender saved to {filepath}")
        
    @classmethod
    def load(cls, filepath='models/content_recommender.pkl'):
        """Load the model from disk"""
        if not os.path.exists(filepath):
            logger.warning(f"Model file {filepath} not found")
            return None
        
        try:
            model = joblib.load(filepath)
            logger.info(f"Content-based recommender loaded from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None

def build_user_profile(user_id, reviews_df, businesses_df):
    """
    Build a user profile based on their past interactions
    
    Args:
        user_id: ID of the user
        reviews_df: DataFrame with review data
        businesses_df: DataFrame with business data
        
    Returns:
        Dict with user preferences
    """
    # Get all reviews by this user
    user_reviews = reviews_df[reviews_df['user_id'] == user_id]
    
    if len(user_reviews) == 0:
        return {}
    
    # Get businesses the user has reviewed
    reviewed_businesses = businesses_df[businesses_df['business_id'].isin(user_reviews['business_id'])]
    
    # Extract categories and cities from reviewed businesses
    categories = reviewed_businesses['category'].dropna().tolist()
    cities = reviewed_businesses['city'].dropna().tolist()
    
    # Weight categories and cities by review ratings
    weighted_categories = {}
    weighted_cities = {}
    
    for _, review in user_reviews.iterrows():
        business_id = review['business_id']
        rating = review['rating']
        
        # Find the business
        business = businesses_df[businesses_df['business_id'] == business_id]
        if len(business) == 0:
            continue
            
        # Get category and city
        category = business['category'].iloc[0]
        city = business['city'].iloc[0]
        
        if pd.notna(category):
            weighted_categories[category] = weighted_categories.get(category, 0) + rating
            
        if pd.notna(city):
            weighted_cities[city] = weighted_cities.get(city, 0) + rating
    
    # Sort by weight and convert to space-separated string
    preferred_categories = ' '.join([
        cat for cat, _ in sorted(weighted_categories.items(), key=lambda x: x[1], reverse=True)
    ])
    
    preferred_cities = ' '.join([
        city for city, _ in sorted(weighted_cities.items(), key=lambda x: x[1], reverse=True)
    ])
    
    return {
        'preferred_categories': preferred_categories,
        'preferred_cities': preferred_cities,
        'interests': ' '.join(categories)  # Simple approach: all categories are interests
    }
