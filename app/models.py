from pydantic import BaseModel
from typing import List, Optional

class Business(BaseModel):
    business_id: str
    name: Optional[str] = None
    rating: Optional[float] = None
    predicted_rating: Optional[float] = None

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[Business]

class TrainingResponse(BaseModel):
    status: str
    message: str