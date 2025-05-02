from pydantic import BaseModel
from typing import List, Optional

class Business(BaseModel):
    business_id: str
    name: Optional[str] = None
    rating: Optional[float] = None
    predicted_rating: Optional[float] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "business_id": "123",
                    "name": "Example Business",
                    "rating": 4.5,
                    "predicted_rating": 4.2
                }
            ]
        }
    }

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[Business]

class TrainingResponse(BaseModel):
    status: str
    message: str