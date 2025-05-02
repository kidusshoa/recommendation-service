# recommendation-service

recommendation engine for khanut-local-business-finder

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

```bash
pip install fastapi uvicorn pandas scikit-learn surprise joblib
```

or

```bash
pip install -r requirements.txt
```

#to start

```bash
uvicorn app.main:app --reload --port 5000
```

# Recommendation Service

## API Endpoints

### 1. Get Recommendations

**URL**: `/api/v1/recommendations/{user_id}`  
**Method**: `GET`  
**Query Parameters**:

- `limit` (int): Number of recommendations to return (default: 5, min: 1, max: 20)

**Response**:

```json
{
  "user_id": "string",
  "recommendations": [
    {
      "business_id": "string",
      "name": "string",
      "category": "string",
      "predicted_rating": 4.5
    }
  ]
}
```
