# Khanut Recommendation Service

A recommendation engine for the Khanut platform that provides personalized business recommendations to users.

## Features

- Hybrid recommendation system combining collaborative filtering and content-based approaches
- MongoDB integration for real-time data access
- RESTful API for easy integration with other services
- Automatic model retraining via webhooks
- Fallback to CSV files when MongoDB is unavailable

## Setup

### Prerequisites

- Python 3.8+
- MongoDB 4.4+
- pip

### Installation

1. Clone the repository
2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure MongoDB connection in `.env` file:

```
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=khanut
MONGODB_REVIEWS_COLLECTION=reviews
MONGODB_BUSINESSES_COLLECTION=businesses
MONGODB_USERS_COLLECTION=users
```

### Running the Service

Start the service with:

```bash
uvicorn app.main:app --reload --port 5000
```

The API will be available at http://localhost:5000

## API Endpoints

### 1. Get Recommendations

**URL**: `/api/v1/recommendations/{user_id}`
**Method**: `GET`
**Query Parameters**:

- `limit` (int): Number of recommendations to return (default: 5, min: 1, max: 20)
- `method` (string): Recommendation method - "hybrid", "collaborative", or "content" (default: "hybrid")
- `use_mongodb` (boolean): Whether to use MongoDB data (true) or local CSV files (false) (default: true)

**Response**:

```json
{
  "user_id": "string",
  "recommendations": [
    {
      "business_id": "string",
      "name": "string",
      "rating": 4.2,
      "predicted_rating": 4.5
    }
  ]
}
```

### 2. Retrain Model

**URL**: `/api/v1/retrain`
**Method**: `POST`
**Query Parameters**:

- `use_mongodb` (boolean): Whether to use MongoDB data (true) or local CSV files (false) (default: true)

**Response**:

```json
{
  "status": "success",
  "message": "Model retrained successfully using MongoDB"
}
```

### 3. Upload Data (CSV files)

**URL**: `/api/v1/data/upload`
**Method**: `POST`
**Form Data**:

- `reviews`: CSV file with review data
- `businesses`: CSV file with business data

**Response**:

```json
{
  "status": "success",
  "message": "Data files uploaded successfully"
}
```

### 4. Health Check

**URL**: `/api/v1/health`
**Method**: `GET`

**Response**:

```json
{
  "status": "healthy",
  "data_files_exist": true,
  "model_files_exist": true,
  "mongodb_connected": true,
  "version": "1.1.0"
}
```

## MongoDB Integration

The recommendation service can fetch data directly from MongoDB. The following collections are used:

### Reviews Collection

Expected fields:

- `_id`: Review ID
- `userId`: User ID
- `businessId`: Business ID
- `rating`: Rating (1-5)
- `comment`: Review text
- `createdAt`: Creation timestamp

### Businesses Collection

Expected fields:

- `_id`: Business ID
- `name`: Business name
- `description`: Business description
- `category`: Business category
- `location`: Business location (GeoJSON)
- `address`: Business address
- `city`: City
- `rating`: Average rating
- `businessType`: Type of business

## Data Format

If using CSV files instead of MongoDB, the files should have the following format:

### reviews.csv

```
user_id,business_id,rating,text,date
user123,business456,4.5,Great service!,2023-01-01
```

### businesses.csv

```
business_id,name,category,description,city,rating
business456,Example Business,Restaurant,A great place to eat,Addis Ababa,4.2
```

## Environment Variables

All configuration can be set via environment variables or in the `.env` file:

- `MONGODB_URI`: MongoDB connection string
- `MONGODB_DB`: MongoDB database name
- `MONGODB_REVIEWS_COLLECTION`: Collection name for reviews
- `MONGODB_BUSINESSES_COLLECTION`: Collection name for businesses
- `MONGODB_USERS_COLLECTION`: Collection name for users
- `MODEL_DIR`: Directory for model files
- `DATA_DIR`: Directory for data files
- `API_PREFIX`: API route prefix
- `WEBHOOK_SECRET`: Secret for webhook authentication
- `DEFAULT_RECOMMENDATIONS`: Default number of recommendations
- `MAX_RECOMMENDATIONS`: Maximum number of recommendations
