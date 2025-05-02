from fastapi.testclient import TestClient
from app.api import router
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)
client = TestClient(app)

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_get_recommendations():
    response = client.get("/api/v1/recommendations/test_user?limit=5")
    assert response.status_code in [200, 404]  # Depends on test data availability