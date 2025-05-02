import logging
from fastapi import FastAPI
from app.api import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Khanut Recommendation Service",
    description="API for business and service recommendations",
    version="1.0.0",
)

# Include API router
app.include_router(router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to Khanut Recommendation Service",
        "docs": "/docs",
        "health": "/api/v1/health",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
