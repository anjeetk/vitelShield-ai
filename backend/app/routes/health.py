from fastapi import APIRouter
from app.services.model_loader import model_loader

router = APIRouter()

@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_loader.is_loaded
    }
