from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import logging

from app.config import settings
from app.routes import health, model, prediction
from app.services.model_loader import model_loader
from app.utils.validation import validation_exception_handler, global_exception_handler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Starting up API and loading models...")
    try:
        model_loader.load_models()
    except Exception as e:
        logger.error(f"Failed to initialize models during startup: {e}")
        # Not exiting so the health endpoint can report the failure
    yield
    # Shutdown logic
    logger.info("Shutting down API...")

from fastapi.responses import RedirectResponse

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

# Exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router, tags=["health"])
app.include_router(model.router, prefix="/model", tags=["model info"])
app.include_router(prediction.router, prefix="/predict", tags=["prediction"])
