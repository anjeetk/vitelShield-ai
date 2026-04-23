from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging

logger = logging.getLogger(__name__)

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    formatted_errors = []
    for error in errors:
        formatted_errors.append({
            "loc": error.get("loc"),
            "msg": error.get("msg"),
            "type": error.get("type")
        })
        
    logger.warning(f"Validation error: {formatted_errors}")
    return JSONResponse(
        status_code=422,
        content={
            "error": True,
            "message": "Invalid input data",
            "details": formatted_errors
        },
    )

async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error during prediction",
            "details": []
        },
    )
