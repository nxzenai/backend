from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.core.exceptions.custom import AIStudioException
from app.core.logging.logger import logger
from app.shared.responses import ErrorResponse

def register_exception_handlers(app: FastAPI):

    @app.exception_handler(AIStudioException)
    async def ai_studio_exception_handler(
        request: Request,
        exc: AIStudioException,
    ):

        logger.error(exc.message)

        return JSONResponse(
    status_code=exc.status_code,
    content=ErrorResponse(
        message=exc.message,
        error_code=exc.error_code,
        details=exc.details,
    ).model_dump(),
)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ):

        logger.error(exc.errors())

        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "message": "Validation Error",
                "error_code": "VALIDATION_ERROR",
                "details": exc.errors(),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ):

        logger.exception(exc)

        return JSONResponse(
    status_code=500,
    content=ErrorResponse(
        message="Internal Server Error",
        error_code="INTERNAL_SERVER_ERROR",
    ).model_dump(),
)