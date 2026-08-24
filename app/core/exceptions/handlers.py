from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.exceptions.custom import AIStudioException
from app.core.logging.logger import logger
from app.shared.responses import ErrorResponse

def register_exception_handlers(app: FastAPI):

    @app.exception_handler(AIStudioException)
    async def ai_studio_exception_handler(
        request: Request,
        exc: AIStudioException,
    ):

        logger.warning("Handled application error code={} path={}", exc.error_code, request.url.path)

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

        logger.warning(
            "Request validation failed count={} path={}",
            len(exc.errors()),
            request.url.path,
        )

        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "message": "Validation Error",
                "error_code": "VALIDATION_ERROR",
                "details": jsonable_encoder(exc.errors()),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ):

        logger.error(
            "Unhandled application exception type={} path={}",
            type(exc).__name__,
            request.url.path,
        )

        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                message="Internal Server Error",
                error_code="INTERNAL_SERVER_ERROR",
            ).model_dump(),
        )
