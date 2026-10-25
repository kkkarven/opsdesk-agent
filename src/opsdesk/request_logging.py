import json
import logging
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request

logger = logging.getLogger("uvicorn.error")

def install_request_logging(app:FastAPI):
    @app.middleware("http")
    async def log_request(request:Request,call_next):
        request_id = uuid4().hex
        request.state.request_id = request_id

        started = perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (perf_counter() - started) *1000

            logger.exception(
                "request_failed request_id = %s method = %s"
                "path = %s duartion_ms = %.2f",
                request_id,
                request.method,
                request.url.path,
                duration_ms,
            )

            raise

        duration_ms = round(
            (perf_counter() - started) *1000,
            2,
        )
        response.headers["X-Request-ID"] = request_id

        message = json.dumps(
            {
                "event":"http_request",
                "request_id":request_id,
                "method":request.method,
                "path":request.url.path,
                "status":response.status_code,
                "duration":duration_ms,
            },
            ensure_ascii=False,
        )

        if response.status_code>= 500:
            logger.error(message)
        else:
            logger.info(message)

        return response