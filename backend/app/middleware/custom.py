from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request

class CustomMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Adding a simple middleware to satisfy the requirement
        # "make sure for I need also middleware folder"
        response = await call_next(request)
        response.headers["X-Custom-Header"] = "Bangladesh-Legal-Advisor-V2"
        return response
