from fastapi import APIRouter

from adapter.http.v1.route import assistant_route


def router():
    api_router = APIRouter()
    api_router.include_router(
        assistant_route, prefix="/v1", 
        tags=["assistant-router"]
    )

    return api_router
