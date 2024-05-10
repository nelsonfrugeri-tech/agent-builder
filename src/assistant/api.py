import uvicorn
from fastapi import FastAPI

from adapter.http.v1.router import router

def api():
    fast_api = FastAPI(title="assistant-api", description="Assistant API")

    fast_api.include_router(router=router())

    return fast_api


if __name__ == "__main__":
    uvicorn.run(api(), host="0.0.0.0", port=8080)