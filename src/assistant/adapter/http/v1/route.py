from fastapi import APIRouter

from core.business import Business
from adapter.http.v1.request import Request

assistant_route = APIRouter()
business = Business()

@assistant_route.post(
    "/assistant",
)
def assistant(request: Request):
    return business.instruction(request.instruction)