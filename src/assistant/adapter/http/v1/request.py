from pydantic import BaseModel

class Request(BaseModel):
    instruction: str  # Define um campo obrigatório que deve ser uma string