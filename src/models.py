from pydantic import BaseModel

class ModelRequest(BaseModel):
    text: str

class ModelResponse(BaseModel):
    response: str