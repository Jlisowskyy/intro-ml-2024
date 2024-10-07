from fastapi import FastAPI
from .frontend import frontend_page
from fastapi.responses import HTMLResponse

from .models import ModelRequest, ModelResponse

app = FastAPI()

@app.get("/")
async def root():
    return HTMLResponse(frontend_page)


@app.post("/run/model")
async def run_model(item: ModelRequest) -> ModelResponse:
    return ModelResponse(response="SUCCESS")

