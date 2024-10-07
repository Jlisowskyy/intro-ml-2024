from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

from .frontend import frontend_page
from .models import ModelResponse

app = FastAPI()

@app.get("/")
async def root():
    return HTMLResponse(frontend_page)

@app.post("/run/model")
async def run_model(file: UploadFile = File(...)) -> ModelResponse:
    try:
        if file.content_type != "audio/wav":
            return ModelResponse(response="Invalid file type. Please upload a WAV file.")

        contents = await file.read()
        print(f"Received file of size: {len(contents)} bytes")

        return ModelResponse(response="SUCCESS")
    except Exception as e:
        return ModelResponse(response=f"Error processing file: {str(e)}")
