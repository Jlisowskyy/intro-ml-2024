"""
Author: Jakub Lisowski, 2024

FastAPI webserver providing a web interface for the model
"""
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

from src.frontend.app_state import AppState
from src.frontend.models import ModelResponse
from src.cnn.model_api import classify_file

app = FastAPI()
app_state = AppState()


@app.get("/")
async def root():
    """GET / response"""
    return HTMLResponse(app_state.page)


@app.post("/run/model/wav")
async def run_model(file: UploadFile = File(...)) -> ModelResponse:
    """POST wave file response"""
    try:
        if file.content_type != "audio/wav":
            return ModelResponse(
                response="Invalid file type. Please upload a WAV file.")

        contents = await file.read()

        Path("uploaded_files").mkdir(exist_ok=True)
        with open("uploaded_files/user-uploaded.wav", "wb") as audio_file:
            audio_file.write(contents)

        result = classify_file("uploaded_files/user-uploaded.wav", app_state.classifier)

        return ModelResponse(response=str(result))
    except Exception as e:  # pylint: disable=broad-exception-caught
        return ModelResponse(response=f"Error processing file: {str(e)}")
