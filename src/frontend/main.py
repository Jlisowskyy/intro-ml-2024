"""
FastAPI webserver providing a web interface for the model
"""
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

from src.frontend.logger import get_fastapi_logger
from src.frontend.models import ModelResponse

app = FastAPI()

logger = get_fastapi_logger()
logger.info('API is starting up')

index_path = Path.resolve(Path(f'{__file__}/../index.html'))

with open(index_path, 'r', encoding='utf-8') as f:
    FRONTEND_PAGE = f.read()


@app.get("/")
async def root():
    """GET / response"""
    return HTMLResponse(FRONTEND_PAGE)


@app.post("/run/model")
async def run_model(file: UploadFile = File(...)) -> ModelResponse:
    """POST wave file response"""
    try:
        if file.content_type != "audio/wav":
            return ModelResponse(
                response="Invalid file type. Please upload a WAV file.")

        contents = await file.read()
        logger.info("Received file of size: %d bytes", len(contents))

        Path("uploaded_files").mkdir(exist_ok=True)
        with open("uploaded_files/user-uploaded.wav", "wb") as audio_file:
            audio_file.write(contents)
        logger.info("File saved to uploaded_files/user-uploaded.wav")

        return ModelResponse(response="SUCCESS")
    except Exception as e:  # pylint: disable=broad-exception-caught
        return ModelResponse(response=f"Error processing file: {str(e)}")
