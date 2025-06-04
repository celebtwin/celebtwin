from typing import Annotated
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# from tempfile import SpooledTemporaryFile
from pathlib import Path

from celebtwin.params import LOCAL_DOWNLOAD_IMAGES_PATH
from celebtwin.ml_logic.registry import load_model
from celebtwin.ml_logic.data import load_image

app = FastAPI()
# app.state.model = load_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.post("/setmodel/")
def setmodel(model_version: str):
    """Set active model"""
    # app.state.model = load_model()
    return {"model_version": model_version, "params": "todef"}


@app.post("/predict/")
async def create_upload_file(file: UploadFile, model: str | None = None):
    Path(LOCAL_DOWNLOAD_IMAGES_PATH).mkdir(parents=True, exist_ok=True)
    filepath_to_save = Path(LOCAL_DOWNLOAD_IMAGES_PATH) / file.filename
    contents = file.file.read()
    with open(filepath_to_save, "wb") as file_to_write:
        file_to_write.write(contents)
    img = load_image(path=filepath_to_save, image_size=64, num_channels=1, resize="pad")

    # TODO : call predict + process response
    return {"result": "result", "model": model, "filename": file.filename}


@app.get("/")
def root():
    return {"celebtwin": "ok"}
