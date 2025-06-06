from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# from tempfile import SpooledTemporaryFile
from pathlib import Path

from celebtwin.params import LOCAL_DOWNLOAD_IMAGES_PATH
from celebtwin.ml_logic.registry import load_latest_experiment

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
    assert file.filename is not None
    filepath_to_save = Path(LOCAL_DOWNLOAD_IMAGES_PATH) / file.filename
    contents = file.file.read()
    with open(filepath_to_save, "wb") as file_to_write:
        file_to_write.write(contents)

    experiment = load_latest_experiment()
    pred, class_name = experiment.predict(filepath_to_save)

    return {
        "result": class_name,
        "model": model,
        "filename": file.filename,
        "probas" : pred.tolist(),
        "classes" : experiment._dataset.class_names
    }


@app.get("/")
def root():
    return {"celebtwin": "ok"}
