# from tempfile import SpooledTemporaryFile
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from celebtwin.ml_logic.preproc_face import NoFaceDetectedError
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
def predict(file: UploadFile, model: str | None = None):
    assert file.filename is not None
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.file.read())
        experiment = load_latest_experiment()
        try:
            pred, class_name = experiment.predict(Path(temp_file.name))
        except NoFaceDetectedError:
            return {"error": "NoFaceDetectedError",
                    "message": "No face detected in the image"}
    return {
        "result": class_name,
        "model": model,
        "filename": file.filename,
        "probas": pred.tolist(),
        "classes": experiment._dataset.class_names
    }


@app.get("/")
def root():
    return {"celebtwin": "ok"}
