# from tempfile import SpooledTemporaryFile
from enum import Enum
from functools import cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing_extensions import TypedDict

from celebtwin.logic.ann import ANNReader
from celebtwin.logic.annenums import ANNBackend, Detector, Model
from celebtwin.logic.preproc_face import NoFaceDetectedError
from celebtwin.logic.registry import load_latest_experiment

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


class FaceModel(str, Enum):
    """Models available for facial recognition."""
    facenet = "facenet"
    vggface = "vggface"

    @property
    def deepface_model(self) -> Model:
        return _deepface_mapping[self]


_deepface_mapping = {
    FaceModel.facenet: Model.FACENET,
    FaceModel.vggface: Model.VGG_FACE
}


class ErrorResponse(TypedDict):
    status: Literal["error"]
    error: str
    message: str


ClassNameResponse = TypedDict(
    "ClassNameResponse",
    {"status": Literal["ok"], "class": str, "name": str})


@app.post("/predict-nn/{model}")
@app.post("/predict-nn/")
# /predict-annory is deprecated.
@app.post("/predict-annoy/{model}")
@app.post("/predict-annoy/")
def predict_ann(file: UploadFile, model: FaceModel = FaceModel.facenet) \
        -> ClassNameResponse | ErrorResponse:
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.file.read())
        reader = load_ann(model)
        try:
            class_, name = reader.find_image(Path(temp_file.name))
        except NoFaceDetectedError:
            return {
                "status": "error",
                "error": "NoFaceDetectedError",
                "message": "No face detected in the image"}
        return {"status": "ok", "class": class_, "name": name}


@cache
def load_ann(model: FaceModel) -> ANNReader:
    # Load the production index. We do not support unloading. Resources are
    # released when the process exits.
    strategy = ANNBackend.BRUTE_FORCE.strategy_class(
        Detector.SKIP, model.deepface_model)
    return ANNReader(strategy)


@app.on_event("startup")
def preload_ann():
    print("Preloading models...")
    from deepface.modules.modeling import build_model  # type: ignore
    for model in FaceModel:
        build_model(
            task="facial_recognition", model_name=model.deepface_model)
        load_ann(model)


@app.get("/")
def root():
    return {"celebtwin": "ok"}
