import logging
from enum import Enum
from functools import cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Literal, TypedDict

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import logger
from .logic.annenums import ANNBackend, Detector, Model

if TYPE_CHECKING:
    from .logic.ann import ANNReader


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


@app.post("/predict/")
def predict(file: UploadFile, model: str | None = None):
    from .logic.preproc_face import NoFaceDetectedError
    from .logic.registry import load_latest_experiment
    assert file.filename is not None
    with NamedTemporaryFile() as temp_file:
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
# /predict-annoy is deprecated.
@app.post("/predict-annoy/{model}")
@app.post("/predict-annoy/")
def predict_ann(file: UploadFile, model: FaceModel = FaceModel.facenet) \
        -> ClassNameResponse | ErrorResponse:
    from .logic.preproc_face import NoFaceDetectedError
    with NamedTemporaryFile() as temp_file:
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


class SupportedDetector(str, Enum):
    builtin = "builtin"
    skip = "skip"


@app.post("/detect/{model}")
def detect(file: UploadFile, model: SupportedDetector):
    import dataclasses

    from .logic import detection
    with NamedTemporaryFile() as temp_file:
        temp_file.write(file.file.read())
        faces = detection.detect_faces(model, Path(temp_file.name))
    return {"status": "ok", "faces": [dataclasses.asdict(face) for face in faces]}


@cache
def load_ann(model: FaceModel) -> 'ANNReader':
    # Load the production index. We do not support unloading. Resources are
    # released when the process exits.
    from .logic.ann import ANNReader
    strategy = ANNBackend.BRUTE_FORCE.strategy_class(
        Detector.BUILTIN, model.deepface_model)
    return ANNReader(strategy)


@app.on_event("startup")
def preload_ann():
    _configure_logging()
    from . import preload
    logger.info("Preloading package")
    preload()
    from deepface.modules.modeling import build_model  # type: ignore
    for model in FaceModel:
        logger.info("Preloading model: %s", model.deepface_model.value)
        build_model(
            task="facial_recognition", model_name=model.deepface_model)
    for model in [FaceModel.facenet, FaceModel.vggface]:
        load_ann(model)


def _configure_logging():
    """Call after uvicorn app is started, to reuse their color logging."""
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,

        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(message)s",
                "use_colors": None,
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {
            "celebtwin": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False
            },
        },
    })


@app.get("/")
def root():
    return {"celebtwin": "ok"}
