from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from celebtwin.params import *
from celebtwin.ml_logic.registry import load_model
from celebtwin.ml_logic.preprocessor import preprocess_features

app = FastAPI()
#app.state.model = load_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?param=X
@app.get("/predict")
def predict(
        param: str,  # 2014-07-06 19:18:00
    ):
    """Make a single course prediction."""
    X_processed = None
    #y_pred = app.state.model.predict(X_processed)
    y_pred = 'pong'
    return {'ping': y_pred}

@app.get("/")
def root():
    return {'celebtwin': 'ok'}
