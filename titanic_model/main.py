from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Define model
model = joblib.load("titanic_model.pkl")


# Define Feature Shape
class Feature(BaseModel):
    SibSp: int
    Parch: int
    Age: int
    Fare: int
    Sex: int
    Pclass_1: int
    Pclass_2: int
    Pclass_3: int
    Embarked_C: int
    Embarked_Q: int
    Embarked_S: int


def passenger_to_numpy(p: Feature) -> np.ndarray:
    return np.array(
        [
            [
                p.SibSp,
                p.Parch,
                p.Age,
                p.Fare,
                p.Sex,
                p.Pclass_1,
                p.Pclass_2,
                p.Pclass_3,
                p.Embarked_C,
                p.Embarked_Q,
                p.Embarked_S,
            ]
        ]
    )


@app.post("/knn/predict")
async def predict_fn(feature: Feature):
    X = passenger_to_numpy(feature)
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return {"survived": prediction, "probability": round(probability, 3)}
