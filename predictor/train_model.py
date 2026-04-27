import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from core.database import SessionLocal
from core import models
from predictor.ml_model import train_model
import joblib

db = SessionLocal()

def load_data():
    readings = db.query(models.TrafficReading).all()

    return pd.DataFrame([{
        "speed": r.speed,
        "vehicle_count": r.vehicle_count,
        "density": r.density
    } for r in readings])


def train():
    df = load_data()

    if len(df) < 50:
        raise Exception("Not enough data")

    model = train_model(df)
    print("Model trained and saved")


if __name__ == "__main__":
    train()