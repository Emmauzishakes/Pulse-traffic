import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from core.database import SessionLocal
from core import models
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

db = SessionLocal()

def load_data():
    readings = db.query(models.TrafficReading).all()

    return pd.DataFrame([{
        "speed": r.speed,
        "vehicle_count": r.vehicle_count,
        "density": r.density
    } for r in readings])


def prepare_data(df, window=5):
    X, y = [], []

    for i in range(window, len(df)):
        chunk = df.iloc[i-window:i]

        avg_speed = chunk["speed"].mean()

        if avg_speed < 20:
            label = 2
        elif avg_speed < 40:
            label = 1
        else:
            label = 0

        features = [
            chunk["speed"].mean(),
            chunk["speed"].min(),
            chunk["speed"].max(),
            chunk["vehicle_count"].mean(),
            chunk["density"].mean(),
            chunk["speed"].iloc[-1] - chunk["speed"].iloc[0],
            chunk["vehicle_count"].iloc[-1] - chunk["vehicle_count"].iloc[0]
        ]

        X.append(features)
        y.append(label)

    return X, y


def train():
    df = load_data()

    if len(df) < 50:
        raise Exception("Not enough data")

    X, y = prepare_data(df)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=50, max_depth=5))
    ])

    model.fit(X, y)

    joblib.dump(model, "model.pkl")
    print("Model trained and saved")


if __name__ == "__main__":
    train()