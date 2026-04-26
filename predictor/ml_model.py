import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from predictor.features import create_features
from .config import MODEL_PATH


def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42
        ))
    ])


def prepare_training_data(df, window=5):
    X, y = [], []

    for i in range(window, len(df)):
        chunk = df.iloc[i-window:i]

        readings = chunk.to_dict("records")

        # convert dict → object-like access
        class R: pass
        objs = []
        for r in readings:
            o = R()
            o.speed = r["speed"]
            o.vehicle_count = r["vehicle_count"]
            o.density = r["density"]
            objs.append(o)

        features = create_features(objs)

        avg_speed = chunk["speed"].mean()

        if avg_speed < 20:
            label = 2
        elif avg_speed < 40:
            label = 1
        else:
            label = 0

        X.append(features)
        y.append(label)

    return X, y


def train_model(df):
    X, y = prepare_training_data(df)

    model = build_pipeline()
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    return model


def load_model():
    return joblib.load(MODEL_PATH)