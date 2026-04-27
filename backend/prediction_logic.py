from datetime import timedelta
from sqlalchemy.orm import Session
import core.models as models
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model.pkl")

LABEL_MAP = {
    0: ("Low", 90.0),
    1: ("Medium", 75.0),
    2: ("High", 85.0)
}



# Features
def create_features(readings):
    speeds = [r.speed for r in readings]
    counts = [r.vehicle_count for r in readings]
    density = [r.density for r in readings]

    return [[
        np.mean(speeds),
        min(speeds),
        max(speeds),
        np.mean(counts),
        np.mean(density),
        speeds[-1] - speeds[0],
        counts[-1] - counts[0]
    ]]



# Fallback 
def fallback_prediction(readings):
    avg_speed = sum(r.speed for r in readings) / len(readings)

    if avg_speed < 20.0:
        return "High", 85.0
    elif avg_speed < 40.0:
        return "Medium", 75.0
    else:
        return "Low", 90.0



# Main Prediction Function

def generate_prediction(node_id: str, db: Session):

    recent_readings = db.query(models.TrafficReading)\
        .filter(models.TrafficReading.node_id == node_id)\
        .order_by(models.TrafficReading.timestamp.desc())\
        .limit(5).all()

    if not recent_readings:
        return None

    # Ensure correct time order (oldest → newest)
    recent_readings = recent_readings[::-1]

    latest_time = recent_readings[-1].timestamp
    future_time = latest_time + timedelta(hours=1)

    # Try ML model
    try:
        if not os.path.exists(MODEL_PATH):
            raise Exception("Model not trained")

        model = joblib.load(MODEL_PATH)

        features = create_features(recent_readings)
        pred = model.predict(features)[0]

        level, confidence = LABEL_MAP[pred]

    except Exception:
        # fallback if anything fails
        level, confidence = fallback_prediction(recent_readings)

    # Save to DB
    prediction = models.CongestionPrediction(
        node_id=node_id,
        target_time=future_time,
        predicted_level=level,
        confidence_score=confidence
    )

    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    return prediction