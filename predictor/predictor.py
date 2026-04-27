from .features import create_features
from .preprocessing import remove_anomalies
from .ml_model import load_model
from .fallback import fallback_prediction
from .config import LABEL_MAP, WINDOW_SIZE

model = None


def predict(readings, node_id):
    global model 
    if model is None:
        model = load_model()
        print ("Model loaded:", model is not None)

    if len(readings) < WINDOW_SIZE:
        level, conf = fallback_prediction(readings)
        return {
            "node_id": node_id,
            "predicted_level": level,
            "confidence_score": conf
        }

    readings = readings[-WINDOW_SIZE:]
    readings = remove_anomalies(readings)

    try:

        features = create_features(readings)
        print("FEATURES:", features)

        pred = model.predict([features])[0]
        print("PRED:", pred)

        level, conf = LABEL_MAP[pred]

    except Exception as e:
        print("ML FAILED:", e)
        level, conf = fallback_prediction(readings)

    return {
        "node_id": node_id,
        "predicted_level": level,
        "confidence_score": conf
    }