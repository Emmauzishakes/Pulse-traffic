from .features import create_features
from .preprocessing import remove_anomalies
from .ml_model import load_model
from .fallback import fallback_prediction
from .config import LABEL_MAP, WINDOW_SIZE

model = load_model()


def predict(readings, node_id):
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
        pred = model.predict([features])[0]

        level, conf = LABEL_MAP[pred]

    except Exception:
        level, conf = fallback_prediction(readings)

    return {
        "node_id": node_id,
        "predicted_level": level,
        "confidence_score": conf
    }