"""
predictor.py  (drop-in replacement)
------------------------------------
Two public functions:

  predict(readings, node_id)
      → same response shape the backend already uses (backwards-compatible)

  predict_with_route(readings, node_id, destination, congestion_map)
      → congestion prediction + ranked alternative routes
"""

from .features import create_features
from .preprocessing import remove_anomalies
from .ml_model import load_model
from .fallback import fallback_prediction
from .config import LABEL_MAP, WINDOW_SIZE
from .nairobi_graph import find_alternative_routes

_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            _model = load_model()
            print("[Predictor] ML model loaded.")
        except Exception as e:
            print(f"[Predictor] Model unavailable ({e}). Using fallback.")
    return _model


def _predict_level(readings):
    """Core logic. Returns (level, confidence, used_ml)."""
    if len(readings) < WINDOW_SIZE:
        level, conf = fallback_prediction(readings)
        return level, conf, False

    window  = remove_anomalies(readings[-WINDOW_SIZE:])
    if len(window) < 2:
        level, conf = fallback_prediction(readings[-WINDOW_SIZE:])
        return level, conf, False

    model = _get_model()
    if model is None:
        level, conf = fallback_prediction(window)
        return level, conf, False

    try:
        features = create_features(window)
        pred     = model.predict([features])[0]
        level, conf = LABEL_MAP[pred]
        return level, conf, True
    except Exception as e:
        print(f"[Predictor] ML error: {e}")
        level, conf = fallback_prediction(window)
        return level, conf, False


# ── backwards-compatible function (backend calls this already) ────────────────

def predict(readings, node_id):
    """
    Drop-in replacement for the old predict().
    Returns the same dict shape the backend expects, plus 'used_ml'.
    """
    level, conf, used_ml = _predict_level(readings)
    return {
        "node_id":          node_id,
        "predicted_level":  level,
        "confidence_score": conf,
        "used_ml":          used_ml,
    }


# ── new function: prediction + route suggestions ──────────────────────────────

def predict_with_route(readings, node_id, destination, congestion_map=None):
    """
    Predict congestion at node_id AND return ranked alternative routes
    to destination, avoiding congested nodes.

    Parameters
    ----------
    readings       : list of ORM objects with .speed, .vehicle_count, .density
    node_id        : str  e.g. "ST-THIKA-002"
    destination    : str  e.g. "ST-EXP-005"  (or "CBD")
    congestion_map : dict  node_id → "Low"|"Medium"|"High"
                     Pass predictions for ALL nodes you have so the router
                     can avoid congested corridors end-to-end.

    Returns
    -------
    {
        "node_id":           "ST-THIKA-002",
        "predicted_level":   "High",
        "confidence_score":  65.0,
        "used_ml":           True,
        "destination":       "ST-EXP-005",
        "alternative_routes": [
            {
                "display_path":        ["ST-THIKA-002", "ST-EXP-002", ...],
                "estimated_time":      28.5,
                "congestion_segments": ["ST-THIKA-002"],
                "summary":             "ST-THIKA-002 → ST-EXP-002 → ...  (~29 min)"
            }, ...
        ],
        "recommendation": "High congestion at ST-THIKA-002. Best route: ..."
    }
    """
    level, conf, used_ml = _predict_level(readings)

    cong_map = dict(congestion_map or {})
    cong_map[node_id] = level

    result = {
        "node_id":          node_id,
        "predicted_level":  level,
        "confidence_score": conf,
        "used_ml":          used_ml,
        "destination":      destination,
    }

    if node_id and destination and node_id != destination:
        routes = find_alternative_routes(
            start=node_id,
            end=destination,
            congestion_map=cong_map,
            top_n=3,
        )
        result["alternative_routes"] = routes

        if routes:
            best = routes[0]
            if level in ("Medium", "High"):
                result["recommendation"] = (
                    f"{level} congestion at {node_id}. "
                    f"Recommended route: {best['summary']}"
                )
            else:
                result["recommendation"] = (
                    f"Traffic is flowing well at {node_id}. "
                    f"Fastest route: {best['summary']}"
                )
        else:
            result["recommendation"] = (
                f"No route found from {node_id} to {destination}. "
                "Check both nodes are on the road network."
            )
    else:
        result["alternative_routes"] = []
        result["recommendation"]     = "Congestion prediction only — no destination provided."

    return result