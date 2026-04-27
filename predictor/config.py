import os

LABEL_MAP = {
    0: ("Low", 90.0),
    1: ("Medium", 75.0),
    2: ("High", 85.0)
}

WINDOW_SIZE = 5
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model.pkl")