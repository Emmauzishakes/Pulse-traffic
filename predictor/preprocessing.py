import numpy as np

def remove_anomalies(readings):
    speeds = [r.speed for r in readings]

    if len(speeds) < 3:
        return readings

    mean = np.mean(speeds)
    std = np.std(speeds)

    return [
        r for r in readings
        if abs(r.speed - mean) <= 2 * std
    ] or readings