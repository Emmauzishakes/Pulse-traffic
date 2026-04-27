def fallback_prediction(readings):
    speeds = [r.speed for r in readings]
    avg_speed = sum(speeds) / len(speeds)

    if avg_speed < 20:
        return "High", 85.0
    elif avg_speed < 40:
        return "Medium", 75.0
    else:
        return "Low", 90.0