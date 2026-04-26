def create_features(readings):
    speeds = [r.speed for r in readings]
    counts = [r.vehicle_count for r in readings]
    density = [r.density for r in readings]

    return [
        sum(speeds) / len(speeds),         # avg_speed
        min(speeds),                       # min_speed
        max(speeds),                       # max_speed
        sum(counts) / len(counts),         # avg_count
        sum(density) / len(density),       # avg_density
        speeds[-1] - speeds[0],            # speed_trend
        counts[-1] - counts[0]             # count_trend
    ]