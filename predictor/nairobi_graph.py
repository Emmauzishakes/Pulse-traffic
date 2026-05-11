"""
nairobi_graph.py
----------------
Road network graph built around the ACTUAL node IDs your simulator uses:
 
  ST-THIKA-001..005   Thika Road       (CBD → Thika, northbound)
  ST-EXP-001..005     Expressway       (CBD → JKIA, southbound)
  ST-LANG-001..002    Langata Road     (CBD → Karen/Langata)
  ST-NGONG-001..003   Ngong Road       (CBD → Ngong)
 
Each node = one sensor station on that corridor.
Edges connect adjacent sensors on the same corridor (sequential travel)
AND cross-links between corridors (realistic Nairobi shortcuts).
 
Edge weight = base travel time in minutes under free-flow conditions.
Dijkstra inflates weights by a congestion multiplier at runtime.
"""
 
import heapq
 
# ---------------------------------------------------------------------------
# Corridor definitions  (node_id, position_km_from_cbd)
# ---------------------------------------------------------------------------
CORRIDORS = {
    "THIKA": {
        "nodes": [
            ("ST-THIKA-001",  2.0),   # Globe Roundabout area
            ("ST-THIKA-002",  5.5),   # Pangani / Muthaiga junction
            ("ST-THIKA-003",  9.0),   # Roysambu
            ("ST-THIKA-004", 13.5),   # Githurai
            ("ST-THIKA-005", 18.0),   # Near Thika Town
        ],
        "speed_kmh": 50,
    },
    "EXP": {
        "nodes": [
            ("ST-EXP-001",  3.0),    # Mlango Kubwa / Enterprise Rd
            ("ST-EXP-002",  7.0),    # Industrial Area
            ("ST-EXP-003", 11.5),    # Embakasi
            ("ST-EXP-004", 15.0),    # Syokimau
            ("ST-EXP-005", 19.0),    # Near JKIA
        ],
        "speed_kmh": 55,
    },
    "LANG": {
        "nodes": [
            ("ST-LANG-001",  4.5),   # Langata / Wilson Airport area
            ("ST-LANG-002",  9.0),   # Karen / Langata south
        ],
        "speed_kmh": 45,
    },
    "NGONG": {
        "nodes": [
            ("ST-NGONG-001",  3.5),  # Upperhill / Kilimani
            ("ST-NGONG-002",  7.0),  # Adams Arcade / Dagoretti
            ("ST-NGONG-003", 12.0),  # Ngong Town
        ],
        "speed_kmh": 45,
    },
}
 
# Virtual CBD hub — connects all corridors at their start
CBD_NODE = "CBD-HUB"
 
# ---------------------------------------------------------------------------
# Cross-links between corridors (realistic Nairobi shortcuts)
# (node_a, node_b, base_time_minutes)
# ---------------------------------------------------------------------------
CROSS_LINKS = [
    # Thika Rd ↔ Expressway  (via Outer Ring Road / Jogoo Road)
    ("ST-THIKA-001", "ST-EXP-001",  8),   # Globe ↔ Enterprise Rd
    ("ST-THIKA-002", "ST-EXP-002", 12),   # Pangani ↔ Industrial Area
    ("ST-THIKA-003", "ST-EXP-003", 14),   # Roysambu ↔ Embakasi
 
    # Thika Rd ↔ Ngong Rd  (via Waiyaki Way / Westlands)
    ("ST-THIKA-001", "ST-NGONG-001", 10), # Globe ↔ Upperhill
    ("ST-THIKA-002", "ST-NGONG-001", 15), # Pangani ↔ Upperhill
 
    # Ngong Rd ↔ Langata Rd  (via Adams Arcade / Mbagathi)
    ("ST-NGONG-001", "ST-LANG-001", 10),  # Upperhill ↔ Langata
    ("ST-NGONG-002", "ST-LANG-001",  8),  # Adams Arcade ↔ Langata
    ("ST-NGONG-002", "ST-LANG-002", 12),  # Adams Arcade ↔ Karen
    ("ST-NGONG-003", "ST-LANG-002", 10),  # Ngong ↔ Karen
 
    # Expressway ↔ Langata Rd  (JKIA / Wilson loop)
    ("ST-EXP-002", "ST-LANG-001",  8),    # Industrial ↔ Langata
    ("ST-EXP-005", "ST-LANG-002", 12),    # JKIA ↔ Karen
]
 
# ---------------------------------------------------------------------------
# Build adjacency graph
# ---------------------------------------------------------------------------
 
def _travel_time(dist_km, speed_kmh):
    return round((dist_km / speed_kmh) * 60, 1)
 
 
def build_graph():
    graph = {}
 
    def add_edge(a, b, t):
        graph.setdefault(a, []).append((b, t))
        graph.setdefault(b, []).append((a, t))
 
    # Connect CBD-HUB to first node of each corridor
    for corridor in CORRIDORS.values():
        first_node, first_km = corridor["nodes"][0]
        t = _travel_time(first_km, corridor["speed_kmh"])
        add_edge(CBD_NODE, first_node, t)
 
    # Connect sequential nodes within each corridor
    for corridor in CORRIDORS.values():
        nodes = corridor["nodes"]
        spd   = corridor["speed_kmh"]
        for i in range(len(nodes) - 1):
            na, ka = nodes[i]
            nb, kb = nodes[i + 1]
            t = _travel_time(abs(kb - ka), spd)
            add_edge(na, nb, t)
 
    # Cross-links
    for a, b, t in CROSS_LINKS:
        add_edge(a, b, t)
 
    return graph
 
 
# ---------------------------------------------------------------------------
# Congestion multipliers
# ---------------------------------------------------------------------------
CONGESTION_MULTIPLIER = {
    "Low":    1.0,
    "Medium": 1.8,
    "High":   3.5,
    None:     1.0,
}
 
# ---------------------------------------------------------------------------
# Dijkstra
# ---------------------------------------------------------------------------
 
def dijkstra(graph, start, end, congestion_map):
    """
    Shortest (fastest) path from start → end given live congestion_map.
    Returns (total_minutes, [path_nodes])
    """
    heap = [(0.0, start, [start])]
    visited = set()
 
    while heap:
        cost, node, path = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
 
        if node == end:
            return cost, path
 
        for neighbour, base_time in graph.get(node, []):
            if neighbour in visited:
                continue
            mult = CONGESTION_MULTIPLIER.get(congestion_map.get(neighbour), 1.0)
            heapq.heappush(heap, (cost + base_time * mult, neighbour, path + [neighbour]))
 
    return float("inf"), []
 
 
# ---------------------------------------------------------------------------
# Top-N alternative routes
# ---------------------------------------------------------------------------
 
def find_alternative_routes(start, end, congestion_map, top_n=3):
    """
    Returns up to top_n route dicts, sorted by estimated travel time.
 
    Each dict:
        path                : list of node IDs
        estimated_time      : float, minutes
        congestion_segments : list of congested node IDs on the path
        summary             : human-readable string
    """
    graph = build_graph()
 
    # Resolve "CBD" alias
    if start in ("CBD", "cbd"):
        start = CBD_NODE
    if end in ("CBD", "cbd"):
        end = CBD_NODE
 
    routes = []
    seen   = set()
 
    def _run(cong):
        t, path = dijkstra(graph, start, end, cong)
        if not path or t == float("inf"):
            return None
        key = tuple(path)
        if key in seen:
            return None
        seen.add(key)
        congested = [n for n in path if cong.get(n) in ("Medium", "High")]
        # Remove CBD-HUB from display path
        display = [n for n in path if n != CBD_NODE]
        summary = " → ".join(display) + f"  (~{round(t)} min)"
        return {
            "path":                path,
            "display_path":        display,
            "estimated_time":      round(t, 1),
            "congestion_segments": congested,
            "summary":             summary,
        }
 
    r1 = _run(congestion_map)
    if r1:
        routes.append(r1)
 
    for _ in range(top_n - 1):
        if not routes:
            break
        detour_cong = dict(congestion_map)
        for n in routes[-1]["path"][1:-1]:
            detour_cong[n] = "High"
        r = _run(detour_cong)
        if r:
            routes.append(r)
 
    # Re-sort by real travel time (not detour-inflated)
    def real_time(route):
        t, _ = dijkstra(graph, start, end, congestion_map)
        return t
 
    routes.sort(key=lambda r: r["estimated_time"])
    return routes[:top_n]