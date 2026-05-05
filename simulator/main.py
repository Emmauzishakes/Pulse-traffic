from __future__ import annotations

import argparse
import signal
import time
import requests

from simulator.config import SimulatorConfig
from simulator.generator import TrafficGenerator
from simulator.logger import get_logger

# The 15 Nairobi Corridors mapped in our frontend UI
NAIROBI_NODES = [
    "ST-THIKA-001", "ST-THIKA-002", "ST-THIKA-003", "ST-THIKA-004", "ST-THIKA-005",
    "ST-EXP-001", "ST-EXP-002", "ST-EXP-003", "ST-EXP-004", "ST-EXP-005",
    "ST-NGONG-001", "ST-NGONG-002", "ST-NGONG-003",
    "ST-LANG-001", "ST-LANG-002"
]

def run_simulation(config: SimulatorConfig) -> None:
    logger = get_logger()
    generator = TrafficGenerator(config)
    running = True

    def stop_handler(signum, frame):
        nonlocal running
        running = False
        logger.info("Stopping simulator after signal %s", signum)

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    
    logger.info(f"Starting Nairobi Live Network Simulation across {len(NAIROBI_NODES)} nodes...")

    while running:
        for node in NAIROBI_NODES:
            if not running:
                break
            
            # 1. Generate the raw numbers
            generated = generator.next()
            
            # 2. Extract the numbers safely
            if generated.validated is not None:
                v_count = generated.validated.vehicle_count
                spd = generated.validated.speed
                den = generated.validated.density
                ts = generated.validated.timestamp
            else:
                v_count = generated.raw_payload.get("vehicle_count", 0)
                spd = generated.raw_payload.get("speed", 0.0)
                den = generated.raw_payload.get("density", 0.0)
                ts = generated.raw_payload.get("timestamp", time.time())
            
            # 3. CONSTRUCT THE PERFECT PAYLOAD DIRECTLY
            payload = {
                "node_id": node,
                "timestamp": str(ts),
                "vehicle_count": v_count,
                "speed": spd,
                "density": den
            }

            # 4. BYPASS TrafficSender AND POST DIRECTLY TO FASTAPI
            try:
                response = requests.post(config.backend_url, json=payload)
                if response.status_code in (200, 201):
                    logger.info(f"Direct POST success: {node} -> {response.status_code}")
                else:
                    logger.error(f"Backend rejected {node}: {response.text}")
            except Exception as e:
                logger.error(f"Network error sending {node}: {e}")

        # Wait before the next network pulse
        time.sleep(max(0.0, config.interval_seconds))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pulse Traffic simulator")
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000/traffic-data")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--scenario", default="normal", choices=["normal", "peak_hour", "low_traffic", "sudden_spike", "faulty_data"])
    parser.add_argument("--fault-injection", action="store_true")
    parser.add_argument("--allow-invalid-payloads", action="store_true")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--node-id", default="simulator")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = SimulatorConfig(
        backend_url=args.backend_url,
        interval_seconds=args.interval,
        scenario=args.scenario,
        fault_injection=args.fault_injection,
        allow_invalid_payloads=args.allow_invalid_payloads,
        timeout_seconds=args.timeout,
        retries=args.retries,
        node_id=args.node_id,
        seed=args.seed,
    )
    run_simulation(config)

if __name__ == "__main__":
    main()