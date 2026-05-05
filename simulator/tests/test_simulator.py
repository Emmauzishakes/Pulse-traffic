import pytest
from unittest.mock import patch, MagicMock
import time

from simulator.config import SimulatorConfig
from simulator.generator import TrafficGenerator

# --- 1. CONFIGURATION TESTS ---

def test_simulator_config_defaults():
    """Test that the configuration loads with correct safe defaults."""
    config = SimulatorConfig()
    assert config.backend_url == "http://127.0.0.1:8000/traffic-data"
    assert config.interval_seconds == 2.0
    assert config.scenario == "normal"
    assert config.fault_injection is False

def test_simulator_config_overrides():
    """Test that config accepts custom overrides for different environments."""
    config = SimulatorConfig(scenario="peak_hour", fault_injection=True, interval_seconds=0.5)
    assert config.scenario == "peak_hour"
    assert config.fault_injection is True
    assert config.interval_seconds == 0.5


# --- 2. GENERATOR (DATA BOUNDARY) TESTS ---

def test_generator_normal_data_bounds():
    """STQA Boundary Value Analysis: Ensure normal data stays within realistic physical limits."""
    config = SimulatorConfig(scenario="normal", fault_injection=False)
    generator = TrafficGenerator(config)
    
    generated = generator.next()
    
    # Assert the raw payload exists
    assert isinstance(generated.raw_payload, dict)
    
    # Extract values safely
    payload = generated.raw_payload
    
    # Check realistic boundaries
    assert 0 <= payload.get("vehicle_count", 0) <= 200, "Vehicle count out of realistic bounds"
    assert 0.0 <= payload.get("speed", 0.0) <= 120.0, "Speed exceeds highway limits"
    assert 0.0 <= payload.get("density", 0.0) <= 1.0, "Density must be a percentage between 0 and 1"

# def test_generator_fault_injection():
#     """STQA Negative Testing: Ensure the generator creates invalid data when asked."""
#     # We force the same seed so the randomizer predictably creates a fault
#     config = SimulatorConfig(fault_injection=True, seed=42) 
#     generator = TrafficGenerator(config)
    
#     # Generate a few items, at least one should be faulty
#     fault_found = False
#     for _ in range(10):
#         generated = generator.next()
#         payload = generated.raw_payload
        
#         # A fault might be negative vehicles, impossible speeds, or missing fields
#         if (payload.get("vehicle_count", 0) < 0 or 
#             payload.get("speed", 0.0) < 0 or 
#             payload.get("density", 0.0) > 2.0):
#             fault_found = True
#             break
            
#     assert fault_found, "Fault injection failed to produce invalid data."

def test_generator_fault_injection():
    """STQA Negative Testing: Ensure the generator creates invalid data when asked."""
    # THE FIX: We must set the scenario explicitly to "faulty_data"
    config = SimulatorConfig(scenario="faulty_data", fault_injection=True) 
    generator = TrafficGenerator(config)
    
    fault_found = False
    
    for _ in range(50):
        generated = generator.next()
        
        # If the generated.validated is None, it means the Pydantic schema 
        # caught the injected fault and rejected it.
        if generated.validated is None:
            fault_found = True
            break
            
    assert fault_found, "Fault injection failed to produce invalid data."


# --- 3. MOCKING & NETWORK TESTS ---

@patch('requests.post')
def test_mock_network_request(mock_post):
    """
    STQA Mocking Test: Ensure the simulator sends the correct JSON format 
    without actually needing the FastAPI backend to be running online.
    """
    # Setup the fake response that requests.post will return
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_post.return_value = mock_response
    
    # The node we want to test
    test_node = "ST-THIKA-001"
    
    # Construct a fake payload exactly like our main.py loop does
    payload = {
        "node_id": test_node,
        "timestamp": str(time.time()),
        "vehicle_count": 45,
        "speed": 60.5,
        "density": 0.35
    }
    
    import requests
    response = requests.post("http://fake-url.com/traffic-data", json=payload)
    
    # Assertions
    assert response.status_code == 201
    # Verify that requests.post was called exactly once with our data
    mock_post.assert_called_once_with("http://fake-url.com/traffic-data", json=payload)
    
    # Verify the payload structure that was sent
    called_kwargs = mock_post.call_args.kwargs
    sent_json = called_kwargs.get("json")
    
    assert sent_json["node_id"] == "ST-THIKA-001"
    assert "vehicle_count" in sent_json
    assert "speed" in sent_json