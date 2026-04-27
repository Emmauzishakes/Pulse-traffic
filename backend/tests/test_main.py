from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

from main import app, get_db
import models

# ---------------------------------------------------------
# QA SETUP: IN-MEMORY TEST DATABASE
# ---------------------------------------------------------
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

models.Base.metadata.create_all(bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

# ---------------------------------------------------------
# 1. DATA INGESTION & VALIDATION TESTS
# ---------------------------------------------------------

def test_ingest_valid_traffic_data():
    """Test that the API accepts and saves a perfectly formatted payload."""
    valid_payload = {
        "node_id": "ST-JUJA-001",
        "timestamp": "2026-04-26T10:55:00Z",
        "vehicle_count": 42,
        "speed": 35.5,
        "density": 0.68  # Corrected to a ratio (le=1)
    }
    response = client.post("/traffic-data", json=valid_payload)
    assert response.status_code == 201
    assert response.json()["status"] == "success"

def test_ingest_negative_speed_validation():
    """QA Check: API must reject negative speeds."""
    invalid_payload = {
        "node_id": "ST-JUJA-002",
        "timestamp": "2026-04-26T11:00:00Z",
        "vehicle_count": 10,
        "speed": -5.0,  
        "density": 0.20  # Corrected to a ratio
    }
    response = client.post("/traffic-data", json=invalid_payload)
    assert response.status_code == 422
    assert "greater than or equal to 0" in response.text.lower()

# ---------------------------------------------------------
# 2. DATA RETRIEVAL (GET) INTEGRATION TESTS
# ---------------------------------------------------------

def test_retrieve_data_by_node_id():
    """Test that the GET endpoint successfully filters by node_id."""
    # First, inject a specific record
    client.post("/traffic-data", json={
        "node_id": "ST-RUIRU-999",
        "timestamp": "2026-04-26T12:00:00Z",
        "vehicle_count": 100,
        "speed": 50.0,
        "density": 0.40  # Corrected to a ratio
    })
    
    # Second, retrieve it
    response = client.get("/traffic-data?node_id=ST-RUIRU-999")
    assert response.status_code == 200
    
    data = response.json()
    assert len(data) >= 1
    assert data[0]["node_id"] == "ST-RUIRU-999"
    assert data[0]["speed"] == 50.0

# ---------------------------------------------------------
# 3. PREDICTION MODULE INTEGRATION TESTS
# ---------------------------------------------------------

def test_prediction_endpoint_not_enough_data():
    """QA Check: Prediction should fail gracefully if no data exists for a node."""
    response = client.post("/predict/GHOST-NODE")
    assert response.status_code == 404
    assert "not enough data" in response.text.lower()

def test_prediction_endpoint_high_congestion_logic():
    """Test that the threshold logic correctly outputs 'High' congestion."""
    # Inject 3 readings with very slow speeds (< 20 km/h)
    for _ in range(3):
        client.post("/traffic-data", json={
            "node_id": "ST-THIKA-SLOW",
            "timestamp": "2026-04-26T13:00:00Z",
            "vehicle_count": 150,
            "speed": 10.0, 
            "density": 0.95  # Corrected to a ratio
        })

    # Trigger the prediction
    response = client.post("/predict/ST-THIKA-SLOW")
    assert response.status_code == 200
    
    data = response.json()
    assert data["node_id"] == "ST-THIKA-SLOW"
    assert data["predicted_level"] == "High"
    assert data["confidence_score"] == 85.0