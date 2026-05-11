import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
import logging
from typing import List, Optional

from schemas import TrafficPayload, TrafficResponse, PredictionResponse
from core import models
from predictor.predictor import predict, predict_with_route
from core.database import engine, SessionLocal

models.Base.metadata.create_all(bind=engine)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Pulse Traffic API",
    description="Backend for simulated traffic data ingestion",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/traffic-data", status_code=status.HTTP_201_CREATED)
async def ingest_traffic_data(payload: TrafficPayload, db: Session = Depends(get_db)):
    try:
        new_reading = models.TrafficReading(
            node_id=payload.node_id or "simulator",
            timestamp=payload.timestamp,
            vehicle_count=payload.vehicle_count,
            speed=payload.speed,
            density=payload.density
        )
        db.add(new_reading)
        db.commit()
        db.refresh(new_reading)

        logger.info(f"Received traffic data: {payload.node_id}: {payload.vehicle_count} vehicles, {payload.speed} km/h")

        return {
            "status": "success",
            "message": "Traffic data saved to database",
            "data_received": payload.model_dump()
        }
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"System failure during ingestion: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error"
        )


@app.get("/traffic-data", response_model=List[TrafficResponse])
async def get_traffic_data(
    skip: int = 0,
    limit: int = 100,
    node_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(models.TrafficReading)

    if node_id:
        query = query.filter(models.TrafficReading.node_id == node_id)

    records = query.order_by(models.TrafficReading.timestamp.desc()).offset(skip).limit(limit).all()

    return records


@app.post("/predict/{node_id}")
async def create_prediction(node_id: str, db: Session = Depends(get_db)):

    readings = db.query(models.TrafficReading)\
        .filter(models.TrafficReading.node_id == node_id)\
        .order_by(models.TrafficReading.timestamp.asc())\
        .all()

    if not readings:
        raise HTTPException(status_code=404, detail="No data")

    result = predict(readings, node_id)

    return result


@app.post("/predict/{node_id}/route")
async def predict_with_alternatives(
    node_id: str,
    destination: str,
    db: Session = Depends(get_db)
):
    readings = (
        db.query(models.TrafficReading)
        .filter(models.TrafficReading.node_id == node_id)
        .order_by(models.TrafficReading.timestamp.asc())
        .all()
    )
    if not readings:
        raise HTTPException(status_code=404, detail=f"No data for node {node_id}")

    latest_ids = (
        db.query(func.max(models.TrafficReading.id))
        .group_by(models.TrafficReading.node_id)
        .all()
    )
    latest_readings = (
        db.query(models.TrafficReading)
        .filter(models.TrafficReading.id.in_([r[0] for r in latest_ids]))
        .all()
    )

    def _quick_level(speed):
        if speed < 20:  return "High"
        if speed < 40:  return "Medium"
        return "Low"

    congestion_map = {
        r.node_id: _quick_level(r.speed)
        for r in latest_readings
    }

    result = predict_with_route(
        readings=readings,
        node_id=node_id,
        destination=destination,
        congestion_map=congestion_map,
    )

    return result