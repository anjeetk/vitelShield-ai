from fastapi import APIRouter, HTTPException
from typing import List
from app.schemas import PatientClinicalData, BatchPatientData, PredictionResult, BatchPredictionResponse, ExplainResponse, PatientWindowData, PatientTrajectoryData, TrajectoryResponse
from app.services.prediction import predict, predict_and_explain, aggregate_window_data
from app.services.model_loader import model_loader

router = APIRouter()

@router.post("/", response_model=PredictionResult)
def single_prediction(patient: PatientClinicalData):
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model is currently unavailable")
        
    horizon = patient.horizon or "6h"
    if horizon == "all":
        horizon = "6h"
        
    results = predict([patient], horizon)
    return results[0]

@router.post("/batch", response_model=BatchPredictionResponse)
def batch_prediction(data: BatchPatientData):
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model is currently unavailable")
        
    all_results = []
    
    patients_by_horizon = {}
    for p in data.patients:
        hz = p.horizon or "6h"
        if hz == "all":
            for h in ["6h", "12h", "18h", "24h"]:
                if h not in patients_by_horizon:
                    patients_by_horizon[h] = []
                p_copy = p.model_copy()
                p_copy.horizon = h
                patients_by_horizon[h].append(p_copy)
        else:
            if hz not in patients_by_horizon:
                patients_by_horizon[hz] = []
            patients_by_horizon[hz].append(p)
            
    for hz, patients in patients_by_horizon.items():
        batch_results = predict(patients, hz)
        all_results.extend(batch_results)
        
    return BatchPredictionResponse(predictions=all_results)

@router.post("/explain", response_model=ExplainResponse)
def explain_prediction(patient: PatientClinicalData):
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model is currently unavailable")
        
    horizon = patient.horizon or "6h"
    if horizon == "all":
        horizon = "6h"
        
    return predict_and_explain(patient, horizon)

@router.post("/window", response_model=PredictionResult)
def window_prediction(window_data: PatientWindowData):
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model is currently unavailable")
        
    aggregated_patient = aggregate_window_data(window_data)
    
    horizon = aggregated_patient.horizon or "6h"
    if horizon == "all":
        horizon = "6h"
        
    results = predict([aggregated_patient], horizon)
    return results[0]

@router.post("/trajectory", response_model=TrajectoryResponse)
def trajectory_prediction(trajectory_data: PatientTrajectoryData):
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model is currently unavailable")
        
    horizon = trajectory_data.horizon or "6h"
    if horizon == "all":
        horizon = "6h"
        
    # We pass the chronological history array directly to the batch predict function
    results = predict(trajectory_data.history, horizon)
    
    return TrajectoryResponse(
        patient_id=trajectory_data.patient_id,
        horizon=horizon,
        trajectory=results
    )
