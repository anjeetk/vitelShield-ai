import shap
import pandas as pd
from typing import List, Dict, Any, Tuple
from app.schemas import PredictionResult, PatientClinicalData, ShapExplanation, ExplainResponse, PatientWindowData
from app.services.preprocessing import preprocess_patients, FEATURE_COLS
from app.services.model_loader import model_loader
import numpy as np

def get_risk_level(prob: float) -> str:
    if prob < 0.3:
        return "LOW"
    elif prob < 0.7:
        return "MEDIUM"
    else:
        return "HIGH"

def aggregate_window_data(window_data: PatientWindowData) -> PatientClinicalData:
    def safe_mean(arr):
        return float(np.mean(arr)) if arr and len(arr) > 0 else None
        
    def safe_std(arr):
        return float(np.std(arr, ddof=1)) if arr and len(arr) > 1 else (0.0 if arr and len(arr) == 1 else None)

    return PatientClinicalData(
        patient_id=window_data.patient_id,
        horizon=window_data.horizon,
        age=window_data.age,
        gender=window_data.gender,
        weight=window_data.weight,
        height=window_data.height,
        lactate_mean=safe_mean(window_data.lactate),
        glucose_mean=safe_mean(window_data.glucose),
        glucose_std=safe_std(window_data.glucose),
        wbc_mean=safe_mean(window_data.wbc),
        wbc_std=safe_std(window_data.wbc),
        rdw_mean=safe_mean(window_data.rdw),
        rdw_std=safe_std(window_data.rdw),
        urea_nitrogen_mean=safe_mean(window_data.urea_nitrogen),
        urea_nitrogen_std=safe_std(window_data.urea_nitrogen),
        bicarbonate_mean=safe_mean(window_data.bicarbonate),
        bicarbonate_std=safe_std(window_data.bicarbonate),
        sbp_mean=safe_mean(window_data.sbp),
        sbp_std=safe_std(window_data.sbp)
    )

def predict(patients: List[PatientClinicalData], horizon: str) -> List[PredictionResult]:
    df_preprocessed = preprocess_patients(patients, horizon)
    model = model_loader.get_model(horizon, "LightGBM")
    
    # Predict probabilities (index 1 is the positive class probability)
    probs = model.predict_proba(df_preprocessed)[:, 1]
    
    # Generate SHAP values for top feature extraction
    explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
    shap_values = explainer.shap_values(df_preprocessed)
    
    # Handle multiclass vs binary shape differences in LightGBM SHAP
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]
    
    results = []
    for idx, p in enumerate(patients):
        prob = float(probs[idx])
        # Using 0.5 threshold as per notebook
        prediction_class = 1 if prob >= 0.5 else 0
        
        # Find top feature
        patient_shap = shap_values[idx]
        max_idx = abs(patient_shap).argmax()
        top_feature = FEATURE_COLS[max_idx]
        top_impact = "increases_risk" if patient_shap[max_idx] > 0 else "decreases_risk"
        
        result = PredictionResult(
            patient_id=p.patient_id,
            horizon=horizon,
            prediction=prediction_class,
            risk_probability=round(prob, 4),
            risk_level=get_risk_level(prob),
            top_contributing_feature=top_feature,
            top_contribution_impact=top_impact
        )
        results.append(result)
        
    return results

def predict_and_explain(patient: PatientClinicalData, horizon: str) -> ExplainResponse:
    df_preprocessed = preprocess_patients([patient], horizon)
    model = model_loader.get_model(horizon, "LightGBM")
    
    prob = float(model.predict_proba(df_preprocessed)[0, 1])
    prediction_class = 1 if prob >= 0.5 else 0
    
    # Generate SHAP explanations
    explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
    shap_values = explainer.shap_values(df_preprocessed)
    
    # Depending on LightGBM version/objective, shap_values might be a list (multiclass) or array
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values_to_use = shap_values[1][0]
    else:
        shap_values_to_use = shap_values[0]
        
    explanations = []
    patient_dict = patient.model_dump()
    
    for idx, feature_name in enumerate(FEATURE_COLS):
        shap_val = float(shap_values_to_use[idx])
        impact = "increases_risk" if shap_val > 0 else "decreases_risk"
        
        # Don't include features with exact zero impact to keep response clean
        if abs(shap_val) > 1e-5:
            explanations.append(ShapExplanation(
                feature=feature_name,
                feature_value=patient_dict.get(feature_name),
                shap_value=round(shap_val, 4),
                impact=impact
            ))
            
    # Sort by absolute SHAP value (highest impact first)
    explanations = sorted(explanations, key=lambda x: abs(x.shap_value), reverse=True)
    
    return ExplainResponse(
        patient_id=patient.patient_id,
        horizon=horizon,
        prediction=prediction_class,
        risk_probability=round(prob, 4),
        explanation=explanations
    )
