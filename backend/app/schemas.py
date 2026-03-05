from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List, Dict, Any, Union

class PatientClinicalData(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "patient_id": "P001",
                "horizon": "6h",
                "age in years": 55,
                "gender": "M",
                "weight in kg": 60,
                "height in cm": 165,
                "lactate_mean in mmol/L": 2.8,
                "glucose_mean in mg/dL": 110.0,
                "glucose_std in mg/dL": 15.0,
                "wbc_mean in K/uL": 12.0,
                "wbc_std in K/uL": 2.0,
                "rdw_mean in %": 14.0,
                "rdw_std in %": 1.0,
                "urea_nitrogen_mean in mg/dL": 20.0,
                "urea_nitrogen_std in mg/dL": 5.0,
                "bicarbonate_mean in mEq/L": 24.0,
                "bicarbonate_std in mEq/L": 2.0,
                "sbp_mean in mmHg": 120.0,
                "sbp_std in mmHg": 10.0
            }
        }
    )

    patient_id: str = Field(..., description="Unique identifier for the patient")
    horizon: Optional[str] = Field("6h", description="Prediction horizon: 6h, 12h, 18h, or 24h. If omitted, predicts for 6h.")
    
    # 17 Features in exact order expected by the model
    age: Optional[float] = Field(None, alias="age in years", description="Patient age")
    gender: Optional[str] = Field(None, description="Patient gender ('M' or 'F')")
    weight: Optional[float] = Field(None, alias="weight in kg", description="Weight in kg")
    height: Optional[float] = Field(None, alias="height in cm", description="Height in cm")
    lactate_mean: Optional[float] = Field(None, alias="lactate_mean in mmol/L", description="Lactate mean")
    glucose_mean: Optional[float] = Field(None, alias="glucose_mean in mg/dL", description="Glucose mean")
    glucose_std: Optional[float] = Field(None, alias="glucose_std in mg/dL", description="Glucose standard deviation")
    wbc_mean: Optional[float] = Field(None, alias="wbc_mean in K/uL", description="White blood cell count mean")
    wbc_std: Optional[float] = Field(None, alias="wbc_std in K/uL", description="White blood cell count std")
    rdw_mean: Optional[float] = Field(None, alias="rdw_mean in %", description="Red cell distribution width mean")
    rdw_std: Optional[float] = Field(None, alias="rdw_std in %", description="Red cell distribution width std")
    urea_nitrogen_mean: Optional[float] = Field(None, alias="urea_nitrogen_mean in mg/dL", description="Blood urea nitrogen mean")
    urea_nitrogen_std: Optional[float] = Field(None, alias="urea_nitrogen_std in mg/dL", description="Blood urea nitrogen std")
    bicarbonate_mean: Optional[float] = Field(None, alias="bicarbonate_mean in mEq/L", description="Bicarbonate mean")
    bicarbonate_std: Optional[float] = Field(None, alias="bicarbonate_std in mEq/L", description="Bicarbonate std")
    sbp_mean: Optional[float] = Field(None, alias="sbp_mean in mmHg", description="Systolic blood pressure mean")
    sbp_std: Optional[float] = Field(None, alias="sbp_std in mmHg", description="Systolic blood pressure std")

    @field_validator('gender', mode='before')
    def parse_gender(cls, v):
        if isinstance(v, str):
            v_upper = v.upper()
            if v_upper in ['M', 'MALE', '1', '1.0']:
                return 'M'
            elif v_upper in ['F', 'FEMALE', '0', '0.0']:
                return 'F'
        return v

    @field_validator('horizon')
    def validate_horizon(cls, v):
        if v not in ['6h', '12h', '18h', '24h', 'all']:
            raise ValueError("Horizon must be '6h', '12h', '18h', '24h', or 'all'")
        return v

class PatientWindowData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    patient_id: str = Field(..., description="Unique identifier for the patient")
    horizon: Optional[str] = Field("6h", description="Prediction horizon")
    
    age: Optional[float] = Field(None, alias="age in years")
    gender: Optional[str] = Field(None)
    weight: Optional[float] = Field(None, alias="weight in kg")
    height: Optional[float] = Field(None, alias="height in cm")
    
    # Arrays of measurements over a time window
    lactate: Optional[List[float]] = Field(None, alias="lactate array in mmol/L")
    glucose: Optional[List[float]] = Field(None, alias="glucose array in mg/dL")
    wbc: Optional[List[float]] = Field(None, alias="wbc array in K/uL")
    rdw: Optional[List[float]] = Field(None, alias="rdw array in %")
    urea_nitrogen: Optional[List[float]] = Field(None, alias="urea_nitrogen array in mg/dL")
    bicarbonate: Optional[List[float]] = Field(None, alias="bicarbonate array in mEq/L")
    sbp: Optional[List[float]] = Field(None, alias="sbp array in mmHg")

class BatchPatientData(BaseModel):
    patients: List[PatientClinicalData]

class PredictionResult(BaseModel):
    patient_id: str
    horizon: str
    prediction: int = Field(..., description="0 for Control, 1 for Sepsis Risk")
    risk_probability: float = Field(..., description="Probability of sepsis between 0 and 1")
    risk_level: str = Field(..., description="Risk level classification (e.g. LOW, MEDIUM, HIGH)")
    top_contributing_feature: Optional[str] = Field(None, description="The feature that contributed most to this prediction")
    top_contribution_impact: Optional[str] = Field(None, description="Impact of the top feature: increases_risk or decreases_risk")

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResult]

class PatientTrajectoryData(BaseModel):
    patient_id: str
    horizon: Optional[str] = "6h"
    history: List[PatientClinicalData] = Field(..., description="Chronological list of clinical data points")

class TrajectoryResponse(BaseModel):
    patient_id: str
    horizon: str
    trajectory: List[PredictionResult]

class ShapExplanation(BaseModel):
    feature: str
    feature_value: Optional[Union[float, str]]
    shap_value: float
    impact: str = Field(..., description="increases_risk or decreases_risk")

class ExplainResponse(BaseModel):
    patient_id: str
    horizon: str
    prediction: int
    risk_probability: float
    explanation: List[ShapExplanation]

class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_type: str
    target: str
    num_features: int
    prediction_type: str
    threshold: float
    features: List[str]
    horizons: List[str]
