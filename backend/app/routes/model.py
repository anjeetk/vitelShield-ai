from fastapi import APIRouter
from app.schemas import ModelInfo
from app.services.preprocessing import FEATURE_COLS

router = APIRouter()

@router.get("/info", response_model=ModelInfo)
def get_model_info():
    return ModelInfo(
        model_type="LightGBM",
        target="Sepsis Onset",
        num_features=len(FEATURE_COLS),
        prediction_type="classification",
        threshold=0.5,
        features=FEATURE_COLS,
        horizons=["6h", "12h", "18h", "24h"]
    )

@router.get("/features")
def get_features():
    features_info = []
    for f in FEATURE_COLS:
        dt = "float"
        if f == "gender":
            dt = "string ('M'/'F')"
        
        features_info.append({
            "name": f,
            "datatype": dt,
            "required": False,  # Using IterativeImputer
            "range": None
        })
    return features_info
