import pandas as pd
import numpy as np
from typing import List, Dict, Any
from app.schemas import PatientClinicalData
from app.services.model_loader import model_loader

# The exact 17 features expected by the LightGBM model, in order.
FEATURE_COLS = [
    'age', 'gender', 'weight', 'height', 'lactate_mean', 'glucose_mean', 'glucose_std', 
    'wbc_mean', 'wbc_std', 'rdw_mean', 'rdw_std', 'urea_nitrogen_mean', 'urea_nitrogen_std', 
    'bicarbonate_mean', 'bicarbonate_std', 'sbp_mean', 'sbp_std'
]

def preprocess_patients(patients: List[PatientClinicalData], horizon: str) -> pd.DataFrame:
    """
    Converts a list of PatientClinicalData into a preprocessed Pandas DataFrame 
    ready for LightGBM inference. Handles missing values by using the loaded
    IterativeImputer and StandardScaler.
    """
    
    # Extract data into a list of dictionaries
    data_list = []
    for p in patients:
        row = p.model_dump(include=set(FEATURE_COLS))
        data_list.append(row)
        
    df = pd.DataFrame(data_list)
    
    # 1. Preprocess gender explicitly as in the notebook
    # df['gender'] = df['gender'].map({'M': 1.0, 'F': 0.0}).fillna(0.5)
    # Since Pydantic already parsed to 'M'/'F' or None:
    df['gender'] = df['gender'].map({'M': 1.0, 'F': 0.0})
    df['gender'] = df['gender'].fillna(0.5)
    
    # 2. Reorder columns to ensure exact match with FEATURE_COLS
    # and fill any completely missing columns with np.nan for the imputer to handle
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan
            
    df = df[FEATURE_COLS]
    
    # Fetch preprocessing objects for this horizon
    imputer, scaler = model_loader.get_preprocessing(horizon)
    
    # 3. Impute missing continuous variables using the MICE imputer
    X_imp = imputer.transform(df)
    
    # 4. Scale features using the StandardScaler
    X_scaled = scaler.transform(X_imp)
    
    # Return as DataFrame to preserve feature names for SHAP/LightGBM
    return pd.DataFrame(X_scaled, columns=FEATURE_COLS)
