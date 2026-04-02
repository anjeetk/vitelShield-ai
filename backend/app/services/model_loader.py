import pickle
import os
import logging
from app.config import settings

logger = logging.getLogger(__name__)

class ModelLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance.models = None
            cls._instance.preprocessing = None
            cls._instance.is_loaded = False
        return cls._instance

    def load_models(self):
        if self.is_loaded:
            return

        model_path = os.path.abspath(settings.MODEL_PATH)
        preprocessing_path = os.path.abspath(settings.PREPROCESSING_PATH)

        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        if not os.path.exists(preprocessing_path):
            logger.error(f"Preprocessing file not found at {preprocessing_path}")
            raise FileNotFoundError(f"Preprocessing file not found at {preprocessing_path}")

        logger.info(f"Loading trained models from {model_path}...")
        try:
            with open(model_path, 'rb') as f:
                self.models = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

        logger.info(f"Loading preprocessing objects from {preprocessing_path}...")
        try:
            with open(preprocessing_path, 'rb') as f:
                self.preprocessing = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load preprocessing objects: {e}")
            raise

        self.is_loaded = True
        logger.info("Models and preprocessing objects loaded successfully.")

    def get_model(self, horizon: str, model_name: str = 'LightGBM'):
        if not self.is_loaded:
            raise RuntimeError("Models are not loaded yet.")
        if horizon not in self.models:
            raise ValueError(f"Horizon {horizon} not found in loaded models.")
        return self.models[horizon].get(model_name)

    def get_preprocessing(self, horizon: str):
        if not self.is_loaded:
            raise RuntimeError("Preprocessing objects are not loaded yet.")
            
        imputers = self.preprocessing.get('imputers', {})
        scalers = self.preprocessing.get('scalers', {})
        
        imputer = imputers.get(horizon)
        scaler = scalers.get(horizon)
        
        if imputer is None or scaler is None:
            raise ValueError(f"Preprocessing objects for horizon {horizon} are missing.")
            
        return imputer, scaler

model_loader = ModelLoader()
