import pickle
import sys

def extract_features(filepath):
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        print("Model Features:")
        for window in data.keys():
            lgb_model = data[window]['LightGBM']
            # LightGBM models in scikit-learn API usually have feature_name_
            if hasattr(lgb_model, 'feature_name_'):
                features = lgb_model.feature_name_
            elif hasattr(lgb_model, 'booster_'):
                features = lgb_model.booster_.feature_name()
            else:
                features = "Could not determine features"
                
            print(f"Window {window} Features ({len(features)}): {features}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    extract_features("c:/Users/anjee/Desktop/HealthCare/trained_models.pkl")
