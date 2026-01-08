import pickle
import sys

def inspect_model(filepath):
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        print("Type of data:", type(data))
        if isinstance(data, dict):
            print("Keys:", data.keys())
            for k in data.keys():
                print(f"Key {k} type: {type(data[k])}")
                if isinstance(data[k], dict):
                    print(f"  Subkeys for {k}: {data[k].keys()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_model("c:/Users/anjee/Desktop/HealthCare/trained_models.pkl")
