import joblib
import pandas as pd
import sys
import argparse

def predict_global(region, location_type, rank):
    try:
        model = joblib.load("models/best_model.pkl")
        le_region = joblib.load("models/le_region.pkl")
        le_type = joblib.load("models/le_type.pkl")
        
        # Encode inputs
        region_enc = le_region.transform([region])[0]
        type_enc = le_type.transform([location_type])[0]
        
        X = pd.DataFrame([[region_enc, type_enc, rank]], columns=['Region_Enc', 'Type_Enc', 'Rank'])
        prediction = model.predict(X)[0]
        
        print(f"\n--- 2025 Air Quality Prediction ---")
        print(f"Region: {region}")
        print(f"Type: {location_type}")
        print(f"Rank: {rank}")
        print(f"Predicted PM2.5 Average: {prediction:.2f} \u00b5g/m\u00b3")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Available Regions:", list(le_region.classes_))
        print("Available Types:", list(le_type.classes_))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default="South Asia")
    parser.add_argument("--type", type=str, default="City")
    parser.add_argument("--rank", type=int, default=1)
    args = parser.parse_args()
    
    predict_global(args.region, args.type, args.rank)
