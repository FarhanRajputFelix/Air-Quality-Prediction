import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

def train_global_model():
    print("Initializing Training for Global 2025 Air Quality Model...")
    df = pd.read_csv("data/global_air_quality_2025.csv")
    
    # Feature Engineering: Encode Region and Type
    le_region = LabelEncoder()
    df['Region_Enc'] = le_region.fit_transform(df['Region'])
    
    le_type = LabelEncoder()
    df['Type_Enc'] = le_type.fit_transform(df['Type'])
    
    # Target: PM2.5_Average
    X = df[['Region_Enc', 'Type_Enc', 'Rank']]
    y = df['PM2.5_Average']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    best_r2 = -1
    best_model = None
    best_name = ""
    
    print("\n--- Model Comparison Results ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results[name] = {"MAE": mae, "R2": r2}
        print(f"{name}: MAE = {mae:.4f}, R2 = {r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name

    print(f"\n[SUCCESS] Best Model: {best_name} (R2 = {best_r2:.4f})")
    
    # Save Artifacts
    if not os.path.exists("models"):
        os.makedirs("models")
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(le_region, "models/le_region.pkl")
    joblib.dump(le_type, "models/le_type.pkl")
    print("Model and Encoders saved to models/")

if __name__ == "__main__":
    train_global_model()
