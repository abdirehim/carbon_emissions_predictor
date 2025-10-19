"""
Test the trained carbon emissions model with real predictions
"""

import pandas as pd
import numpy as np
import joblib
import json
from src.prediction import EmissionsPredictionService

def load_model_and_test():
    """Load the trained model and test predictions"""
    print("🧪 TESTING CARBON EMISSIONS PREDICTIONS")
    print("=" * 50)
    
    # Load model artifacts
    try:
        model = joblib.load("models/trained_models/best_model_gradient_boosting.pkl")
        scaler = joblib.load("models/trained_models/scaler.pkl")
        
        with open("models/trained_models/feature_names.json", 'r') as f:
            feature_names = json.load(f)
            
        print("✅ Model loaded successfully")
        print(f"📊 Features: {len(feature_names)}")
        
    except FileNotFoundError as e:
        print(f"❌ Model files not found: {e}")
        print("Run train_model.py first!")
        return
    
    # Create prediction service
    predictor = EmissionsPredictionService()
    predictor.model = model
    predictor.scaler = scaler
    
    # Load some real data for testing
    df = pd.read_csv("data/raw/owid-co2-data.csv")
    df = df[df['year'] >= 2020]  # Recent data
    
    # Test with a few countries
    test_countries = ['United States', 'China', 'Germany', 'Brazil', 'India']
    
    print(f"\n🌍 TESTING PREDICTIONS FOR {len(test_countries)} COUNTRIES:")
    print("-" * 50)
    
    for country in test_countries:
        country_data = df[df['country'] == country]
        if not country_data.empty:
            # Get latest data for the country
            latest = country_data.iloc[-1]
            
            # Prepare features (use available features)
            features = {}
            for feature in feature_names:
                if feature in latest.index and pd.notna(latest[feature]):
                    features[feature] = latest[feature]
                else:
                    # Use median from dataset if feature missing
                    features[feature] = df[feature].median() if feature in df.columns else 0
            
            try:
                # Make prediction
                prediction = predictor.predict_single(features)
                actual = latest['co2'] if pd.notna(latest['co2']) else 'N/A'
                
                print(f"{country}:")
                print(f"  Predicted CO2: {prediction:.2f} million tons")
                print(f"  Actual CO2: {actual}")
                if actual != 'N/A':
                    error = abs(prediction - actual) / actual * 100
                    print(f"  Error: {error:.1f}%")
                print()
                
            except Exception as e:
                print(f"  ❌ Prediction failed: {e}")
    
    # Test scenario analysis
    print("🔮 SCENARIO ANALYSIS:")
    print("-" * 50)
    
    # Use median values as baseline
    baseline_features = {}
    for feature in feature_names:
        if feature in df.columns:
            baseline_features[feature] = df[feature].median()
        else:
            baseline_features[feature] = 0
    
    baseline_prediction = predictor.predict_single(baseline_features)
    print(f"📊 Baseline scenario: {baseline_prediction:.2f} million tons CO2")
    
    # Test different scenarios
    scenarios = {
        "🌱 Green Energy Transition": {
            "coal_co2": baseline_features["coal_co2"] * 0.5,
            "oil_co2": baseline_features["oil_co2"] * 0.8,
            "energy_efficiency": baseline_features["energy_efficiency"] * 1.2
        },
        "📈 Economic Growth": {
            "gdp_per_capita": baseline_features["gdp_per_capita"] * 1.3,
            "primary_energy_consumption": baseline_features["primary_energy_consumption"] * 1.15
        },
        "🏭 Industrial Expansion": {
            "cement_co2": baseline_features["cement_co2"] * 1.5,
            "fossil_fuel_co2": baseline_features["fossil_fuel_co2"] * 1.2
        }
    }
    
    for scenario_name, changes in scenarios.items():
        modified_features = baseline_features.copy()
        modified_features.update(changes)
        
        prediction = predictor.predict_single(modified_features)
        change = ((prediction - baseline_prediction) / baseline_prediction) * 100
        
        print(f"{scenario_name}:")
        print(f"  Predicted: {prediction:.2f} million tons")
        print(f"  Change: {change:+.1f}% from baseline")
        print()
    
    print("✅ Prediction testing completed!")

if __name__ == "__main__":
    load_model_and_test()