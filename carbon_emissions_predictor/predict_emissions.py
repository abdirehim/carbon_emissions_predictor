#!/usr/bin/env python3
"""
Carbon Emissions Predictor - Command Line Interface
Easy-to-use script for making CO2 emission predictions
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import json
from src.prediction import EmissionsPredictionService

def load_model():
    """Load the trained model and components"""
    try:
        model = joblib.load("models/trained_models/best_model_gradient_boosting.pkl")
        scaler = joblib.load("models/trained_models/scaler.pkl")
        
        with open("models/trained_models/feature_names.json", 'r') as f:
            feature_names = json.load(f)
            
        return model, scaler, feature_names
    except FileNotFoundError as e:
        print(f"❌ Error: Model files not found. Run 'python train_model.py' first!")
        return None, None, None

def predict_country_emissions(country_name, year=2023):
    """Predict emissions for a specific country"""
    model, scaler, feature_names = load_model()
    if model is None:
        return
    
    # Load data
    df = pd.read_csv("data/raw/owid-co2-data.csv")
    
    # Find country data
    country_data = df[df['country'].str.lower() == country_name.lower()]
    if country_data.empty:
        print(f"❌ Country '{country_name}' not found in dataset")
        available_countries = df['country'].unique()[:10]
        print(f"Available countries (sample): {', '.join(available_countries)}")
        return
    
    # Get latest available data
    latest_data = country_data.iloc[-1]
    
    # Prepare features
    features = {}
    for feature in feature_names:
        if feature in latest_data.index and pd.notna(latest_data[feature]):
            features[feature] = latest_data[feature]
        else:
            # Use global median for missing features
            features[feature] = df[feature].median() if feature in df.columns else 0
    
    # Update year
    features['year'] = year
    
    # Make prediction
    predictor = EmissionsPredictionService()
    predictor.model = model
    predictor.scaler = scaler
    
    prediction = predictor.predict_single(features)
    
    print(f"\n🌍 CARBON EMISSIONS PREDICTION")
    print(f"Country: {country_name.title()}")
    print(f"Year: {year}")
    print(f"Predicted CO2 Emissions: {prediction:.2f} million tons")
    
    # Show actual if available
    if pd.notna(latest_data.get('co2')):
        actual = latest_data['co2']
        print(f"Latest Actual ({latest_data['year']}): {actual:.2f} million tons")
        if latest_data['year'] == year:
            error = abs(prediction - actual) / actual * 100
            print(f"Prediction Error: {error:.1f}%")

def run_scenario_analysis():
    """Run predefined scenario analysis"""
    model, scaler, feature_names = load_model()
    if model is None:
        return
    
    # Load data for baseline
    df = pd.read_csv("data/raw/owid-co2-data.csv")
    
    # Create baseline
    baseline_features = {}
    for feature in feature_names:
        if feature in df.columns:
            baseline_features[feature] = df[feature].median()
        else:
            baseline_features[feature] = 0
    
    predictor = EmissionsPredictionService()
    predictor.model = model
    predictor.scaler = scaler
    
    baseline_prediction = predictor.predict_single(baseline_features)
    
    print(f"\n🔮 CLIMATE POLICY SCENARIO ANALYSIS")
    print(f"Baseline Prediction: {baseline_prediction:.2f} million tons CO2")
    print("=" * 50)
    
    # Scenarios
    scenarios = {
        "Green Energy Transition": {
            "coal_co2": baseline_features["coal_co2"] * 0.3,
            "oil_co2": baseline_features["oil_co2"] * 0.8,
            "energy_efficiency": baseline_features["energy_efficiency"] * 1.2
        },
        "Aggressive Climate Action": {
            "coal_co2": baseline_features["coal_co2"] * 0.1,
            "oil_co2": baseline_features["oil_co2"] * 0.6,
            "gas_co2": baseline_features["gas_co2"] * 0.7,
            "energy_efficiency": baseline_features["energy_efficiency"] * 1.5
        },
        "Economic Growth": {
            "gdp_per_capita": baseline_features["gdp_per_capita"] * 1.5,
            "primary_energy_consumption": baseline_features["primary_energy_consumption"] * 1.3
        }
    }
    
    for scenario_name, changes in scenarios.items():
        modified_features = baseline_features.copy()
        modified_features.update(changes)
        
        # Recalculate fossil fuel total
        if any(fuel in changes for fuel in ['coal_co2', 'gas_co2', 'oil_co2']):
            modified_features['fossil_fuel_co2'] = (
                modified_features['coal_co2'] + 
                modified_features['gas_co2'] + 
                modified_features['oil_co2']
            )
        
        prediction = predictor.predict_single(modified_features)
        change = ((prediction - baseline_prediction) / baseline_prediction) * 100
        
        print(f"{scenario_name}:")
        print(f"  Predicted: {prediction:.2f} million tons")
        print(f"  Change: {change:+.1f}% from baseline")
        print()

def list_available_countries():
    """List available countries in the dataset"""
    try:
        df = pd.read_csv("data/raw/owid-co2-data.csv")
        countries = sorted(df['country'].unique())
        
        print(f"\n📋 AVAILABLE COUNTRIES ({len(countries)} total):")
        print("=" * 50)
        
        # Print in columns
        for i, country in enumerate(countries):
            if i % 3 == 0:
                print()
            print(f"{country:<25}", end="")
        print("\n")
        
    except FileNotFoundError:
        print("❌ Dataset not found. Make sure owid-co2-data.csv is in data/raw/")

def main():
    parser = argparse.ArgumentParser(
        description="Carbon Emissions Predictor - Climate Action Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_emissions.py --country "United States"
  python predict_emissions.py --country China --year 2025
  python predict_emissions.py --scenarios
  python predict_emissions.py --list-countries
        """
    )
    
    parser.add_argument('--country', '-c', type=str, 
                       help='Country name to predict emissions for')
    parser.add_argument('--year', '-y', type=int, default=2023,
                       help='Year for prediction (default: 2023)')
    parser.add_argument('--scenarios', '-s', action='store_true',
                       help='Run climate policy scenario analysis')
    parser.add_argument('--list-countries', '-l', action='store_true',
                       help='List all available countries')
    
    args = parser.parse_args()
    
    print("🌍 CARBON EMISSIONS PREDICTOR")
    print("Supporting UN SDG 13: Climate Action")
    print("=" * 40)
    
    if args.list_countries:
        list_available_countries()
    elif args.scenarios:
        run_scenario_analysis()
    elif args.country:
        predict_country_emissions(args.country, args.year)
    else:
        parser.print_help()
        print("\n💡 Quick start:")
        print("  python predict_emissions.py --country 'United States'")
        print("  python predict_emissions.py --scenarios")

if __name__ == "__main__":
    main()