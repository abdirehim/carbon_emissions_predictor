"""
Complete Model Training Pipeline for Carbon Emissions Predictor
Uses existing src modules: data_processing, model_training, visualization
"""

import os
import pandas as pd
import numpy as np
import joblib
import json

from src.data_processing import DataProcessor
from src.model_training import EmissionsPredictor
from src.visualization import EmissionsVisualizer

def main():
    print("🚀 CARBON EMISSIONS MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # 1. Load and process data using existing DataProcessor
    print("\n1️⃣ LOADING AND PROCESSING DATA...")
    processor = DataProcessor()
    
    # Load raw data
    raw_file = "data/raw/owid-co2-data.csv"
    df = processor.load_data(raw_file)
    if df is None:
        print("❌ Failed to load data")
        return
    
    # Clean and prepare data
    df_clean = processor.clean_data(df)
    
    # Split data for training
    print("\n2️⃣ SPLITTING DATA...")
    X_train, X_test, y_train, y_test = processor.prepare_data(df_clean)
    
    # Scale features
    X_train_scaled, X_test_scaled = processor.scale_features(X_train, X_test)
    feature_names = processor.get_feature_names()
    
    # 3. Train models using existing EmissionsPredictor
    print("\n3️⃣ TRAINING MODELS...")
    trainer = EmissionsPredictor()
    results = trainer.train_models(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # 4. Evaluate and save best model
    print("\n4️⃣ MODEL EVALUATION...")
    print("\nModel Performance Summary:")
    print("-" * 50)
    
    best_model_name = None
    best_r2 = -float('inf')
    
    for name, metrics in results.items():
        print(f"{name.upper()}:")
        print(f"  R² Score: {metrics['r2']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print()
        
        if metrics['r2'] > best_r2:
            best_r2 = metrics['r2']
            best_model_name = name
    
    print(f"🏆 Best Model: {best_model_name.upper()} (R² = {best_r2:.4f})")
    
    # 5. Save models and metadata
    print("\n5️⃣ SAVING MODELS...")
    os.makedirs("models/trained_models", exist_ok=True)
    os.makedirs("models/model_evaluation", exist_ok=True)
    
    # Save best model
    best_model = results[best_model_name]['model']
    model_path = f"models/trained_models/best_model_{best_model_name}.pkl"
    joblib.dump(best_model, model_path)
    
    # Save scaler
    scaler_path = "models/trained_models/scaler.pkl"
    joblib.dump(processor.scaler, scaler_path)
    
    # Save feature names
    feature_names_path = "models/trained_models/feature_names.json"
    with open(feature_names_path, 'w') as f:
        json.dump(feature_names, f)
    
    # Save evaluation results
    evaluation_results = {
        'model_performance': {name: {k: float(v) if k != 'model' else str(type(v)) 
                                   for k, v in metrics.items() if k != 'model'} 
                            for name, metrics in results.items()},
        'best_model': best_model_name,
        'best_r2_score': float(best_r2),
        'feature_count': len(feature_names),
        'training_samples': len(X_train_scaled),
        'test_samples': len(X_test_scaled)
    }
    
    results_path = "models/model_evaluation/training_results.json"
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"✅ Best model saved: {model_path}")
    print(f"✅ Scaler saved: {scaler_path}")
    print(f"✅ Results saved: {results_path}")
    
    # 6. Generate visualizations using existing EmissionsVisualizer
    print("\n6️⃣ GENERATING VISUALIZATIONS...")
    visualizer = EmissionsVisualizer()
    
    # Predictions vs actual
    y_pred = best_model.predict(X_test_scaled)
    visualizer.plot_predictions_vs_actual(y_test, y_pred)
    
    # Feature importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        visualizer.plot_feature_importance(importance_df)
    
    # Model comparison
    visualizer.plot_model_comparison(results)
    
    print("\n🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📊 Best Model: {best_model_name}")
    print(f"🎯 R² Score: {best_r2:.4f}")
    print(f"💾 Model saved to: {model_path}")
    
    return best_model, feature_names, results

if __name__ == "__main__":
    model, features, results = main()