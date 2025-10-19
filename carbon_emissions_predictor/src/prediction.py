"""
Prediction Module for Carbon Emissions Predictor
Manages model inference and predictions
"""

import pandas as pd
import numpy as np
import joblib
import logging

class EmissionsPredictionService:
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = None
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained model"""
        try:
            self.model = joblib.load(model_path)
            logging.info(f"Model loaded from {model_path}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
    
    def predict_single(self, features):
        """Make prediction for a single sample"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Convert to DataFrame if it's a dict
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        # Ensure we have numpy array for prediction (removes feature name warnings)
        features_array = features.values if hasattr(features, 'values') else features
        
        prediction = self.model.predict(features_array)
        return prediction[0]
    
    def predict_batch(self, features_df):
        """Make predictions for multiple samples"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Convert to numpy array to avoid feature name warnings
        features_array = features_df.values if hasattr(features_df, 'values') else features_df
        predictions = self.model.predict(features_array)
        return predictions
    
    def predict_with_confidence(self, features, n_estimators=None):
        """Make predictions with confidence intervals (for ensemble models)"""
        if hasattr(self.model, 'estimators_'):
            # For ensemble models like Random Forest
            predictions = []
            for estimator in self.model.estimators_:
                pred = estimator.predict(features)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            
            # 95% confidence interval
            confidence_interval = 1.96 * std_pred
            
            return {
                'prediction': mean_pred,
                'confidence_lower': mean_pred - confidence_interval,
                'confidence_upper': mean_pred + confidence_interval,
                'std': std_pred
            }
        else:
            # For non-ensemble models, return simple prediction
            prediction = self.model.predict(features)
            return {'prediction': prediction}
    
    def generate_scenario_predictions(self, base_features, scenarios):
        """Generate predictions for different scenarios"""
        results = {}
        
        for scenario_name, scenario_changes in scenarios.items():
            # Create modified features
            modified_features = base_features.copy()
            for feature, value in scenario_changes.items():
                modified_features[feature] = value
            
            # Make prediction
            prediction = self.predict_single(modified_features)
            results[scenario_name] = prediction
        
        return results
    
    def calculate_emission_reduction(self, baseline_prediction, scenario_prediction):
        """Calculate emission reduction percentage"""
        reduction = baseline_prediction - scenario_prediction
        reduction_percentage = (reduction / baseline_prediction) * 100
        return {
            'absolute_reduction': reduction,
            'percentage_reduction': reduction_percentage
        }