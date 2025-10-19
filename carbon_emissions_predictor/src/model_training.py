"""
Model Training Module for Carbon Emissions Predictor
Contains ML model implementation and training logic
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import logging

class EmissionsPredictor:
    def __init__(self):
        self.models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.best_model = None
        self.best_score = float('-inf')
        
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train multiple models and select the best one"""
        results = {}
        
        for name, model in self.models.items():
            logging.info(f"Training {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            # Update best model
            if r2 > self.best_score:
                self.best_score = r2
                self.best_model = model
                
            logging.info(f"{name} - R²: {r2:.4f}, RMSE: {rmse:.4f}")
        
        return results
    
    def save_model(self, model, filepath):
        """Save trained model to file"""
        try:
            joblib.dump(model, filepath)
            logging.info(f"Model saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
    
    def load_model(self, filepath):
        """Load trained model from file"""
        try:
            model = joblib.load(filepath)
            logging.info(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return None
    
    def get_feature_importance(self, model, feature_names):
        """Get feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            logging.warning("Model doesn't support feature importance")
            return None