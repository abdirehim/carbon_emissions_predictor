"""
Visualization Module for Carbon Emissions Predictor
Creates charts and visualizations for data analysis and results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class EmissionsVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_emissions_trends(self, df, date_column, emissions_column):
        """Plot emissions trends over time"""
        plt.figure(figsize=(12, 6))
        plt.plot(df[date_column], df[emissions_column], linewidth=2)
        plt.title('Carbon Emissions Trends Over Time', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Carbon Emissions (tons CO2)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, df):
        """Plot correlation matrix heatmap"""
        plt.figure(figsize=(10, 8))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, importance_df):
        """Plot feature importance"""
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df.head(10), x='importance', y='feature')
        plt.title('Top 10 Feature Importance', fontsize=16)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    def plot_predictions_vs_actual(self, y_true, y_pred):
        """Plot predictions vs actual values"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Emissions', fontsize=12)
        plt.ylabel('Predicted Emissions', fontsize=12)
        plt.title('Predictions vs Actual Values', fontsize=16)
        
        # Add R² score
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(self, y_true, y_pred):
        """Plot residuals"""
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values', fontsize=12)
        plt.ylabel('Residuals', fontsize=12)
        plt.title('Residual Plot', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(self, results):
        """Compare model performance"""
        models = list(results.keys())
        r2_scores = [results[model]['r2'] for model in models]
        rmse_scores = [results[model]['rmse'] for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # R² scores
        ax1.bar(models, r2_scores, color='skyblue')
        ax1.set_title('Model Comparison - R² Score', fontsize=14)
        ax1.set_ylabel('R² Score', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # RMSE scores
        ax2.bar(models, rmse_scores, color='lightcoral')
        ax2.set_title('Model Comparison - RMSE', fontsize=14)
        ax2.set_ylabel('RMSE', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()