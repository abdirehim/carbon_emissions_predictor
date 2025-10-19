"""
Data Processing Module for Carbon Emissions Predictor
Handles data loading, cleaning, and preprocessing for OWID CO2 dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import logging

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = []
        self.target_column = 'co2'
        
    def load_data(self, file_path):
        """Load OWID CO2 data from CSV file"""
        try:
            data = pd.read_csv(file_path)
            print(f"✅ Data loaded successfully: {data.shape}")
            
            # Filter for recent years and countries only
            data = data[data['year'] >= 2000]
            exclude_entities = ['World', 'Europe', 'Asia', 'Africa', 'North America', 
                               'South America', 'Oceania', 'European Union (27)']
            data = data[~data['country'].isin(exclude_entities)]
            
            print(f"✅ After filtering: {data.shape}")
            return data
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def clean_data(self, df):
        """Clean and preprocess the CO2 dataset"""
        print("🧹 Cleaning data...")
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Select relevant features for CO2 prediction
        self.feature_columns = [
            'year', 'population', 'gdp', 'energy_per_capita',
            'primary_energy_consumption', 'co2_per_capita',
            'coal_co2', 'gas_co2', 'oil_co2', 'cement_co2',
            'methane', 'nitrous_oxide'
        ]
        
        # Keep only relevant columns plus target and identifiers
        keep_columns = ['country', 'iso_code', self.target_column] + self.feature_columns
        available_columns = [col for col in keep_columns if col in df.columns]
        df = df[available_columns].copy()
        
        # Handle missing values with median imputation
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = self.imputer.fit_transform(df[numeric_columns])
        
        # Create derived features
        df = self._create_derived_features(df)
        
        # Remove rows where target is missing or zero
        df = df[(df[self.target_column].notna()) & (df[self.target_column] > 0)]
        
        print(f"✅ Data cleaned: {df.shape}")
        return df
    
    def _create_derived_features(self, df):
        """Create additional features from existing data"""
        # GDP per capita
        if 'gdp' in df.columns and 'population' in df.columns:
            df['gdp_per_capita'] = (df['gdp'] / df['population']).fillna(0)
            self.feature_columns.append('gdp_per_capita')
        
        # Fossil fuel total
        fossil_cols = ['coal_co2', 'gas_co2', 'oil_co2']
        available_fossil = [col for col in fossil_cols if col in df.columns]
        if available_fossil:
            df['fossil_fuel_co2'] = df[available_fossil].sum(axis=1)
            self.feature_columns.append('fossil_fuel_co2')
        
        # Energy efficiency
        if 'gdp' in df.columns and 'primary_energy_consumption' in df.columns:
            df['energy_efficiency'] = (df['gdp'] / df['primary_energy_consumption']).fillna(0)
            df['energy_efficiency'] = df['energy_efficiency'].replace([np.inf, -np.inf], 0)
            self.feature_columns.append('energy_efficiency')
        
        return df
    
    def encode_categorical_features(self, df, categorical_columns):
        """Encode categorical variables"""
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col])
        return df
    
    def scale_features(self, X_train, X_test):
        """Scale numerical features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    def prepare_data(self, df, test_size=0.2):
        """Prepare data for model training"""
        # Select features that exist in the dataframe
        available_features = [col for col in self.feature_columns if col in df.columns]
        
        X = df[available_features].copy()
        y = df[self.target_column].copy()
        
        # Handle any remaining infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"✅ Data split - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test
    
    def get_feature_names(self):
        """Get list of feature column names"""
        return self.feature_columns