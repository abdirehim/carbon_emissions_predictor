# Carbon Emissions Predictor 🌍🤖

## Project Overview
This machine learning project addresses **UN Sustainable Development Goal 13: Climate Action** by predicting carbon emissions based on economic and energy indicators. The model helps policymakers understand the impact of various factors on carbon emissions and make data-driven decisions for climate action.

## Problem Statement
Climate change poses an existential threat to our planet. Accurate prediction of carbon emissions is crucial for:
- Setting realistic emission reduction targets
- Evaluating policy effectiveness
- Planning sustainable development strategies
- Allocating resources for climate initiatives

## Solution
We developed a **Multiple Linear Regression model** that predicts carbon emissions using six key features:
- GDP per capita
- Energy consumption
- Renewable energy percentage
- Population
- Industrial output
- Vehicle count

## Technical Implementation
- **Algorithm**: Multiple Linear Regression
- **Libraries**: Scikit-learn, Pandas, NumPy, Matplotlib
- **Metrics**: MAE, MSE, RMSE, R² Score
- **Data**: Synthetic dataset simulating real-world climate and economic data

## Key Features
1. **Data Generation**: Synthetic data representing 33 years of climate and economic indicators
2. **Model Training**: Supervised learning with feature scaling
3. **Visualization**: Comprehensive plots showing trends and relationships
4. **Prediction**: Real-time emission predictions for policy scenarios

## Installation & Usage

### Quick Start
```bash
# Install dependencies
pip install pandas numpy matplotlib scikit-learn seaborn joblib

# Train the model
python train_model.py

# Make predictions
python predict_emissions.py --country "United States"
python predict_emissions.py --scenarios

# Run demo
python demo.py
```

### Command Line Interface
```bash
# Predict emissions for a specific country
python predict_emissions.py --country "China" --year 2025

# Run climate policy scenario analysis
python predict_emissions.py --scenarios

# List available countries
python predict_emissions.py --list-countries
```

### Jupyter Notebooks
- `notebooks/01_data_exploration.ipynb` - Data analysis
- `notebooks/02_model_development.ipynb` - Model training
- `notebooks/03_complete_analysis.ipynb` - Full system demonstration
##
 Model Performance
- **Best Model**: Gradient Boosting Regressor
- **R² Score**: 0.6379 (63.79% variance explained)
- **RMSE**: 1680.93 million tons CO2
- **Features**: 15 engineered features from economic and energy data

## Key Features Used
1. **Economic**: GDP, GDP per capita, population
2. **Energy**: Energy consumption, energy efficiency
3. **Emissions**: CO2 per capita, fossil fuel sources
4. **Greenhouse Gases**: Methane, nitrous oxide
5. **Derived**: Energy efficiency, fossil fuel totals

## Real-World Applications
- **Policy Evaluation**: Test climate policy effectiveness
- **Target Setting**: Set realistic emission reduction goals
- **Scenario Planning**: Compare development pathways
- **Climate Negotiations**: Support international discussions
- **Investment Decisions**: Guide clean energy investments

## Project Structure
```
carbon_emissions_predictor/
├── src/                    # Core modules
│   ├── data_processing.py  # Data preprocessing
│   ├── model_training.py   # ML model training
│   ├── prediction.py       # Prediction service
│   └── visualization.py    # Plotting functions
├── models/                 # Trained models
│   ├── trained_models/     # Model artifacts
│   └── model_evaluation/   # Performance metrics
├── data/                   # Datasets
│   ├── raw/               # Original OWID data
│   └── processed/         # Clean data
├── notebooks/             # Jupyter analysis
├── train_model.py         # Main training pipeline
├── predict_emissions.py   # CLI interface
└── demo.py               # Interactive demo
```

## Contributing to Climate Action
This project directly supports **UN Sustainable Development Goal 13: Climate Action** by providing data-driven insights for emission reduction strategies. The model helps policymakers understand the relationship between economic development and carbon emissions, enabling more informed decisions for a sustainable future.

## License
This project is open source and available for climate research and policy applications.