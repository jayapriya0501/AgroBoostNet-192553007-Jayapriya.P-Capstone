# AgroBoostNet: Hybrid Machine Learning Algorithm for Crop Yield Prediction and Disease Risk Assessment

## Project Overview

AgroBoostNet is a novel hybrid machine learning algorithm that combines ensemble learning and neural networks for enhanced crop yield prediction and disease risk assessment. The algorithm integrates data from Indian crop yield datasets and smart farming sensor data to provide comprehensive agricultural insights.

## Algorithm Architecture

### Two-Stage Predictive Model

1. **Stage 1: Gradient Boosted Decision Trees (XGBoost)**
   - Feature importance analysis and selection
   - Initial yield prediction
   - Disease risk classification

2. **Stage 2: Attention-based Neural Network**
   - Processes most important features from Stage 1
   - Multi-task learning for yield and disease prediction
   - Climate resilience scoring

### Key Components

1. **Data Integration Layer**
   - Handles different temporal and spatial resolutions
   - Implements data fusion techniques
   - Spatial blocking for cross-validation

2. **Custom Loss Function**
   - Multi-objective optimization
   - Yield prediction accuracy (RMSE)
   - Disease risk classification (F1-score)
   - Spatial consistency of predictions

3. **Climate Resilience Score**
   - Predicts yield changes under different climate scenarios
   - Incorporates weather pattern analysis
   - Risk assessment for climate variability

4. **Explainability Features**
   - SHAP values for feature importance
   - Attention visualization
   - Model interpretability tools

## Project Structure

```
soil_ml/
├── agri_data/                    # Indian crop yield dataset
│   └── crop_yield.csv
├── sensor_data/                  # Smart farming sensor data
│   └── Smart_Farming_Crop_Yield_2024.csv
├── src/
│   ├── data_integration.py       # Data fusion and preprocessing
│   ├── stage1_xgboost.py        # XGBoost model implementation
│   ├── stage2_neural_net.py     # Attention-based neural network
│   ├── custom_losses.py         # Custom loss functions
│   ├── climate_resilience.py    # Climate resilience scoring
│   ├── explainability.py        # SHAP and attention visualization
│   ├── evaluation.py            # Model evaluation metrics
│   └── utils.py                 # Utility functions
├── models/                       # Saved model files
├── notebooks/                    # Jupyter notebooks
├── dashboard/                    # Visualization dashboard
└── docs/                        # Technical documentation
```

## Key Features

### Data Integration
- Seamless combination of historical crop data and real-time sensor data
- Temporal and spatial resolution alignment
- Missing data imputation and outlier detection

### Novel Algorithm Components
- Hybrid ensemble-neural architecture
- Multi-task learning framework
- Spatial consistency constraints
- Climate scenario modeling

### Evaluation Metrics
- Primary: Combined RMSE (yield) + F1-score (disease)
- Secondary: Computational efficiency (predictions/second)
- Tertiary: Farmer usability score

## Implementation Requirements

- Python 3.8+
- TensorFlow/Keras for neural network components
- XGBoost for ensemble learning
- SHAP for explainability
- Spatial analysis libraries
- Cross-validation with spatial blocking

## Deliverables

1. Fully documented Jupyter notebook
2. Trained model files (.h5 and .pkl formats)
3. Visualization dashboard
4. Technical paper with algorithm explanation
5. Evaluation metrics and benchmarking results

Technical Paper: [Link](https://github.com/yourusername/AgroBoostNet/blob/main/docs/AgroBoostNet_Technical_Paper.pdf)

### To Run the Project

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/AgroBoostNet.git
   cd AgroBoostNet
   ```

2. Install required packages:
   ```
   uv add -r requirements.txt
   ```

3. Run the Jupyter notebook:
   ```
   jupyter notebook notebooks/AgroBoostNet_Implementation.ipynb
   ```
4. For Training the Model:
   - Run the training script:
     ```
     uv run python agroboostnet.py
     ```

4. Explore the dashboard:
   - Start the dashboard server:
     ```
     uv run streamlit run dashboard.py
     ```
   - Access the dashboard in your web browser at `http://localhost:8501`

## Future Work

- Integration with real-time sensor data
- Expansion to other crop types and regions
- Development of a mobile application for farmer use
- Collaboration with agricultural experts for validation