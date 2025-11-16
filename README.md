# AgroBoostNet: Hybrid Machine Learning Algorithm for Crop Yield Prediction and Disease Risk Assessment

## Project Overview

AgroBoostNet is a novel hybrid machine learning algorithm that combines ensemble learning and neural networks for enhanced crop yield prediction and disease risk assessment. The algorithm integrates data from Indian crop yield datasets and smart farming sensor data to provide comprehensive agricultural insights.

## Abstract

This paper presents AgroBoostNet, a novel hybrid machine learning algorithm that combines the strengths of ensemble learning and neural networks for enhanced crop yield prediction and disease risk assessment. The algorithm integrates data from Indian crop yield datasets and smart farming sensor data through a sophisticated data fusion layer that handles different temporal and spatial resolutions. AgroBoostNet employs a two-stage architecture: Stage 1 utilizes Gradient Boosted Decision Trees (XGBoost) for feature importance analysis and initial yield prediction, while Stage 2 employs an attention-based neural network that processes the most important features identified in Stage 1. The algorithm introduces a custom loss function that simultaneously optimizes for yield prediction accuracy (RMSE), disease risk classification (F1-score), and spatial consistency of predictions. Additionally, AgroBoostNet provides a novel "Climate Resilience Score" that predicts how yield might change under different climate scenarios. Experimental results demonstrate superior performance compared to baseline models, with significant improvements in both yield prediction accuracy and disease risk classification.

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

### Performance Comparison

AgroBoostNet demonstrates superior performance across all primary metrics:

| Model | Yield RMSE | Disease F1 | Combined Score |
|-------|------------|------------|----------------|
| Random Forest | 512.3 | 0.742 | 1.742 |
| XGBoost | 489.7 | 0.768 | 1.768 |
| Neural Network | 501.2 | 0.751 | 1.751 |
| **AgroBoostNet** | **445.8** | **0.812** | **1.812** |

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

Technical Paper: [Link](https://github.com/jayapriya0501/AgroBoostNet/blob/main/technical_paper.md)

### To Run the Project

1. Clone the repository:
   ```
   git clone https://github.com/jayapriya0501/AgroBoostNet.git
   cd AgroBoostNet
   ```

2. Install required packages:
   ```
   uv add -r requirements.txt
   ```

3. For Training the Model:
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