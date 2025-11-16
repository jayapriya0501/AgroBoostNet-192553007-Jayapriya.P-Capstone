import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, f1_score, mean_absolute_error
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional
import joblib
import warnings
warnings.filterwarnings('ignore')

class DataIntegration:
    """Data integration and fusion layer for combining datasets with different resolutions"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_and_integrate_data(self, crop_yield_path: str, sensor_data_path: str) -> pd.DataFrame:
        """Load and integrate crop yield and sensor data"""
        
        # Load datasets
        crop_df = pd.read_csv(crop_yield_path)
        sensor_df = pd.read_csv(sensor_data_path)
        
        print(f"Crop yield data shape: {crop_df.shape}")
        print(f"Sensor data shape: {sensor_df.shape}")
        
        # Data cleaning and preprocessing
        crop_df = self._clean_crop_data(crop_df)
        sensor_df = self._clean_sensor_data(sensor_df)
        
        # Feature engineering
        crop_df = self._engineer_crop_features(crop_df)
        sensor_df = self._engineer_sensor_features(sensor_df)
        
        # Data fusion
        integrated_df = self._fuse_datasets(crop_df, sensor_df)
        
        return integrated_df
    
    def _clean_crop_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess crop yield data"""
        
        # Remove duplicates and handle missing values
        df = df.drop_duplicates()
        
        # Convert categorical variables
        categorical_cols = ['Crop', 'Season', 'State']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Create yield per hectare
        if 'Area' in df.columns and 'Production' in df.columns:
            df['yield_per_hectare'] = df['Production'] / df['Area']
        
        return df
    
    def _clean_sensor_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess sensor data"""
        
        # Remove duplicates and handle missing values
        df = df.drop_duplicates()
        
        # Convert date columns
        date_cols = ['sowing_date', 'harvest_date', 'timestamp']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Encode categorical variables
        categorical_cols = ['region', 'crop_type', 'irrigation_type', 'fertilizer_type', 'crop_disease_status']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        return df
    
    def _engineer_crop_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from crop data"""
        
        # Fertilizer efficiency
        if 'Fertilizer' in df.columns and 'Area' in df.columns:
            df['fertilizer_per_hectare'] = df['Fertilizer'] / df['Area']
        
        # Pesticide efficiency
        if 'Pesticide' in df.columns and 'Area' in df.columns:
            df['pesticide_per_hectare'] = df['Pesticide'] / df['Area']
        
        # Rainfall efficiency
        if 'Annual_Rainfall' in df.columns and 'Area' in df.columns:
            df['rainfall_per_hectare'] = df['Annual_Rainfall'] / df['Area']
        
        return df
    
    def _engineer_sensor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from sensor data"""
        
        # Growing degree days
        if 'temperature_C' in df.columns and 'total_days' in df.columns:
            df['growing_degree_days'] = df['temperature_C'] * df['total_days']
        
        # Water stress index
        if 'soil_moisture_%' in df.columns and 'rainfall_mm' in df.columns:
            df['water_stress_index'] = df['rainfall_mm'] / (df['soil_moisture_%'] + 1)
        
        # Nutrient balance
        if 'soil_pH' in df.columns:
            df['ph_deviation'] = abs(df['soil_pH'] - 6.5)  # Optimal pH
        
        # Disease risk score
        if 'crop_disease_status' in df.columns:
            disease_mapping = {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
            df['disease_risk_score'] = df['crop_disease_status'].map(disease_mapping)
        
        return df
    
    def _fuse_datasets(self, crop_df: pd.DataFrame, sensor_df: pd.DataFrame) -> pd.DataFrame:
        """Fuse crop and sensor datasets"""
        
        # For demonstration, we'll create a synthetic fusion
        # In practice, you would use spatial and temporal matching
        
        # Sample from both datasets to create integrated dataset
        n_samples = min(len(crop_df), len(sensor_df))
        
        # Select common features
        integrated_data = []
        
        for i in range(n_samples):
            crop_sample = crop_df.iloc[i % len(crop_df)]
            sensor_sample = sensor_df.iloc[i % len(sensor_df)]
            
            # Create fused feature vector
            fused_features = {
                # From crop data
                'crop_type_encoded': crop_sample.get('Crop_encoded', 0),
                'season_encoded': crop_sample.get('Season_encoded', 0),
                'state_encoded': crop_sample.get('State_encoded', 0),
                'annual_rainfall': crop_sample.get('Annual_Rainfall', 0),
                'fertilizer_per_hectare': crop_sample.get('fertilizer_per_hectare', 0),
                'pesticide_per_hectare': crop_sample.get('pesticide_per_hectare', 0),
                
                # From sensor data
                'soil_moisture': sensor_sample.get('soil_moisture_%', 0),
                'soil_ph': sensor_sample.get('soil_pH', 0),
                'temperature': sensor_sample.get('temperature_C', 0),
                'rainfall': sensor_sample.get('rainfall_mm', 0),
                'humidity': sensor_sample.get('humidity_%', 0),
                'sunlight_hours': sensor_sample.get('sunlight_hours', 0),
                'ndvi_index': sensor_sample.get('NDVI_index', 0),
                'growing_degree_days': sensor_sample.get('growing_degree_days', 0),
                'water_stress_index': sensor_sample.get('water_stress_index', 0),
                'ph_deviation': sensor_sample.get('ph_deviation', 0),
                'disease_risk_score': sensor_sample.get('disease_risk_score', 0),
                
                # Target variables
                'yield_per_hectare': sensor_sample.get('yield_kg_per_hectare', crop_sample.get('yield_per_hectare', 0)),
                'crop_disease_status_encoded': sensor_sample.get('crop_disease_status_encoded', 0)
            }
            
            integrated_data.append(fused_features)
        
        return pd.DataFrame(integrated_data)


class AgroBoostNet:
    """
    AgroBoostNet: A hybrid machine learning algorithm combining ensemble learning 
    and neural networks for enhanced crop yield prediction and disease risk assessment.
    
    Architecture:
    - Stage 1: XGBoost for feature importance and initial predictions
    - Stage 2: Attention-based Neural Network for refined predictions
    - Custom loss function for multi-objective optimization
    - Climate Resilience Score prediction
    """
    
    def __init__(self, n_important_features: int = 15):
        self.n_important_features = n_important_features
        self.stage1_model = None
        self.stage2_model = None
        self.feature_importance = None
        self.important_features = None
        self.scaler_stage1 = StandardScaler()
        self.scaler_stage2 = StandardScaler()
        self.label_encoders = {}
        
    def build_custom_loss(self):
        """Build custom loss function for multi-objective optimization"""
        
        def custom_loss(y_true, y_pred):
            # Split predictions
            yield_true = y_true[:, 0]
            disease_true = y_true[:, 1]
            yield_pred = y_pred[:, 0]
            disease_pred = y_pred[:, 1]
            
            # Yield prediction loss (RMSE)
            yield_loss = tf.sqrt(tf.reduce_mean(tf.square(yield_true - yield_pred)))
            
            # Disease classification loss (Focal loss for imbalanced data)
            disease_loss = tf.keras.losses.sparse_categorical_crossentropy(
                disease_true, disease_pred
            )
            
            # Spatial consistency loss (encourages similar predictions for similar inputs)
            spatial_loss = self._spatial_consistency_loss(yield_pred)
            
            # Combined loss with learnable weights
            total_loss = yield_loss + 0.5 * disease_loss + 0.1 * spatial_loss
            
            return total_loss
        
        return custom_loss
    
    def _spatial_consistency_loss(self, predictions):
        """Implement spatial consistency loss"""
        # This is a simplified version - in practice, you'd use actual spatial coordinates
        return tf.reduce_mean(tf.square(predictions - tf.reduce_mean(predictions)))
    
    def build_attention_network(self, input_dim: int) -> Model:
        """Build attention-based neural network for Stage 2"""
        
        inputs = layers.Input(shape=(input_dim,))
        
        # Attention mechanism
        attention = layers.Dense(input_dim, activation='tanh')(inputs)
        attention = layers.Dense(input_dim, activation='softmax')(attention)
        
        # Apply attention
        attended_features = layers.Multiply()([inputs, attention])
        
        # Dense layers
        x = layers.Dense(128, activation='relu')(attended_features)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Multi-task outputs
        yield_output = layers.Dense(1, activation='linear', name='yield_output')(x)
        disease_output = layers.Dense(4, activation='softmax', name='disease_output')(x)
        climate_output = layers.Dense(1, activation='linear', name='climate_output')(x)
        
        model = Model(inputs=inputs, outputs=[yield_output, disease_output, climate_output])
        
        return model
    
    def train_stage1(self, X: pd.DataFrame, y_yield: np.ndarray, y_disease: np.ndarray):
        """Train Stage 1: XGBoost for feature importance"""
        
        print("Training Stage 1: XGBoost model...")
        
        # Prepare data
        X_scaled = self.scaler_stage1.fit_transform(X)
        
        # Multi-output XGBoost
        self.stage1_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        # Train on yield prediction (primary task)
        self.stage1_model.fit(X_scaled, y_yield)
        
        # Get feature importance
        self.feature_importance = self.stage1_model.feature_importances_
        
        # Select top features
        feature_names = X.columns
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        self.important_features = importance_df.head(self.n_important_features)['feature'].tolist()
        
        print(f"Selected {len(self.important_features)} important features")
        print("Top 10 features:")
        print(importance_df.head(10))
        
        return importance_df
    
    def train_stage2(self, X: pd.DataFrame, y_yield: np.ndarray, y_disease: np.ndarray):
        """Train Stage 2: Attention-based neural network"""
        
        print("Training Stage 2: Attention-based Neural Network...")
        
        # Use only important features
        X_important = X[self.important_features]
        
        # Fit scaler on important features only (for Stage 2)
        X_scaled = self.scaler_stage2.fit_transform(X_important)
        
        # Build model
        self.stage2_model = self.build_attention_network(len(self.important_features))
        
        # Compile model
        self.stage2_model.compile(
            optimizer='adam',
            loss={
                'yield_output': 'mse',
                'disease_output': 'sparse_categorical_crossentropy',
                'climate_output': 'mse'
            },
            loss_weights={'yield_output': 1.0, 'disease_output': 0.5, 'climate_output': 0.3},
            metrics={
                'yield_output': ['mae'],
                'disease_output': ['accuracy'],
                'climate_output': ['mae']
            }
        )
        
        # Prepare targets
        y_disease_encoded = LabelEncoder().fit_transform(y_disease)
        
        # Create climate resilience targets (synthetic for demonstration)
        y_climate = self._generate_climate_resilience_targets(y_yield)
        
        # Train model
        history = self.stage2_model.fit(
            X_scaled,
            [y_yield, y_disease_encoded, y_climate],
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        return history
    
    def _generate_climate_resilience_targets(self, y_yield: np.ndarray) -> np.ndarray:
        """Generate synthetic climate resilience targets"""
        # This would be based on actual climate scenario modeling
        # For demonstration, we create targets based on yield variation
        return y_yield * np.random.normal(1.0, 0.1, len(y_yield))
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions using the trained model"""
        
        # Stage 1 prediction
        X_scaled = self.scaler_stage1.transform(X)
        stage1_pred = self.stage1_model.predict(X_scaled)
        
        # Stage 2 prediction
        X_important = X[self.important_features]
        X_important_scaled = self.scaler_stage2.transform(X_important)
        
        yield_pred, disease_pred, climate_pred = self.stage2_model.predict(X_important_scaled)
        
        return {
            'stage1_yield': stage1_pred,
            'stage2_yield': yield_pred.flatten(),
            'disease_risk': np.argmax(disease_pred, axis=1),
            'disease_probabilities': disease_pred,
            'climate_resilience': climate_pred.flatten()
        }
    
    def evaluate(self, X: pd.DataFrame, y_yield: np.ndarray, y_disease: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        
        predictions = self.predict(X)
        
        # Yield prediction metrics
        rmse_yield = np.sqrt(mean_squared_error(y_yield, predictions['stage2_yield']))
        mae_yield = mean_absolute_error(y_yield, predictions['stage2_yield'])
        
        # Disease classification metrics
        f1_disease = f1_score(y_disease, predictions['disease_risk'], average='weighted')
        
        # Combined score
        combined_score = (1 / (1 + rmse_yield)) + f1_disease
        
        return {
            'yield_rmse': rmse_yield,
            'yield_mae': mae_yield,
            'disease_f1_score': f1_disease,
            'combined_score': combined_score
        }
    
    def explain_predictions(self, X: pd.DataFrame, sample_idx: int = 0):
        """Generate explanations using SHAP values"""
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.stage1_model)
        
        # Get SHAP values for a sample
        X_sample = X.iloc[sample_idx:sample_idx+1]
        X_scaled = self.scaler_stage1.transform(X_sample)
        
        shap_values = explainer.shap_values(X_scaled)
        
        # Create SHAP summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=X.columns)
        plt.title(f'SHAP Values for Sample {sample_idx}')
        plt.tight_layout()
        plt.savefig('shap_explanation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return shap_values
    
    def save_models(self, filepath_prefix: str):
        """Save trained models"""
        
        # Save Stage 1 model
        joblib.dump(self.stage1_model, f'{filepath_prefix}_stage1_xgboost.pkl')
        
        # Save Stage 2 model
        self.stage2_model.save(f'{filepath_prefix}_stage2_neuralnet.h5')
        
        # Save feature importance and scalers
        joblib.dump({
            'important_features': self.important_features,
            'scaler_stage1': self.scaler_stage1,
            'scaler_stage2': self.scaler_stage2,
            'label_encoders': self.label_encoders
        }, f'{filepath_prefix}_preprocessing.pkl')
        
        print(f"Models saved with prefix: {filepath_prefix}")
    
    def load_models(self, filepath_prefix: str):
        """Load trained models"""
        
        # Load Stage 1 model
        self.stage1_model = joblib.load(f'{filepath_prefix}_stage1_xgboost.pkl')
        
        # Load Stage 2 model
        self.stage2_model = keras.models.load_model(f'{filepath_prefix}_stage2_neuralnet.h5')
        
        # Load preprocessing objects
        preprocessing_data = joblib.load(f'{filepath_prefix}_preprocessing.pkl')
        self.important_features = preprocessing_data['important_features']
        self.scaler_stage1 = preprocessing_data['scaler_stage1']
        self.scaler_stage2 = preprocessing_data['scaler_stage2']
        self.label_encoders = preprocessing_data['label_encoders']
        
        print(f"Models loaded from prefix: {filepath_prefix}")


# Main execution function
def main():
    """Main execution function for AgroBoostNet"""
    
    print("=== AgroBoostNet: Hybrid ML for Crop Yield Prediction ===")
    
    # Initialize data integration
    data_integrator = DataIntegration()
    
    # Load and integrate data
    print("Loading and integrating datasets...")
    integrated_data = data_integrator.load_and_integrate_data(
        'agri_data/crop_yield.csv',
        'sensor_data/Smart_Farming_Crop_Yield_2024.csv'
    )
    
    print(f"Integrated data shape: {integrated_data.shape}")
    
    # Prepare features and targets
    feature_cols = [col for col in integrated_data.columns 
                   if col not in ['yield_per_hectare', 'crop_disease_status_encoded']]
    
    X = integrated_data[feature_cols]
    y_yield = integrated_data['yield_per_hectare'].values
    y_disease = integrated_data['crop_disease_status_encoded'].values
    
    print(f"Features shape: {X.shape}")
    print(f"Yield target shape: {y_yield.shape}")
    print(f"Disease target shape: {y_disease.shape}")
    
    # Split data
    X_train, X_test, y_yield_train, y_yield_test, y_disease_train, y_disease_test = train_test_split(
        X, y_yield, y_disease, test_size=0.2, random_state=42
    )
    
    # Initialize AgroBoostNet
    agroboost = AgroBoostNet(n_important_features=15)
    
    # Train Stage 1
    print("\n--- Training Stage 1 ---")
    importance_df = agroboost.train_stage1(X_train, y_yield_train, y_disease_train)
    
    # Train Stage 2
    print("\n--- Training Stage 2 ---")
    history = agroboost.train_stage2(X_train, y_yield_train, y_disease_train)
    
    # Evaluate model
    print("\n--- Model Evaluation ---")
    metrics = agroboost.evaluate(X_test, y_yield_test, y_disease_test)
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Generate explanations
    print("\n--- Generating Explanations ---")
    shap_values = agroboost.explain_predictions(X_test, sample_idx=0)
    
    # Save models
    print("\n--- Saving Models ---")
    agroboost.save_models('models/agroboostnet')
    
    print("\n=== Training Complete ===")
    
    return agroboost, metrics, history


if __name__ == "__main__":
    agroboost_model, evaluation_metrics, training_history = main()