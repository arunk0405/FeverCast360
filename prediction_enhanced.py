#!/usr/bin/env python3
"""
FeverCast360 — Enhanced ML Pipeline for Disease Outbreak Prediction

Architecture:
-------------
Stage 1: Severity Index Modeling (Training Phase)
    - Learns relationship between environmental factors and disease burden
    - Creates a severity index based on historical disease cases
    
Stage 2: Outbreak Risk Prediction (Prediction Phase)
    - Uses learned severity patterns + current conditions
    - Predicts future outbreak risk

Training Data Required:
-----------------------
State, District, Year, Month, Dengue_Cases, Malaria_Cases, 
Chikungunya_Cases, Temperature, Humidity, Rainfall, 
Sanitation_Score, Population_Density

Prediction Data Required:
-------------------------
State, District, Year, Month, Temperature, Humidity, Rainfall,
Sanitation_Score, Population_Density

Output:
-------
District, Month, Predicted_Severity_Index, Outbreak_Risk, 
Dominant_Disease_Type, Confidence_Score
"""
from __future__ import annotations
import os
import warnings
from typing import Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)


class FeverOutbreakPredictor:
    """Enhanced fever outbreak prediction using historical disease data"""
    
    def __init__(self, models_dir: str = "models_enhanced"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        self.severity_model: Optional[RandomForestRegressor] = None
        self.outbreak_classifier: Optional[GradientBoostingClassifier] = None
        self.scaler = StandardScaler()
        self.state_encoder = LabelEncoder()
        self.district_encoder = LabelEncoder()
        
        # Store feature columns
        self.feature_cols: list[str] = []
        
    def calculate_severity_index(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate severity index based on disease cases
        Severity = weighted sum of disease cases normalized by population
        """
        weights = {
            'Dengue_Cases': 0.4,      # Higher weight due to severity
            'Malaria_Cases': 0.35,    # Significant impact
            'Chikungunya_Cases': 0.25  # Lower but still important
        }
        
        # Normalize by population density to get per-capita impact
        total_cases = (
            df['Dengue_Cases'] * weights['Dengue_Cases'] +
            df['Malaria_Cases'] * weights['Malaria_Cases'] +
            df['Chikungunya_Cases'] * weights['Chikungunya_Cases']
        )
        
        # Normalize to 0-1 scale (per 100k population)
        severity_raw = (total_cases / df['Population_Density']) * 100000
        
        # Handle edge cases
        severity_raw = severity_raw.fillna(0)
        severity_raw = severity_raw.replace([np.inf, -np.inf], 0)
        
        # Scale to 0-1 range
        if severity_raw.max() > severity_raw.min():
            severity_index = (severity_raw - severity_raw.min()) / (severity_raw.max() - severity_raw.min())
        else:
            severity_index = pd.Series([0.5] * len(df), index=df.index)
        
        return severity_index
    
    def calculate_outbreak_threshold(self, df: pd.DataFrame) -> pd.Series:
        """
        Determine outbreak status based on severity index
        Outbreak = 1 if severity above 75th percentile, else 0
        """
        threshold = df['Severity_Index'].quantile(0.75)
        return (df['Severity_Index'] > threshold).astype(int)
    
    def identify_dominant_disease(self, df: pd.DataFrame) -> pd.Series:
        """
        Identify which disease is most prevalent
        """
        disease_cols = ['Dengue_Cases', 'Malaria_Cases', 'Chikungunya_Cases']
        dominant = df[disease_cols].idxmax(axis=1)
        
        # Map to disease names
        disease_map = {
            'Dengue_Cases': 'Dengue',
            'Malaria_Cases': 'Malaria',
            'Chikungunya_Cases': 'Chikungunya'
        }
        
        return dominant.map(disease_map)
    
    def prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, list[str]]:
        """
        Prepare features for modeling
        """
        df = df.copy()
        
        # Encode categorical variables
        if is_training:
            df['State_Encoded'] = self.state_encoder.fit_transform(df['State'].astype(str))
            df['District_Encoded'] = self.district_encoder.fit_transform(df['District'].astype(str))
        else:
            # Handle unseen labels
            df['State_Encoded'] = df['State'].astype(str).apply(
                lambda x: self.state_encoder.transform([x])[0] if x in self.state_encoder.classes_ else -1
            )
            df['District_Encoded'] = df['District'].astype(str).apply(
                lambda x: self.district_encoder.transform([x])[0] if x in self.district_encoder.classes_ else -1
            )
        
        # Create temporal features (cyclical encoding for month)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # Interaction features
        df['Temp_Humidity'] = df['Temperature'] * df['Humidity'] / 100
        df['Rain_Sanitation'] = df['Rainfall'] * (1 - df['Sanitation_Score'])
        df['Temp_Rainfall'] = df['Temperature'] * df['Rainfall']
        
        # Feature columns for modeling
        feature_cols = [
            'State_Encoded', 'District_Encoded', 'Year', 
            'Month_Sin', 'Month_Cos',
            'Temperature', 'Humidity', 'Rainfall',
            'Sanitation_Score', 'Population_Density',
            'Temp_Humidity', 'Rain_Sanitation', 'Temp_Rainfall'
        ]
        
        return df, feature_cols
    
    def train_severity_model(self, train_df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
        """
        Stage 1: Train model to predict severity index
        """
        print("\n" + "="*70)
        print("STAGE 1: TRAINING SEVERITY INDEX MODEL")
        print("="*70)
        
        # Calculate severity index from disease cases
        train_df['Severity_Index'] = self.calculate_severity_index(train_df)
        
        # Prepare features
        train_df, feature_cols = self.prepare_features(train_df, is_training=True)
        self.feature_cols = feature_cols
        
        X = train_df[feature_cols]
        y = train_df['Severity_Index']
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train Random Forest Regressor for severity prediction
        print("\n📊 Training Random Forest Regressor...")
        self.severity_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.severity_model.fit(X_train_scaled, y_train)
        
        # Validation metrics
        y_pred = self.severity_model.predict(X_val_scaled)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        
        print(f"\n✅ Severity Model Performance:")
        print(f"   • MSE:        {mse:.4f}")
        print(f"   • RMSE:       {rmse:.4f}")
        print(f"   • R² Score:   {r2:.4f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': self.severity_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\n📈 Top 5 Important Features:")
        for idx, row in importance.head().iterrows():
            print(f"   • {row['Feature']:<20} {row['Importance']:.4f}")
        
        return train_df, feature_cols
    
    def train_outbreak_classifier(self, train_df: pd.DataFrame, feature_cols: list[str]) -> None:
        """
        Stage 2: Train classifier for outbreak prediction
        """
        print("\n" + "="*70)
        print("STAGE 2: TRAINING OUTBREAK CLASSIFIER")
        print("="*70)
        
        # Calculate outbreak labels
        train_df['Outbreak_Label'] = self.calculate_outbreak_threshold(train_df)
        train_df['Dominant_Disease'] = self.identify_dominant_disease(train_df)
        
        X = train_df[feature_cols]
        y_outbreak = train_df['Outbreak_Label']
        
        # Check if we have both classes
        if len(y_outbreak.unique()) < 2:
            print("\n⚠️  Warning: Only one class present in outbreak labels.")
            print("    Using all data as training set.")
            X_train = X
            y_train = y_outbreak
            X_val = X
            y_val = y_outbreak
        else:
            # Split for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_outbreak, test_size=0.2, random_state=42, stratify=y_outbreak
            )
        
        # Scale features
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train Gradient Boosting Classifier
        print("\n🎯 Training Gradient Boosting Classifier...")
        self.outbreak_classifier = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.outbreak_classifier.fit(X_train_scaled, y_train)
        
        # Validation metrics
        y_pred = self.outbreak_classifier.predict(X_val_scaled)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"\n✅ Outbreak Classifier Performance:")
        print(f"   • Accuracy: {accuracy:.4f}")
        print("\n📊 Classification Report:")
        report = classification_report(y_val, y_pred, 
                                      target_names=['No Outbreak', 'Outbreak'],
                                      zero_division=0)
        print(report)
        
    def save_models(self) -> None:
        """Save all trained models and encoders"""
        print("\n" + "="*70)
        print("SAVING MODELS")
        print("="*70)
        
        joblib.dump(self.severity_model, 
                   os.path.join(self.models_dir, 'severity_model.pkl'))
        joblib.dump(self.outbreak_classifier, 
                   os.path.join(self.models_dir, 'outbreak_classifier.pkl'))
        joblib.dump(self.scaler, 
                   os.path.join(self.models_dir, 'scaler.pkl'))
        joblib.dump(self.state_encoder, 
                   os.path.join(self.models_dir, 'state_encoder.pkl'))
        joblib.dump(self.district_encoder, 
                   os.path.join(self.models_dir, 'district_encoder.pkl'))
        
        # Save feature columns
        with open(os.path.join(self.models_dir, 'feature_cols.txt'), 'w') as f:
            f.write('\n'.join(self.feature_cols))
        
        print(f"\n✅ Models saved to: {self.models_dir}/")
        print(f"   • severity_model.pkl")
        print(f"   • outbreak_classifier.pkl")
        print(f"   • scaler.pkl")
        print(f"   • state_encoder.pkl")
        print(f"   • district_encoder.pkl")
        print(f"   • feature_cols.txt")
        
    def load_models(self) -> None:
        """Load trained models"""
        print("\n📂 Loading trained models...")
        
        self.severity_model = joblib.load(
            os.path.join(self.models_dir, 'severity_model.pkl'))
        self.outbreak_classifier = joblib.load(
            os.path.join(self.models_dir, 'outbreak_classifier.pkl'))
        self.scaler = joblib.load(
            os.path.join(self.models_dir, 'scaler.pkl'))
        self.state_encoder = joblib.load(
            os.path.join(self.models_dir, 'state_encoder.pkl'))
        self.district_encoder = joblib.load(
            os.path.join(self.models_dir, 'district_encoder.pkl'))
        
        # Load feature columns
        with open(os.path.join(self.models_dir, 'feature_cols.txt'), 'r') as f:
            self.feature_cols = [line.strip() for line in f.readlines()]
        
        print(f"✅ Models loaded from: {self.models_dir}/")
        
    def predict(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on new data (without disease cases)
        """
        print("\n" + "="*70)
        print("MAKING PREDICTIONS")
        print("="*70)
        
        # Prepare features
        pred_df, _ = self.prepare_features(pred_df, is_training=False)
        
        X = pred_df[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        
        # Predict severity index
        print("\n📊 Predicting severity index...")
        severity_pred = self.severity_model.predict(X_scaled)
        
        # Predict outbreak probability
        print("🎯 Predicting outbreak risk...")
        outbreak_prob = self.outbreak_classifier.predict_proba(X_scaled)[:, 1]
        outbreak_pred = self.outbreak_classifier.predict(X_scaled)
        
        # Create results dataframe
        results = pd.DataFrame({
            'State': pred_df['State'],
            'District': pred_df['District'],
            'Year': pred_df['Year'].astype(int),
            'Month': pred_df['Month'].astype(int),
            'Predicted_Severity_Index': severity_pred,
            'Outbreak_Probability': outbreak_prob,
            'Outbreak_Prediction': outbreak_pred,
            'Outbreak_Risk': ['High' if p > 0.75 else 'Moderate' if p > 0.5 else 'Low' 
                             for p in outbreak_prob],
            'Temperature': pred_df['Temperature'],
            'Humidity': pred_df['Humidity'],
            'Rainfall': pred_df['Rainfall'],
            'Sanitation_Score': pred_df['Sanitation_Score'],
            'Population_Density': pred_df['Population_Density']
        })
        
        print(f"\n✅ Predictions completed for {len(results)} records")
        print(f"   • High Risk:     {sum(results['Outbreak_Risk'] == 'High')}")
        print(f"   • Moderate Risk: {sum(results['Outbreak_Risk'] == 'Moderate')}")
        print(f"   • Low Risk:      {sum(results['Outbreak_Risk'] == 'Low')}")
        
        return results
    
    def train_pipeline(self, training_data_path: str) -> 'FeverOutbreakPredictor':
        """
        Complete training pipeline
        """
        print("\n" + "="*70)
        print("FEVERCAST360 ENHANCED ML PIPELINE - TRAINING MODE")
        print("="*70)
        
        # Load training data
        print(f"\n📁 Loading training data from: {training_data_path}")
        train_df = pd.read_csv(training_data_path)
        
        print(f"✅ Loaded {len(train_df)} records")
        print(f"✅ Columns: {list(train_df.columns)}")
        
        # Validate required columns
        required_cols = ['State', 'District', 'Year', 'Month', 'Dengue_Cases', 
                        'Malaria_Cases', 'Chikungunya_Cases', 'Temperature', 
                        'Humidity', 'Rainfall', 'Sanitation_Score', 'Population_Density']
        
        missing_cols = [col for col in required_cols if col not in train_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Train models
        train_df, feature_cols = self.train_severity_model(train_df)
        self.train_outbreak_classifier(train_df, feature_cols)
        
        # Save models
        self.save_models()
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        
        return self
    
    def predict_pipeline(self, prediction_data_path: str, 
                        output_path: str = "outputs/predictions_enhanced.csv") -> pd.DataFrame:
        """
        Complete prediction pipeline
        """
        print("\n" + "="*70)
        print("FEVERCAST360 ENHANCED ML PIPELINE - PREDICTION MODE")
        print("="*70)
        
        # Load models
        self.load_models()
        
        # Load prediction data
        print(f"\n📁 Loading prediction data from: {prediction_data_path}")
        pred_df = pd.read_csv(prediction_data_path)
        
        print(f"✅ Loaded {len(pred_df)} records")
        
        # Validate required columns
        required_cols = ['State', 'District', 'Year', 'Month', 'Temperature', 
                        'Humidity', 'Rainfall', 'Sanitation_Score', 'Population_Density']
        
        missing_cols = [col for col in required_cols if col not in pred_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Make predictions
        results = self.predict(pred_df)
        
        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results.to_csv(output_path, index=False)
        
        print(f"\n💾 Predictions saved to: {output_path}")
        
        print("\n" + "="*70)
        print("✅ PREDICTION COMPLETE!")
        print("="*70)
        
        return results


# Streamlit integration functions
def run_enhanced_pipeline_train(training_csv: str, models_dir: str = "models_enhanced") -> FeverOutbreakPredictor:
    """
    Wrapper function for Streamlit training integration
    
    Parameters:
    -----------
    training_csv : str
        Path to training data CSV
    models_dir : str
        Directory to save models
    
    Returns:
    --------
    FeverOutbreakPredictor : Trained predictor instance
    """
    predictor = FeverOutbreakPredictor(models_dir=models_dir)
    predictor.train_pipeline(training_csv)
    return predictor


def run_enhanced_pipeline_predict(prediction_csv: str, 
                                  models_dir: str = "models_enhanced",
                                  output_path: str = "outputs/predictions_enhanced.csv") -> pd.DataFrame:
    """
    Wrapper function for Streamlit prediction integration
    
    Parameters:
    -----------
    prediction_csv : str
        Path to prediction data CSV
    models_dir : str
        Directory containing trained models
    output_path : str
        Path to save predictions
    
    Returns:
    --------
    pd.DataFrame : Prediction results
    """
    predictor = FeverOutbreakPredictor(models_dir=models_dir)
    results = predictor.predict_pipeline(prediction_csv, output_path)
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FeverCast360 Enhanced ML Pipeline")
    parser.add_argument("--mode", choices=["train", "predict"], required=True,
                       help="Mode: train or predict")
    parser.add_argument("--training_data", type=str,
                       help="Path to training CSV file")
    parser.add_argument("--prediction_data", type=str,
                       help="Path to prediction CSV file")
    parser.add_argument("--models_dir", type=str, default="models_enhanced",
                       help="Directory for models")
    parser.add_argument("--output", type=str, default="outputs/predictions_enhanced.csv",
                       help="Output path for predictions")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        if not args.training_data:
            raise ValueError("--training_data is required for training mode")
        run_enhanced_pipeline_train(
            training_csv=args.training_data,
            models_dir=args.models_dir
        )
    else:
        if not args.prediction_data:
            raise ValueError("--prediction_data is required for prediction mode")
        run_enhanced_pipeline_predict(
            prediction_csv=args.prediction_data,
            models_dir=args.models_dir,
            output_path=args.output
        )
