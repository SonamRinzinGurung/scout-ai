# =============================================
# PRODUCTION-READY NBA DRAFT PREDICTOR CLASS
# =============================================
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Union


class NBADraftPredictor:
    """
    NBA Draft Prospect Predictor
    Predicts if a college basketball player will be drafted to NBA
    """

    def __init__(self, model_path: str, preprocessor_path: str):
        """Load trained model and preprocessors"""
        self.model = joblib.load(model_path)
        self.preprocessors = joblib.load(preprocessor_path)

        # Extract preprocessors
        self.num_imputer = self.preprocessors['num_imputer']
        self.cat_imputer = self.preprocessors['cat_imputer']
        self.scaler = self.preprocessors['scaler']
        self.encoder = self.preprocessors['encoder']
        self.numeric_features = self.preprocessors['numeric_features']
        self.categorical_features = self.preprocessors['categorical_features']
        self.feature_columns = self.preprocessors['feature_columns']

        print("✅ NBA Draft Predictor loaded successfully!")
        print(f"   Model: {type(self.model).__name__}")
        print(f"   Expected features: {len(self.feature_columns)}")

    def add_draft_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features used during training"""
        df_enhanced = df.copy()

        # Elite performance indicators (top 10% thresholds from training data)
        df_enhanced['elite_scorer'] = (df['pts'] > 20.0).astype(
            int)  # Approximate threshold
        df_enhanced['elite_efficiency'] = (df['TS_per'] > 0.65).astype(int)
        df_enhanced['elite_impact'] = (df['bpm'] > 8.0).astype(int)

        # Size + skill combination
        df_enhanced['versatile_big'] = (
            (df['role_position'].isin(['PF', 'C'])) &
            (df['TP_per'] > 0.3)
        ).astype(int)

        # Usage + efficiency
        df_enhanced['high_usage_efficient'] = (
            (df['usg'] > 25.0) &  # Approximate 80th percentile
            (df['TS_per'] > 0.55)
        ).astype(int)

        # Power conference performance
        power_confs = ['ACC', 'Big 12', 'Big Ten', 'SEC', 'Pac-12', 'Big East']
        df_enhanced['power_conf_star'] = (
            (df['conf'].isin(power_confs)) &
            (df['pts'] > 15)
        ).astype(int)

        return df_enhanced

    def preprocess_data(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocess input data exactly like training data"""

        # Add engineered features
        df_processed = self.add_draft_features(df)

        # Ensure all required columns are present
        missing_cols = set(self.feature_columns) - set(df_processed.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Select only the features used in training
        df_processed = df_processed[self.feature_columns]

        # Separate numeric and categorical
        df_num = df_processed[self.numeric_features]
        df_cat = df_processed[self.categorical_features]

        # Apply same preprocessing as training
        X_num = self.num_imputer.transform(df_num)
        X_cat = self.cat_imputer.transform(df_cat)

        # Scale and encode
        X_num = self.scaler.transform(X_num)
        X_cat = self.encoder.transform(X_cat)

        # Combine features
        X_processed = np.hstack([X_num, X_cat])

        return X_processed

    def predict_draft_probability(self, player_data: Union[Dict, pd.DataFrame]) -> Dict:
        """
        Predict draft probability for a player or players

        Args:
            player_data: Dict with player stats or DataFrame with multiple players

        Returns:
            Dict with prediction results
        """
        # Convert dict to DataFrame if needed
        if isinstance(player_data, dict):
            df = pd.DataFrame([player_data])
            single_player = True
        else:
            df = player_data.copy()
            single_player = False

        try:
            # Preprocess data
            X_processed = self.preprocess_data(df)

            # Get predictions and probabilities
            predictions = self.model.predict(X_processed)
            probabilities = self.model.predict_proba(
                X_processed)[:, 1]  # Probability of being drafted

            # Prepare results
            results = {
                'predictions': predictions.tolist(),
                'draft_probabilities': probabilities.tolist(),
                'draft_decisions': ['Drafted' if pred == 1 else 'Undrafted' for pred in predictions]
            }

            # Add player names if available
            if 'player_name' in df.columns:
                results['player_names'] = df['player_name'].tolist()

            # If single player, return simplified format
            if single_player:
                return {
                    'player_name': df['player_name'].iloc[0] if 'player_name' in df.columns else 'Unknown',
                    'draft_probability': float(probabilities[0]),
                    'prediction': 'Drafted' if predictions[0] == 1 else 'Undrafted',
                    'confidence': 'High' if probabilities[0] > 0.7 or probabilities[0] < 0.3 else 'Medium'
                }

            return results

        except Exception as e:
            return {'error': f"Prediction failed: {str(e)}"}

