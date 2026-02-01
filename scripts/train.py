"""
Model Training Script
Trains XGBoost/LightGBM ensemble for prompt injection detection
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.features.extractor import FeatureExtractor


class ModelTrainer:
    """Train and evaluate prompt injection detection models"""
    
    def __init__(self, config_path="config/config.yaml"):
        """Initialize trainer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = Path(self.config['paths']['data_dir'])
        self.model_dir = Path(self.config['paths']['model_dir'])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_extractor = None
        self.models = {}
        self.feature_names = None
    
    def load_data(self):
        """Load and split dataset"""
        print("\n" + "="*60)
        print("Loading Dataset")
        print("="*60)
        
        dataset_path = self.data_dir / "processed" / "combined_dataset.csv"
        
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}.\n"
                f"Run 'python scripts/download_datasets.py' first."
            )
        
        df = pd.read_csv(dataset_path)
        print(f"✓ Loaded {len(df)} samples")
        print(f"  - Injection: {df['label'].sum()}")
        print(f"  - Benign: {(1 - df['label']).sum()}")
        print(f"  - Balance: {df['label'].mean():.1%} injection")
        
        # Split into train/validation/test
        train_df, temp_df = train_test_split(
            df, 
            test_size=0.3, 
            random_state=42,
            stratify=df['label']
        )
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=42,
            stratify=temp_df['label']
        )
        
        print(f"\nDataset split:")
        print(f"  - Train: {len(train_df)} samples")
        print(f"  - Validation: {len(val_df)} samples")
        print(f"  - Test: {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def extract_features(self, train_df, val_df, test_df):
        """Extract features from text"""
        print("\n" + "="*60)
        print("Extracting Features")
        print("="*60)
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(self.config.get('features', {}))
        
        # Extract features
        print("\nExtracting training features...")
        X_train = self.feature_extractor.extract_batch(
            train_df['prompt'].tolist(),
            show_progress=True
        )
        y_train = train_df['label'].values
        
        print("\nExtracting validation features...")
        X_val = self.feature_extractor.extract_batch(
            val_df['prompt'].tolist(),
            show_progress=True
        )
        y_val = val_df['label'].values
        
        print("\nExtracting test features...")
        X_test = self.feature_extractor.extract_batch(
            test_df['prompt'].tolist(),
            show_progress=True
        )
        y_test = test_df['label'].values
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        print(f"\n✓ Feature extraction complete")
        print(f"  - Feature count: {len(self.feature_names)}")
        print(f"  - Sample shape: {X_train.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        print("\n" + "="*60)
        print("Training XGBoost")
        print("="*60)
        
        params = self.config['models']['xgboost']
        
        model = xgb.XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50
        )
        
        # Validation predictions
        val_pred = model.predict(X_val)
        val_prob = model.predict_proba(X_val)[:, 1]
        
        print("\nValidation Results:")
        print(classification_report(y_val, val_pred, target_names=['Benign', 'Injection']))
        print(f"ROC-AUC: {roc_auc_score(y_val, val_prob):.4f}")
        
        self.models['xgboost'] = model
        return model
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model"""
        print("\n" + "="*60)
        print("Training LightGBM")
        print("="*60)
        
        params = self.config['models']['lightgbm']
        
        model = lgb.LGBMClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            num_leaves=params.get('num_leaves', 31),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            random_state=42,
            n_jobs=-1,
            verbosity=1
        )
        
        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(50)]
        )
        
        # Validation predictions
        val_pred = model.predict(X_val)
        val_prob = model.predict_proba(X_val)[:, 1]
        
        print("\nValidation Results:")
        print(classification_report(y_val, val_pred, target_names=['Benign', 'Injection']))
        print(f"ROC-AUC: {roc_auc_score(y_val, val_prob):.4f}")
        
        self.models['lightgbm'] = model
        return model
    
    def create_ensemble(self, X_val, y_val):
        """Create ensemble of XGBoost and LightGBM"""
        print("\n" + "="*60)
        print("Creating Ensemble")
        print("="*60)
        
        # Get predictions from both models
        xgb_prob = self.models['xgboost'].predict_proba(X_val)[:, 1]
        lgb_prob = self.models['lightgbm'].predict_proba(X_val)[:, 1]
        
        # Simple average ensemble
        ensemble_prob = (xgb_prob + lgb_prob) / 2
        ensemble_pred = (ensemble_prob > 0.5).astype(int)
        
        print("\nEnsemble Validation Results:")
        print(classification_report(y_val, ensemble_pred, target_names=['Benign', 'Injection']))
        print(f"ROC-AUC: {roc_auc_score(y_val, ensemble_prob):.4f}")
        
        # Store ensemble weights
        self.ensemble_weights = {'xgboost': 0.5, 'lightgbm': 0.5}
        
        return ensemble_pred, ensemble_prob
    
    def evaluate_on_test(self, X_test, y_test):
        """Final evaluation on test set"""
        print("\n" + "="*60)
        print("Final Test Set Evaluation")
        print("="*60)
        
        # Individual models
        xgb_prob = self.models['xgboost'].predict_proba(X_test)[:, 1]
        lgb_prob = self.models['lightgbm'].predict_proba(X_test)[:, 1]
        
        # Ensemble
        ensemble_prob = (xgb_prob + lgb_prob) / 2
        ensemble_pred = (ensemble_prob > 0.5).astype(int)
        
        print("\n--- Ensemble Results ---")
        print(classification_report(y_test, ensemble_pred, target_names=['Benign', 'Injection']))
        
        print(f"\nROC-AUC Score: {roc_auc_score(y_test, ensemble_prob):.4f}")
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, ensemble_pred)
        print(cm)
        print(f"\nTrue Negatives:  {cm[0,0]}")
        print(f"False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}")
        print(f"True Positives:  {cm[1,1]}")
        
        return ensemble_pred, ensemble_prob
    
    def save_models(self):
        """Save trained models and feature extractor"""
        print("\n" + "="*60)
        print("Saving Models")
        print("="*60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual models
        for name, model in self.models.items():
            path = self.model_dir / f"{name}_{timestamp}.joblib"
            joblib.dump(model, path)
            print(f"✓ Saved {name}: {path}")
        
        # Save feature extractor
        extractor_path = self.model_dir / f"feature_extractor_{timestamp}.joblib"
        joblib.dump(self.feature_extractor, extractor_path)
        print(f"✓ Saved feature extractor: {extractor_path}")
        
        # Save ensemble weights
        weights_path = self.model_dir / f"ensemble_weights_{timestamp}.yaml"
        with open(weights_path, 'w') as f:
            yaml.dump(self.ensemble_weights, f)
        print(f"✓ Saved ensemble weights: {weights_path}")
        
        # Save feature names
        features_path = self.model_dir / f"feature_names_{timestamp}.txt"
        with open(features_path, 'w') as f:
            f.write('\n'.join(self.feature_names))
        print(f"✓ Saved feature names: {features_path}")
        
        # Save "latest" symlinks
        for name, model in self.models.items():
            latest_path = self.model_dir / f"{name}_latest.joblib"
            if latest_path.exists():
                latest_path.unlink()
            os.symlink(f"{name}_{timestamp}.joblib", latest_path)
        
        latest_extractor = self.model_dir / "feature_extractor_latest.joblib"
        if latest_extractor.exists():
            latest_extractor.unlink()
        os.symlink(f"feature_extractor_{timestamp}.joblib", latest_extractor)
        
        print("\n✓ All models saved successfully")
    
    def run(self):
        """Run complete training pipeline"""
        print("\n" + "="*60)
        print("PROMPT INJECTION DETECTOR - MODEL TRAINING")
        print("="*60)
        
        # Load data
        train_df, val_df, test_df = self.load_data()
        
        # Extract features
        X_train, X_val, X_test, y_train, y_val, y_test = self.extract_features(
            train_df, val_df, test_df
        )
        
        # Train models
        self.train_xgboost(X_train, y_train, X_val, y_val)
        self.train_lightgbm(X_train, y_train, X_val, y_val)
        
        # Create ensemble
        self.create_ensemble(X_val, y_val)
        
        # Final evaluation
        self.evaluate_on_test(X_test, y_test)
        
        # Save everything
        self.save_models()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print("\nNext steps:")
        print("  1. Export to ONNX: python scripts/export_onnx.py")
        print("  2. Test inference: python scripts/test_inference.py")
        print("  3. Deploy to OpenClaw: See deployment guide")


def main():
    """Main execution"""
    trainer = ModelTrainer()
    trainer.run()
    return 0


if __name__ == "__main__":
    exit(main())
