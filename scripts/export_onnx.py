"""Export models to ONNX format for faster inference"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import joblib
from pathlib import Path
import onnx
from skl2onnx import to_onnx
import numpy as np

def main():
    model_dir = Path('./data/models')
    export_dir = Path('./exports')
    export_dir.mkdir(exist_ok=True)
    
    print("Loading models...")
    xgb_model = joblib.load(model_dir / 'xgboost_latest.joblib')
    lgb_model = joblib.load(model_dir / 'lightgbm_latest.joblib')
    
    # Get feature dimension
    extractor = joblib.load(model_dir / 'feature_extractor_latest.joblib')
    dummy_features = extractor.extract("test")
    n_features = len(dummy_features)
    
    print(f"Feature dimension: {n_features}")
    
    # Export XGBoost
    print("\nExporting XGBoost to ONNX...")
    initial_type = [('float_input', np.float32([None, n_features]))]
    xgb_onnx = to_onnx(xgb_model, initial_types=initial_type, target_opset=12)
    
    xgb_path = export_dir / 'xgboost_model.onnx'
    with open(xgb_path, 'wb') as f:
        f.write(xgb_onnx.SerializeToString())
    print(f"✓ Saved: {xgb_path}")
    
    # Export LightGBM
    print("\nExporting LightGBM to ONNX...")
    lgb_onnx = to_onnx(lgb_model, initial_types=initial_type, target_opset=12)
    
    lgb_path = export_dir / 'lightgbm_model.onnx'
    with open(lgb_path, 'wb') as f:
        f.write(lgb_onnx.SerializeToString())
    print(f"✓ Saved: {lgb_path}")
    
    print("\n✓ ONNX export complete")
    print("\nYou can now use onnxruntime for faster inference:")
    print("  pip install onnxruntime")

if __name__ == "__main__":
    main()
