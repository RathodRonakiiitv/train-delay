"""Train all models and generate submission - Command line version."""
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from data_utils import load_data, preprocess_data, create_train_val_split
from features import engineer_features
from models import train_all_models, evaluate_model
from submission import generate_submission, generate_ensemble_submission, save_model


def main():
    print("="*60)
    print("Indian Railways Train Delay Prediction - Training Pipeline")
    print("="*60)

    # 1. Load data
    print("\n[1/6] Loading data...")
    train, test = load_data('./data/')
    print(f"Train: {train.shape}, Test: {test.shape}")
    print(f"Delay rate: {train['is_delayed'].mean():.2%}")

    # 2. Preprocess
    print("\n[2/6] Preprocessing...")
    train_processed, test_processed = preprocess_data(train, test)
    print(f"Processed - Train: {train_processed.shape}, Test: {test_processed.shape}")

    # 3. Feature engineering
    print("\n[3/6] Feature engineering...")
    train_fe, test_fe = engineer_features(train_processed, test_processed)
    print(f"Features - Train: {train_fe.shape}, Test: {test_fe.shape}")

    # 4. Prepare data
    print("\n[4/6] Preparing train/validation split...")
    test_journey_ids = test_fe['journey_id']

    X_train, X_val, y_train, y_val = create_train_val_split(train_fe, test_size=0.2, random_state=42)
    X_train = X_train.drop('journey_id', axis=1)
    X_val = X_val.drop('journey_id', axis=1)
    test_features = test_fe.drop('journey_id', axis=1)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # 5. Train models
    print("\n[5/6] Training models...")
    results = train_all_models(X_train, X_val, y_train, y_val)

    print("\n" + "="*60)
    print("MODEL RESULTS")
    print("="*60)
    for name, result in results.items():
        print(f"{name:20s}: Val AUC = {result['auc']:.4f}")

    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['auc'])
    print(f"\nBest model: {best_model_name} (AUC: {results[best_model_name]['auc']:.4f})")

    # 6. Generate submissions
    print("\n[6/6] Generating submissions...")

    # Individual submissions
    for name, result in results.items():
        generate_submission(result['model'], test_features, test_journey_ids, name, '../submissions/')

    # Ensemble
    weights = {
        'xgboost': 0.35,
        'lightgbm': 0.35,
        'random_forest': 0.15,
        'gradient_boosting': 0.15
    }

    generate_ensemble_submission(
        results, test_features, test_journey_ids,
        weights=weights, output_dir='../submissions/'
    )

    # Save models
    print("\nSaving models...")
    for name, result in results.items():
        save_model(result['model'], name, '../models/')

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nCheck these files:")
    print("  - Submissions: ../submissions/")
    print("  - Best model:  ../models/")
    print("\nSubmit 'submission_ensemble.csv' to Kaggle!")


if __name__ == "__main__":
    main()

