"""Generate submission files for Kaggle."""
import pandas as pd
import numpy as np
from typing import Dict, Any
import pickle
import os


def _prepare_inference_features(model, test_df: pd.DataFrame) -> pd.DataFrame:
    """Build inference matrix consistent with numeric-only model training."""
    features = test_df.select_dtypes(include=[np.number]).copy()
    if hasattr(model, 'feature_names_in_'):
        cols = [c for c in model.feature_names_in_ if c in features.columns]
        if cols:
            features = features[cols]
    return features


def generate_submission(model, test_df: pd.DataFrame, journey_ids: pd.Series,
                       model_name: str, output_dir: str = '../submissions/') -> str:
    """Generate Kaggle submission file."""

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    inference_df = _prepare_inference_features(model, test_df)

    # Predict probabilities
    if hasattr(model, 'predict_proba'):
        predictions = model.predict_proba(inference_df)[:, 1]
    else:
        predictions = model.predict(inference_df)

    # Create submission dataframe
    submission = pd.DataFrame({
        'journey_id': journey_ids,
        'is_delayed': predictions
    })

    # Ensure predictions are between 0 and 1
    submission['is_delayed'] = submission['is_delayed'].clip(0, 1)

    # Save to CSV
    filename = f"{output_dir}submission_{model_name}.csv"
    submission.to_csv(filename, index=False)

    print(f"Submission saved: {filename}")
    print(f"Predictions shape: {submission.shape}")
    print(f"Prediction stats:")
    print(f"  Mean: {predictions.mean():.4f}")
    print(f"  Std: {predictions.std():.4f}")
    print(f"  Min: {predictions.min():.4f}")
    print(f"  Max: {predictions.max():.4f}")

    return filename


def generate_ensemble_submission(model_results: Dict[str, Any], test_df: pd.DataFrame,
                                 journey_ids: pd.Series, weights: Dict[str, float] = None,
                                 output_dir: str = '../submissions/') -> str:
    """Generate ensemble submission with weighted averaging."""

    os.makedirs(output_dir, exist_ok=True)

    if weights is None:
        # Use equal weights if not specified
        weights = {name: 1.0 for name in model_results.keys()}

    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    # Weighted average of predictions
    ensemble_pred = np.zeros(len(test_df))
    for model_name, result in model_results.items():
        if model_name in weights:
            model = result['model']
            inference_df = _prepare_inference_features(model, test_df)
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(inference_df)[:, 1]
            else:
                pred = model.predict(inference_df)
            ensemble_pred += pred * weights[model_name]

    # Create submission
    submission = pd.DataFrame({
        'journey_id': journey_ids,
        'is_delayed': ensemble_pred
    })

    submission['is_delayed'] = submission['is_delayed'].clip(0, 1)

    filename = f"{output_dir}submission_ensemble.csv"
    submission.to_csv(filename, index=False)

    print(f"\nEnsemble submission saved: {filename}")
    print(f"Ensemble weights: {weights}")

    return filename


def save_model(model, model_name: str, output_dir: str = '../models/'):
    """Save trained model to disk."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}{model_name}.pkl"

    with open(filename, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved: {filename}")


def load_model(model_name: str, output_dir: str = '../models/'):
    """Load trained model from disk."""
    filename = f"{output_dir}{model_name}.pkl"

    with open(filename, 'rb') as f:
        model = pickle.load(f)

    print(f"Model loaded: {filename}")
    return model
