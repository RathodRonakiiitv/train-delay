"""Model training and evaluation."""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')


def train_logistic_regression(X_train, X_val, y_train, y_val) -> Dict[str, Any]:
    """Train logistic regression model."""
    # Select only numeric columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train_numeric = X_train[numeric_cols]
    X_val_numeric = X_val[numeric_cols]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_numeric)
    X_val_scaled = scaler.transform(X_val_numeric)

    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    val_pred = model.predict_proba(X_val_scaled)[:, 1]
    auc = roc_auc_score(y_val, val_pred)

    return {
        'model': model,
        'scaler': scaler,
        'auc': auc,
        'predictions': val_pred
    }


def train_random_forest(X_train, X_val, y_train, y_val) -> Dict[str, Any]:
    """Train Random Forest model."""
    # Select only numeric columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train_numeric = X_train[numeric_cols]
    X_val_numeric = X_val[numeric_cols]
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train_numeric, y_train)

    val_pred = model.predict_proba(X_val_numeric)[:, 1]
    auc = roc_auc_score(y_val, val_pred)

    return {
        'model': model,
        'auc': auc,
        'predictions': val_pred,
        'feature_importance': dict(zip(numeric_cols, model.feature_importances_))
    }


def train_gradient_boosting(X_train, X_val, y_train, y_val) -> Dict[str, Any]:
    """Train Gradient Boosting model."""
    # Select only numeric columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train_numeric = X_train[numeric_cols]
    X_val_numeric = X_val[numeric_cols]
    
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train_numeric, y_train)

    val_pred = model.predict_proba(X_val_numeric)[:, 1]
    auc = roc_auc_score(y_val, val_pred)

    return {
        'model': model,
        'auc': auc,
        'predictions': val_pred,
        'feature_importance': dict(zip(numeric_cols, model.feature_importances_))
    }


def train_xgboost(X_train, X_val, y_train, y_val) -> Dict[str, Any]:
    """Train XGBoost model."""
    # Select only numeric columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train_numeric = X_train[numeric_cols]
    X_val_numeric = X_val[numeric_cols]
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=0.4,  # For imbalanced data (28% positive)
        random_state=42,
        n_jobs=-1,
        eval_metric='auc'
    )

    try:
        model.fit(
            X_train_numeric, y_train,
            eval_set=[(X_val_numeric, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
    except TypeError:
        # XGBoost 3.x removed early_stopping_rounds from fit() kwargs.
        model.fit(
            X_train_numeric, y_train,
            eval_set=[(X_val_numeric, y_val)],
            verbose=False
        )

    val_pred = model.predict_proba(X_val_numeric)[:, 1]
    auc = roc_auc_score(y_val, val_pred)

    importance = model.get_booster().get_score(importance_type='gain')
    feature_importance = {k: v for k, v in importance.items()}
    try:
        best_iteration = model.best_iteration
    except AttributeError:
        best_iteration = None

    return {
        'model': model,
        'auc': auc,
        'predictions': val_pred,
        'feature_importance': feature_importance,
        'best_iteration': best_iteration
    }


def train_lightgbm(X_train, X_val, y_train, y_val) -> Dict[str, Any]:
    """Train LightGBM model."""
    # Select only numeric columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train_numeric = X_train[numeric_cols]
    X_val_numeric = X_val[numeric_cols]
    
    model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=0.4,
        random_state=42,
        n_jobs=-1,
        metric='auc',
        verbose=-1
    )

    model.fit(
        X_train_numeric, y_train,
        eval_set=[(X_val_numeric, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
    )

    val_pred = model.predict_proba(X_val_numeric)[:, 1]
    auc = roc_auc_score(y_val, val_pred)

    importance = dict(zip(numeric_cols, model.feature_importances_))

    return {
        'model': model,
        'auc': auc,
        'predictions': val_pred,
        'feature_importance': importance,
        'best_iteration': model.best_iteration_
    }


def train_all_models(X_train, X_val, y_train, y_val) -> Dict[str, Any]:
    """Train all models and return results."""
    results = {}

    print("Training Logistic Regression...")
    results['logistic_regression'] = train_logistic_regression(X_train, X_val, y_train, y_val)

    print("Training Random Forest...")
    results['random_forest'] = train_random_forest(X_train, X_val, y_train, y_val)

    print("Training Gradient Boosting...")
    results['gradient_boosting'] = train_gradient_boosting(X_train, X_val, y_train, y_val)

    print("Training XGBoost...")
    results['xgboost'] = train_xgboost(X_train, X_val, y_train, y_val)

    print("Training LightGBM...")
    results['lightgbm'] = train_lightgbm(X_train, X_val, y_train, y_val)

    return results


def get_top_features(model_result: Dict, n: int = 15) -> pd.DataFrame:
    """Get top N important features."""
    if 'feature_importance' not in model_result:
        return None

    importance = model_result['feature_importance']
    df = pd.DataFrame(list(importance.items()), columns=['feature', 'importance'])
    df = df.sort_values('importance', ascending=False).head(n)
    return df


def evaluate_model(y_true, y_pred, model_name: str = ""):
    """Print evaluation metrics."""
    auc = roc_auc_score(y_true, y_pred)
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(f"AUC-ROC: {auc:.4f}")

    # Classification at 0.5 threshold
    y_pred_binary = (y_pred > 0.5).astype(int)
    print(f"\nClassification Report (threshold=0.5):")
    print(classification_report(y_true, y_pred_binary))

    return auc
