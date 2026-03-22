"""Data loading and preprocessing utilities."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


def load_data(data_path: str = './data/') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test datasets."""
    train = pd.read_csv(f'{data_path}ir_train.csv')
    test = pd.read_csv(f'{data_path}ir_test.csv')
    return train, test


def get_data_dict(data_path: str = './data/') -> pd.DataFrame:
    """Load data dictionary."""
    return pd.read_csv(f'{data_path}ir_data_dictionary.csv')


def identify_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Identify numeric, categorical, and datetime columns."""
    # Drop leakage columns
    leakage_cols = ['primary_delay_cause', 'delay_minutes', 'journey_id']
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns], errors='ignore')

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Separate target
    if 'is_delayed' in numeric_cols:
        numeric_cols.remove('is_delayed')

    # Identify datetime-like columns
    datetime_cols = []
    for col in categorical_cols:
        if 'date' in col.lower() or 'time' in col.lower():
            datetime_cols.append(col)

    # Remove datetime from categorical
    categorical_cols = [c for c in categorical_cols if c not in datetime_cols]

    return numeric_cols, categorical_cols, datetime_cols


def preprocess_data(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Basic preprocessing: handle missing values, encode categoricals."""
    # Remove leakage columns
    leakage_cols = ['primary_delay_cause', 'delay_minutes']
    train = train.drop(columns=[c for c in leakage_cols if c in train.columns], errors='ignore')

    # Combine for consistent encoding
    train['is_train'] = 1
    test['is_train'] = 0

    combined = pd.concat([train, test], ignore_index=True)

    # Parse dates
    if 'departure_date' in combined.columns:
        combined['departure_date'] = pd.to_datetime(combined['departure_date'])
        combined['departure_year'] = combined['departure_date'].dt.year
        combined['departure_month'] = combined['departure_date'].dt.month
        combined['departure_day'] = combined['departure_date'].dt.day
        combined['departure_dayofyear'] = combined['departure_date'].dt.dayofyear

    # Handle missing values
    numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ['is_delayed', 'is_train']]

    for col in numeric_cols:
        combined[col] = combined[col].fillna(combined[col].median())

    # Encode categoricals
    categorical_cols = combined.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in ['journey_id', 'departure_date']]

    for col in categorical_cols:
        combined[col] = combined[col].fillna('Unknown')
        combined[col] = combined[col].astype('category').cat.codes

    # Split back
    train_processed = combined[combined['is_train'] == 1].drop('is_train', axis=1)
    test_processed = combined[combined['is_train'] == 0].drop(['is_train', 'is_delayed'], axis=1, errors='ignore')

    return train_processed, test_processed


def create_train_val_split(train: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Create train/validation split preserving time order if needed."""
    y = train['is_delayed']
    X = train.drop('is_delayed', axis=1)

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
