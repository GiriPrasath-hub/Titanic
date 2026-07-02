"""
Preprocessing utilities for Titanic Survival Prediction.
Provides modular and reusable functions for dataset loading, inspection,
cleaning, feature engineering, and path setup.
"""

import re
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads the dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        raise IOError(f"Error loading dataset from {filepath}: {e}")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically normalizes all column names:
    - Removes leading numbers, spaces, hidden characters, and special symbols.
    - Standardizes target column (matching 'survived' or similar) to 'Survived'.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with normalized column names.
    """
    df_norm = df.copy()
    new_cols = {}
    for col in df_norm.columns:
        # Strip spaces and special symbols
        clean_col = col.strip()
        # Remove leading numbers and underscores
        clean_col = re.sub(r"^[\d_]+", "", clean_col)
        # Standardize target
        if clean_col.lower() in ["survived", "urvived"]:
            clean_col = "Survived"
        else:
            # Remove any special character
            clean_col = re.sub(r"[^\w]", "", clean_col)
        new_cols[col] = clean_col
    df_norm.rename(columns=new_cols, inplace=True)
    return df_norm


def split_features_target(
    df: pd.DataFrame, target_col: str = "Survived"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits the dataset into features (X) and target (y).

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Name of the target column. Default is 'Survived'.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features DataFrame and Target Series.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def detect_feature_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Automatically detects and categorizes feature types:
    numerical, categorical, and binary features.

    Args:
        df (pd.DataFrame): DataFrame containing features.

    Returns:
        Dict[str, List[str]]: Dictionary containing lists of column names for
        each category: 'numerical', 'categorical', 'binary'.
    """
    numerical = []
    categorical = []
    binary = []

    for col in df.columns:
        # Exclude common ID columns
        if col.lower() in ["passengerid", "id"]:
            continue

        unique_vals = df[col].dropna().unique()
        n_unique = len(unique_vals)

        # Binary feature check
        if n_unique == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0, "0", "1"}):
            binary.append(col)
        # Categorical feature check (non-float with low unique values)
        elif n_unique < 10 and not np.issubdtype(df[col].dtype, np.floating):
            categorical.append(col)
        # Numerical feature check
        elif np.issubdtype(df[col].dtype, np.number):
            # If the float has few unique values (e.g. Embarked is float but categorical)
            if n_unique < 5:
                categorical.append(col)
            else:
                numerical.append(col)
        else:
            categorical.append(col)

    return {
        "numerical": numerical,
        "categorical": categorical,
        "binary": binary,
    }


def clean_dataset(df: pd.DataFrame, drop_zeros: bool = True) -> pd.DataFrame:
    """
    Cleans the dataset by:
    1. Dropping redundant 'zero' columns.
    2. Dropping identifier columns like 'Passengerid'.

    Args:
        df (pd.DataFrame): Input DataFrame.
        drop_zeros (bool): Whether to drop columns containing 'zero'.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()

    # Drop zero columns
    if drop_zeros:
        zero_cols = [col for col in df_clean.columns if "zero" in col.lower()]
        df_clean.drop(columns=zero_cols, inplace=True, errors="ignore")

    # Drop PassengerId
    id_cols = [col for col in df_clean.columns if col.lower() in ["passengerid", "id"]]
    df_clean.drop(columns=id_cols, inplace=True, errors="ignore")

    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates engineered features:
    1. 'FamilySize' (SibSp + Parch + 1)
    2. 'IsAlone' (1 if FamilySize == 1 else 0)
    3. extracts 'Title' from 'Name' (if present)
    4. extracts 'Deck' from 'Cabin' (if present)

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    df_feat = df.copy()

    # Normalize column names in lowercase for case-insensitive checks
    col_mapping = {col.lower(): col for col in df_feat.columns}

    # Family Size & IsAlone
    sibsp_col = col_mapping.get("sibsp")
    parch_col = col_mapping.get("parch")
    if sibsp_col and parch_col:
        df_feat["FamilySize"] = df_feat[sibsp_col] + df_feat[parch_col] + 1
        df_feat["IsAlone"] = (df_feat["FamilySize"] == 1).astype(int)

    # Title extraction
    name_col = col_mapping.get("name")
    if name_col:
        df_feat["Title"] = df_feat[name_col].str.extract(r" ([A-Za-z]+)\.", expand=False)
        # Fill missing Title with 'Unknown'
        df_feat["Title"] = df_feat["Title"].fillna("Unknown")

    # Deck extraction
    cabin_col = col_mapping.get("cabin")
    if cabin_col:
        df_feat["Deck"] = df_feat[cabin_col].astype(str).str[0].str.upper()
        # Handle default nans or empty cabin values
        df_feat["Deck"] = df_feat["Deck"].replace({"N": "Unknown"})
        df_feat["Deck"] = df_feat["Deck"].fillna("Unknown")

    return df_feat


def prepare_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits features and target into training and testing sets.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        test_size (float): Proportion of test split.
        random_state (int): Random state seed.

    Returns:
        Tuple: X_train, X_test, y_train, y_test.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
