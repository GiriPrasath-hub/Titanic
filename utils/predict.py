"""
Prediction module for Titanic Survival Prediction.
Provides model loading, input validation, feature preparation,
inference, decoding, and model metadata utilities.
"""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Base directory using Path
BASE_DIR = Path(__file__).resolve().parent.parent

# Cache instances
_MODEL_INSTANCE = None
_ENCODER_INSTANCE = None


def load_model(
    model_path: str = str(BASE_DIR / "model" / "model.pkl"),
) -> Pipeline:
    """
    Loads the complete sklearn Pipeline from a pickle file.
    Caches the loaded pipeline to prevent repeated disk reads.

    Args:
        model_path (str): Path to the model.pkl file.

    Returns:
        Pipeline: Loaded sklearn Pipeline.
    """
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        with open(model_path, "rb") as f:
            _MODEL_INSTANCE = pickle.load(f)
    return _MODEL_INSTANCE


def load_encoder(
    encoder_path: str = str(BASE_DIR / "model" / "encoder.pkl"),
) -> Any:
    """
    Loads the fitted OneHotEncoder from a pickle file.
    Caches the loaded encoder to prevent repeated disk reads.

    Args:
        encoder_path (str): Path to the encoder.pkl file.

    Returns:
        Any: Loaded OneHotEncoder.
    """
    global _ENCODER_INSTANCE
    if _ENCODER_INSTANCE is None:
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder file not found at: {encoder_path}")
        with open(encoder_path, "rb") as f:
            _ENCODER_INSTANCE = pickle.load(f)
    return _ENCODER_INSTANCE


def validate_input(data: dict) -> Tuple[bool, str]:
    """
    Validates user input fields.
    Checks:
    - Required fields exist.
    - Fields have correct data types.
    - Fields contain valid categories or non-negative ranges.

    Args:
        data (dict): Dict of features.

    Returns:
        Tuple[bool, str]: (isValid, errorMessage)
    """
    required_fields = ["Age", "Fare", "Sex", "sibsp", "Parch", "Pclass", "Embarked"]
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"

    # Age validation
    try:
        age = float(data["Age"])
        if age < 0 or age > 120:
            return False, "Age must be between 0 and 120."
    except (ValueError, TypeError):
        return False, "Age must be a numeric value."

    # Fare validation
    try:
        fare = float(data["Fare"])
        if fare < 0:
            return False, "Fare cannot be negative."
    except (ValueError, TypeError):
        return False, "Fare must be a numeric value."

    # Sex validation
    sex = data["Sex"]
    if sex not in [0, 1, "0", "1"]:
        return False, "Sex must be 0 (Male) or 1 (Female)."

    # Pclass validation
    pclass = data["Pclass"]
    if pclass not in [1, 2, 3, "1", "2", "3"]:
        return False, "Pclass must be 1, 2, or 3."

    # Embarked validation
    embarked = data["Embarked"]
    # Embarked is represented as 0.0, 1.0, 2.0 in the dataset
    try:
        emb_val = float(embarked)
        if emb_val not in [0.0, 1.0, 2.0]:
            return False, "Embarked must be 0.0, 1.0, or 2.0."
    except (ValueError, TypeError):
        return False, "Embarked must be a numeric value (0.0, 1.0, 2.0)."

    # sibsp validation
    try:
        sibsp = int(data["sibsp"])
        if sibsp < 0:
            return False, "sibsp cannot be negative."
    except (ValueError, TypeError):
        return False, "sibsp must be an integer."

    # Parch validation
    try:
        parch = int(data["Parch"])
        if parch < 0:
            return False, "Parch cannot be negative."
    except (ValueError, TypeError):
        return False, "Parch must be an integer."

    return True, ""


def prepare_dataframe(data: Union[dict, pd.DataFrame]) -> pd.DataFrame:
    """
    Transforms the input dictionary or raw DataFrame into the exact structure
    expected by the pipeline. Automatically calculates FamilySize and IsAlone.

    Args:
        data (Union[dict, pd.DataFrame]): Input data.

    Returns:
        pd.DataFrame: Structured DataFrame for the pipeline.
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    # Convert numeric fields
    df["Age"] = df["Age"].astype(float)
    df["Fare"] = df["Fare"].astype(float)
    df["Sex"] = df["Sex"].astype(int)
    df["sibsp"] = df["sibsp"].astype(int)
    df["Parch"] = df["Parch"].astype(int)
    df["Pclass"] = df["Pclass"].astype(float)
    df["Embarked"] = df["Embarked"].astype(float)

    # Feature Engineering
    df["FamilySize"] = df["sibsp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Return only columns that were used in the feature set during training
    final_cols = ["Age", "Fare", "Sex", "sibsp", "Parch", "Pclass", "Embarked", "FamilySize", "IsAlone"]
    return df[final_cols]


def decode_prediction(class_id: int) -> str:
    """
    Decodes prediction class ID into human-readable label.

    Args:
        class_id (int): Prediction class.

    Returns:
        str: Human readable prediction.
    """
    return "Survived" if class_id == 1 else "Did Not Survive"


def predict(
    data: Union[dict, pd.DataFrame]
) -> Tuple[str, int, float, Dict[str, float]]:
    """
    Executes end-to-end prediction.

    Args:
        data (Union[dict, pd.DataFrame]): Features dict or DataFrame.

    Returns:
        Tuple[str, int, float, Dict[str, float]]: (Prediction, Class ID, Confidence, Probability Dict)
    """
    # If dict, validate first
    if isinstance(data, dict):
        is_valid, err = validate_input(data)
        if not is_valid:
            raise ValueError(err)

    df_prepared = prepare_dataframe(data)
    model = load_model()

    class_id = int(model.predict(df_prepared)[0])
    prob_arr = model.predict_proba(df_prepared)[0]
    
    prob_dict = {
        "Did Not Survive": float(prob_arr[0]),
        "Survived": float(prob_arr[1]),
    }
    confidence = float(prob_arr[class_id])
    prediction = decode_prediction(class_id)

    return prediction, class_id, confidence, prob_dict


def predict_probability(data: Union[dict, pd.DataFrame]) -> np.ndarray:
    """
    Predicts class probabilities using the loaded pipeline.

    Args:
        data (Union[dict, pd.DataFrame]): Input features.

    Returns:
        np.ndarray: Probabilities array.
    """
    df_prepared = prepare_dataframe(data)
    model = load_model()
    return model.predict_proba(df_prepared)


def model_information() -> Dict[str, Any]:
    """
    Returns metadata about the serialized model's configuration and training.

    Returns:
        Dict[str, Any]: Model details.
    """
    return {
        "algorithm": "Decision Tree Classifier",
        "tuning_method": "GridSearchCV (5-fold stratified)",
        "cv_roc_auc": 0.7993,
        "train_accuracy": 0.8118,
        "test_accuracy": 0.7519,
        "test_precision": 0.5319,
        "test_recall": 0.3676,
        "test_f1": 0.7356,
        "best_hyperparameters": {
            "criterion": "gini",
            "max_depth": 3,
            "min_samples_leaf": 4,
            "min_samples_split": 2,
            "max_features": None,
        },
    }
