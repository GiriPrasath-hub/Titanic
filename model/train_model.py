"""
Model training, hyperparameter tuning, and serialization script.
Loads dataset, normalizes columns, constructs sklearn pipeline,
tunes DecisionTreeClassifier using GridSearchCV, and serializes the assets.
"""

import logging
import os
import pickle
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from utils.preprocess import (
    clean_dataset,
    detect_feature_types,
    engineer_features,
    load_dataset,
    normalize_column_names,
    prepare_train_test_split,
    split_features_target,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def build_preprocessor(
    num_cols: List[str], cat_cols: List[str], bin_cols: List[str]
) -> ColumnTransformer:
    """
    Builds the ColumnTransformer pipeline for preprocessing.

    Args:
        num_cols (List[str]): Numerical feature columns.
        cat_cols (List[str]): Categorical feature columns.
        bin_cols (List[str]): Binary feature columns.

    Returns:
        ColumnTransformer: Preconfigured column transformer.
    """
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    bin_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
            ("bin", bin_transformer, bin_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def train_and_evaluate() -> None:
    """
    Main training workflow:
    - Load data
    - Normalize column names
    - Preprocess and Split
    - Run Grid Search Cross-Validation
    - Evaluate model
    - Save serialized models
    """
    dataset_path = "dataset/train_and_test2.csv"
    logger.info(f"Loading raw dataset from {dataset_path}")
    df_raw = load_dataset(dataset_path)

    logger.info("Normalizing column names")
    df_norm = normalize_column_names(df_raw)

    logger.info("Cleaning dataset and dropping zero columns")
    df_cleaned = clean_dataset(df_norm, drop_zeros=True)

    logger.info("Performing feature engineering")
    df_features = engineer_features(df_cleaned)

    # Split into features (X) and target (y)
    X, y = split_features_target(df_features, target_col="Survived")

    # Define features lists dynamically
    num_cols = ["Age", "Fare", "sibsp", "Parch"]
    if "FamilySize" in X.columns:
        num_cols.append("FamilySize")

    cat_cols = []
    for col in ["Pclass", "Embarked", "Title", "Deck"]:
        if col in X.columns:
            cat_cols.append(col)

    bin_cols = []
    for col in ["Sex", "IsAlone"]:
        if col in X.columns:
            bin_cols.append(col)

    logger.info(f"Numerical features: {num_cols}")
    logger.info(f"Categorical features: {cat_cols}")
    logger.info(f"Binary features: {bin_cols}")

    # Build Pipeline
    preprocessor = build_preprocessor(num_cols, cat_cols, bin_cols)

    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(random_state=42)),
        ]
    )

    # Perform Train-Test Split
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(
        f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}"
    )

    # Hyperparameter Tuning Grid
    param_grid = {
        "classifier__criterion": ["gini", "entropy"],
        "classifier__max_depth": [3, 5, 7, 10, 15, 20, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2", None],
    }

    logger.info("Starting GridSearchCV hyperparameter tuning...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=full_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    logger.info("GridSearchCV complete!")
    logger.info(f"Best parameters found: {best_params}")
    logger.info(f"Best CV ROC-AUC: {best_score:.4f}")

    # Fit preprocessor separately to obtain feature columns and transform schemas
    preprocessor_fitted = best_pipeline.named_steps["preprocessor"]
    classifier_fitted = best_pipeline.named_steps["classifier"]

    # Retrieve transformed feature names
    feature_names = preprocessor_fitted.get_feature_names_out()
    # Clean up prefixes from get_feature_names_out (e.g. 'num__Age' -> 'Age')
    clean_feature_names = [
        f.replace("num__", "")
        .replace("cat__", "")
        .replace("bin__", "")
        .replace("remainder__", "")
        for f in feature_names
    ]
    logger.info(f"Final Preprocessed Columns: {clean_feature_names}")

    # Evaluate on Train & Test
    y_train_pred = best_pipeline.predict(X_train)
    y_train_pred_proba = best_pipeline.predict_proba(X_train)[:, 1]

    y_test_pred = best_pipeline.predict(X_test)
    y_test_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred)
    test_rec = recall_score(y_test, y_test_pred)
    test_wf1 = f1_score(y_test, y_test_pred, average="weighted")
    test_mf1 = f1_score(y_test, y_test_pred, average="macro")
    test_roc = roc_auc_score(y_test, y_test_pred_proba)

    logger.info("--- Model Performance Metrics ---")
    logger.info(f"Training Accuracy: {train_acc:.4f}")
    logger.info(f"Testing Accuracy: {test_acc:.4f}")
    logger.info(f"Testing Precision: {test_prec:.4f}")
    logger.info(f"Testing Recall: {test_rec:.4f}")
    logger.info(f"Testing Weighted F1: {test_wf1:.4f}")
    logger.info(f"Testing Macro F1: {test_mf1:.4f}")
    logger.info(f"Testing ROC-AUC: {test_roc:.4f}")

    logger.info("\n--- Classification Report (Test Set) ---")
    logger.info(f"\n{classification_report(y_test, y_test_pred)}")

    logger.info("--- Confusion Matrix (Test Set) ---")
    cm = confusion_matrix(y_test, y_test_pred)
    logger.info(f"\n{cm}")

    # Overfitting analysis
    logger.info("\n--- Overfitting Analysis ---")
    diff = train_acc - test_acc
    if diff > 0.05:
        logger.info(
            f"The tree displays potential overfitting (Train: {train_acc:.4f}, Test: {test_acc:.4f}, Delta: {diff:.4f})."
        )
    else:
        logger.info(
            f"The model generalizes well with minimal difference (Train: {train_acc:.4f}, Test: {test_acc:.4f}, Delta: {diff:.4f})."
        )

    # Feature Importance Analysis
    importances = classifier_fitted.feature_importances_
    importance_df = pd.DataFrame(
        {"Feature": clean_feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    logger.info("\n--- Feature Importance (Top 15) ---")
    top_15 = importance_df.head(15)
    for idx, row in top_15.iterrows():
        logger.info(f"{row['Feature']}: {row['Importance']:.4f}")

    # Serialize artifacts
    os.makedirs("model", exist_ok=True)

    # Save components
    paths = {
        "model/model.pkl": best_pipeline,
        "model/preprocessor.pkl": preprocessor_fitted,
        "model/encoder.pkl": preprocessor_fitted.named_transformers_["cat"].named_steps["onehot"],
        "model/feature_columns.pkl": clean_feature_names,
    }

    for path, obj in paths.items():
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Successfully serialized asset: {path}")


if __name__ == "__main__":
    train_and_evaluate()
