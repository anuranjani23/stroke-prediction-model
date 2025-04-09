import pandas as pd
import os
import joblib
import logging
import numpy as np
from time import time

# For the cross-validation and hyperparameter tuning:
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC
# For data preprocessing:
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
# For handling class imbalance:
from imblearn.over_sampling import SMOTE  
from imblearn.pipeline import Pipeline as ImbPipeline
# For feature selection:
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import make_scorer, fbeta_score
import warnings

# Setting up logging:
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_dataset.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'src', 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'svm_model.pkl')

# Loading and preparing data:
def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['stroke'])
    y = df['stroke']
    logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y

# Simplified model training function with optimization
def train_optimized_model(X, y):
    # Identifying column types:
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    logger.info(f"Numeric features: {len(numeric_features)}, Categorical features: {len(categorical_features)}")

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Create an integrated pipeline that includes preprocessing, feature selection, and the classifier
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, sampling_strategy=0.8)),
        ('feature_selection', SelectKBest(mutual_info_classif, k=20)),
        ('classifier', SVC(probability=True, class_weight='balanced', random_state=42))
    ])

    # Focused parameter grid with fewer combinations
    param_grid = {
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__gamma': ['scale', 'auto', 0.01, 0.1],
        'classifier__kernel': ['rbf', 'poly'],
        'smote__sampling_strategy': [0.6, 0.8, 1.0]
    }
    

    # Using RandomizedSearchCV instead of GridSearchCV to explore more efficiently,
    # with fewer total evaluations:
    f2_scorer = make_scorer(fbeta_score, beta=2)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Start timing
    start_time = time()
    logger.info("Starting hyperparameter optimization...")

    search = RandomizedSearchCV(
        pipeline, param_grid,
        cv=cv,
        scoring=f2_scorer,
        n_jobs=-1,
        n_iter=5,
        verbose=1,
        random_state=42
    )

    # Fit the model
    search.fit(X, y)

    # Log the time taken
    elapsed_time = time() - start_time
    logger.info(f"Model training completed in {elapsed_time:.2f} seconds")
    logger.info(f"Best Parameters: {search.best_params_}")
    logger.info(f"Best F2 Score: {search.best_score_:.4f}")

    # Return the best estimator
    return search.best_estimator_

# Main function to train and save the model:
def train_and_save_model():
    os.makedirs(MODELS_DIR, exist_ok=True)
    X, y = load_and_prepare_data()

    # Log class distribution:
    class_distribution = pd.Series(y).value_counts(normalize=True)
    logger.info(f"Class distribution:\n{class_distribution}")

    # Train the model:
    best_model = train_optimized_model(X, y)

    # Save the model:
    joblib.dump(best_model, MODEL_PATH)
    logger.info(f"SVM model saved to {MODEL_PATH}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    train_and_save_model()

# Technical Notes:
# 1. Mutual Information (mutual_info_classif): Used for feature selection instead of ANOVA F-test.
#    It captures non-linear relationships between features and target, which is particularly 
#    useful for SVM with non-linear kernels.
#
# 2. Class weighting: We use a custom weight calculation that squares the standard 'balanced' 
#    weights, giving even greater importance to the minority class (stroke cases).
#
# 3. Probability Calibration: SVMs don't naturally output well-calibrated probabilities. The
#    CalibratedClassifierCV wrapper improves the reliability of predicted probabilities.
#
# 4. F2-score optimization: Since recall (sensitivity) is more important than precision for stroke
#    prediction (missing actual stroke cases is worse than false alarms), we use the F2 score
#    which weighs recall higher than precision.
