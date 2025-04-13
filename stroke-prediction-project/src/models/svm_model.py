import pandas as pd
import os
import pickle
import logging
from time import time
import numpy as np

# For model selection and hyperparameter tuning:
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.svm import SVC
# For data preprocessing:
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# For feature selection:
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import recall_score, confusion_matrix, make_scorer
def recall_specificity_score(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    # Weighted custom score
    return 0.95 * recall + 0.1 * specificity
# Setting up the logging to display important information during execution:
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Define the paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
X_TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'X_train_resampled.csv')
Y_TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'y_train_resampled.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'src', 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'svm_model.pkl')

# Loading the oversampled dataset and preparing the feature set (X) and target variable (y):
def load_and_prepare_data():
    X = pd.read_csv(X_TRAIN_PATH)
    y = pd.read_csv(Y_TRAIN_PATH)
    y = y.values.ravel()
    logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    logger.info(f"Target variable shape: {y.shape}")
    return X, y

def perform_hyperparameter_tuning(X, y):
    # Identifying column types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    logger.info(f"Numeric features: {list(numeric_features)}")
    logger.info(f"Categorical features: {list(categorical_features)}")

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    # Create pipeline with preprocessing, feature selection, and classifier
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(mutual_info_classif)),
        ('classifier', SVC(probability=True, class_weight='balanced', random_state=42))
    ])

    # Define parameter distribution for RandomizedSearchCV
    param_dist = {
        'feature_selection__k': [10, 15, 20, 25],
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__gamma': ['scale', 'auto', 0.01, 0.1],
        'classifier__kernel': ['rbf', 'poly', 'sigmoid'],
        'classifier__degree': [2, 3, 4],  # For poly kernel
        'classifier__coef0': [-1, 0, 1]  # For poly/sigmoid kernels
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    start_time = time()
    logger.info("Starting hyperparameter optimization...")
    custom_scorer = make_scorer(recall_specificity_score, greater_is_better=True)

    # Perform randomized search
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=100,  # More iterations for better exploration
        cv=cv,
        scoring=custom_scorer,
        refit=True,  # Refit based on recall
        verbose=2,
        n_jobs=-1,
        random_state=42,
        return_train_score=True
    )
    search.fit(X, y)
    
    logger.info(f"Best Parameters: {search.best_params_}")
    logger.info(f"Best Score: {search.best_score_:.4f}")
    
    # Log top 3 models for comparison - FIX: use 'mean_test_recall' since we're refitting based on recall
    results = pd.DataFrame(search.cv_results_)
    # Sort by mean_test_recall which is available (instead of rank_test_score which doesn't exist)
    top_results = results.sort_values('mean_test_score', ascending=False).head(3)
    for i, (params, score) in enumerate(zip(top_results['params'], top_results['mean_test_score'])):
        logger.info(f"Rank {i+1} - Score: {score:.4f} - Params: {params}")

    return search.best_estimator_

def train_and_save_model():
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        X, y = load_and_prepare_data()
        
        logger.info("Starting model training...")
        best_model = perform_hyperparameter_tuning(X, y)
        
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(best_model, f)
        logger.info(f"Optimized SVM model saved to {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_save_model()

# Technical Notes:
# 1. Feature Selection: Using mutual_info_classif which captures non-linear relationships
#    between features and target, particularly useful for SVM with non-linear kernels.
#
# 2. Class Weighting: Using 'balanced' mode to automatically adjust weights inversely
#    proportional to class frequencies.
#
# 3. Custom Scoring: The scoring function emphasizes recall (70% weight) to minimize
#    false negatives in stroke detection while still considering precision and F1 score.
#
# 4. Kernel Options: Exploring rbf, poly and sigmoid kernels with appropriate parameters
#    for each kernel type.
#
# 5. Randomized Search: More efficient than grid search for exploring large parameter spaces,
#    with 50 iterations for thorough exploration.