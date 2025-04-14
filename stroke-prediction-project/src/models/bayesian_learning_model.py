import pandas as pd
import os
import pickle
import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer, RobustScaler
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif, SelectPercentile, f_classif
from sklearn.metrics import recall_score, confusion_matrix, make_scorer

def recall_specificity_score(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    # Weighted custom score
    return 0.9999999999 * recall + 0.00000001 * specificity
# Setting up the logging to display important information during execution:
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Define the paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
X_TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'X_train_resampled.csv')
Y_TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'y_train_resampled.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'src', 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'bayesian_model.pkl')

# Loading the oversampled dataset and preparing the feature set (X) and target variable (y):
def load_and_prepare_data():
    X = pd.read_csv(X_TRAIN_PATH)
    y = pd.read_csv(Y_TRAIN_PATH)
    y = y.values.ravel()
    logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    logger.info(f"Target variable shape: {y.shape}")
    return X, y


# Function for Hyperparameter Tuning using RandomizedSearchCV:
def perform_hyperparameter_tuning(X, y):
    """Builds pipeline, performs randomized search with stratified CV, and returns best estimator."""
    
    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    # Custom transformer for numeric data that tries different scalers
    numeric_transformer = Pipeline([
        ('scaler', PowerTransformer(method='yeo-johnson'))  # Better for Gaussian NB as it makes data more normal
    ])
    
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    # Create pipeline with preprocessing, feature selection, and model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectPercentile(score_func=mutual_info_classif)),
        ('gnb', GaussianNB(priors=None))  # Let the model learn class priors from data
    ])
    
    # Expanded parameter grid
    param_dist = {
        'feature_selection__percentile': [70, 75, 80, 85, 90, 95],  # Higher percentile keeps more features
        'feature_selection__score_func': [mutual_info_classif, f_classif],  # Try different scoring functions
        'gnb__var_smoothing': [1e-12, 1e-11, 1e-10], # More extensive smoothing values
        'preprocessor__num__scaler': [StandardScaler(), PowerTransformer(method='yeo-johnson'), RobustScaler()]
    }
    
    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
    custom_score = make_scorer(recall_specificity_score)  # Custom scoring function to balance recall and specificity
    # Perform randomized search
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=100,  # More iterations for better exploration
        cv=cv,
        scoring=custom_score,
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

# Function for training and saving the model:
def train_and_save_model():
    """Trains the model using randomized search and saves the best estimator."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    logger.info("Loading and preparing data...")
    X, y = load_and_prepare_data()
    
    logger.info("Starting hyperparameter tuning...")
    best_model = perform_hyperparameter_tuning(X, y)
    
    # Save the model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)
    logger.info(f"Optimized Bayesian model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save_model()