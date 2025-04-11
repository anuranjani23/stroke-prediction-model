import pandas as pd
import os
import pickle
import logging
import time
import numpy as np

# For cross-validation and hyperparameter tuning:
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier

# For data preprocessing:
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from imblearn.pipeline import Pipeline

# For handling class imbalance:
from imblearn.over_sampling import SMOTE

from sklearn.feature_selection import SelectKBest, f_classif

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_dataset.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'src', 'models')
ANN_MODEL_PATH = os.path.join(MODELS_DIR, 'ann_model.pkl')

# Cache to store preprocessed data
_cached_data = None

# Function for loading and preparing data with optional caching for multiple runs:
def load_and_prepare_data(use_cache=True):
    global _cached_data
    
    # Return cached data if available:
    if use_cache and _cached_data is not None:
        logger.info("Using cached data")
        return _cached_data
    
    start_time = time.time()
    logger.info("Loading dataset...")
    
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['stroke'])
    y = df['stroke']
    
    logger.info(f"Data loaded in {time.time() - start_time:.2f} seconds")
    
    if use_cache:
        _cached_data = (X, y)
    
    return X, y

# Creating column transformer for preprocessing:
def create_preprocessor(X):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

def train_ann(X, y):
    start_time = time.time()

    # Create column transformer for preprocessing
    preprocessor = create_preprocessor(X)

    # Create pipeline with preprocessing, feature selection, SMOTE, and ANN:
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(f_classif)),
        ('smote', SMOTE(random_state=42)),
        ('ann', MLPClassifier(random_state=42, max_iter=2000, early_stopping=True))
    ])
    
    # Parameter grid including feature selection and ANN hyperparameters
    param_dist = {
        'feature_selection__k': [5, 10, 12, 15, 'all'],
        'ann__hidden_layer_sizes': [(100,), (256,), (100, 50), (64, 64)],
        'ann__activation': ['relu', 'tanh'],
        'ann__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
        'ann__learning_rate': [0.00001, 0.0001, 0.001, 'adaptive']
    }
    
    # Use RandomizedSearchCV instead of GridSearchCV for faster training:
    cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, 
        n_iter=40,  
        cv=cv, scoring='f1', 
        n_jobs=-1,  
        verbose=1
    )
    
    logger.info("Training ANN model with RandomizedSearchCV...")
    search.fit(X, y)
    
    # Log results
    logger.info(f"ANN - Best Parameters: {search.best_params_}")
    logger.info(f"ANN - Best F1 Score: {search.best_score_:.4f}")
    logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    # Return the best pipeline already trained on full data
    return search.best_estimator_

def train_and_save_model():
    start_time = time.time()
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    X, y = load_and_prepare_data(use_cache=True)
    
    # Train and save ANN model
    best_ann_model = train_ann(X, y)
    with open(f"{ANN_MODEL_PATH}", 'wb') as f:
        pickle.dump(best_ann_model, f)
    logger.info(f"ANN model saved to {ANN_MODEL_PATH}")
    logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    train_and_save_model()
