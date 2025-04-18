import pandas as pd
import os
import pickle
import logging
import time

# For cross-validation and hyperparameter tuning:
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier

# For data preprocessing:
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from imblearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest, f_classif


# Setting up the logging to display important information during execution:
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Define the paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
X_TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'X_train_resampled.csv')
Y_TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'y_train_resampled.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'src', 'models') 
MODEL_PATH = os.path.join(MODELS_DIR, 'ann_model.pkl')  


# Cache to store preprocessed data
_cached_data = None
def load_and_prepare_data(use_cache=False):
    global _cached_data
    if use_cache and _cached_data is not None:
        logger.info("Using cached data.")
        return _cached_data

    X = pd.read_csv(X_TRAIN_PATH)
    y = pd.read_csv(Y_TRAIN_PATH)
    y = y.values.ravel()
    logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    logger.info(f"Target variable shape: {y.shape}")

    # Cache the data
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
        ('ann', MLPClassifier(random_state=42, max_iter=2000, early_stopping=True))
    ])
    
    # Parameter grid including feature selection and ANN hyperparameters
    param_dist = {
        'feature_selection__k': [5, 10, 12, 15, 'all'],
        'ann__hidden_layer_sizes': [(100,), (256,), (100, 50), (64, 64)],
        'ann__activation': ['relu', 'tanh'],
        'ann__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
        'ann__learning_rate': ['adaptive', 'constant', 'invscaling']
    }
    
    # Use RandomizedSearchCV instead of GridSearchCV for faster training:
    cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, 
        n_iter=40,  
        cv=cv, scoring=['recall', 'roc_auc', 'f1'],
        refit='f1',
        n_jobs=-1,  
        verbose=1
    )
    
    logger.info("Training ANN model with RandomizedSearchCV...")
    search.fit(X, y)
    
    
    # Calculate and log training time
    training_time = time.time() - start_time
    logger.info(f"Model training completed in {training_time:.2f} seconds")
    
    # Now retrain on full dataset with best parameters
    logger.info("Retraining best model on full dataset...")
    
    logger.info(f"Best Parameters: {search.best_params_}")
    logger.info(f"Best Score: {search.best_score_:.4f}")
    
    # Log top 3 models by ROC AUC
    results = pd.DataFrame(search.cv_results_)
    # Log top 3 models by Average Precision
    top_ap = results.sort_values('rank_test_f1').head(3)
    for i, (params, score) in enumerate(zip(top_ap['params'], top_ap['mean_test_f1'])):
        logger.info(f"[f1] Rank {i+1} - Score: {score:.4f} - Params: {params}")
    # Return the best pipeline already trained on full data
    return search.best_estimator_

def train_and_save_model():
    start_time = time.time()
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    X, y = load_and_prepare_data(use_cache=True)
    
    # Train and save ANN model
    best_ann_model = train_ann(X, y)
    with open(f"{MODEL_PATH}", 'wb') as f:
        pickle.dump(best_ann_model, f)
    logger.info(f"ANN model saved to {MODEL_PATH}")
    logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    train_and_save_model()
