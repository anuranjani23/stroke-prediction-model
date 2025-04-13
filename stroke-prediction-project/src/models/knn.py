import pandas as pd
import os
import pickle
import logging
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectPercentile, f_classif, mutual_info_classif
from sklearn.metrics import recall_score, confusion_matrix, make_scorer

def recall_specificity_score(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    # Weighted custom score
    return 0.95 * recall + 0.001 * specificity
# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
X_TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'X_train_resampled.csv')
Y_TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'y_train_resampled.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'src', 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'knn_model.pkl')

def load_and_prepare_data():
    X = pd.read_csv(X_TRAIN_PATH)
    y = pd.read_csv(Y_TRAIN_PATH)
    y = y.values.ravel()
    logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    logger.info(f"Target variable shape: {y.shape}")
    return X, y


def perform_hyperparameter_tuning(X, y):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    # Using both standard and robust scalers in the search space
    numeric_transformer = Pipeline(steps=[
        ('scaler', RobustScaler())  # RobustScaler handles outliers better
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Adding multiple feature selection techniques
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectPercentile(score_func=mutual_info_classif)),  # Changed to mutual_info for better feature selection
        ('knn', KNeighborsClassifier())
    ])
    
    # Expanded parameter grid with focus on parameters that improve recall
    param_dist = {
        'feature_selection__percentile': [70, 80, 90, 95, 100],  # Higher percentages to keep more potentially useful features
        'feature_selection__score_func': [f_classif, mutual_info_classif],  # Try different scoring functions
        'knn__n_neighbors': [3, 5, 7, 9, 11, 13],  # Focus on smaller neighborhoods
        'knn__weights': ['uniform', 'distance'],  # Distance weighing can improve recall
        'knn__metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
        'knn__p': [1, 2, 3],  # Parameter for Minkowski metric
        'knn__leaf_size': [10, 20, 30, 40, 50]  # For improving algorithm efficiency
    }
    
    cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
    custom_score = make_scorer(recall_specificity_score)
    
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=90,  # Increased iterations for better exploration
        cv=cv,
        scoring=custom_score,
        refit=True,  
        n_jobs=-1,
        verbose=2,
        random_state=42,
        return_train_score=True
    )
    
    search.fit(X, y)
    
    best_model = search.best_estimator_
    logger.info(f"Best parameters: {search.best_params_}")  # Log the best combination found
    logger.info(f"Best score: {search.best_score_:.4f}")  # Log the best score achieved

    # Log top 3 models for comparison (based on custom score)
    results = pd.DataFrame(search.cv_results_)
    top_results = results.sort_values('rank_test_score').head(3)
    for i, (params, score) in enumerate(zip(top_results['params'], top_results['mean_test_score'])):
        logger.info(f"Rank {i+1} - Custom Score: {score:.4f} - Params: {params}")

    return search.best_estimator_

def train_and_save_model():
    os.makedirs(MODELS_DIR, exist_ok=True)
    X, y = load_and_prepare_data()
    
    logger.info("Starting hyperparameter tuning...")
    best_model = perform_hyperparameter_tuning(X, y)
    
    # Save the model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)
    logger.info(f"Optimized KNN model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save_model()