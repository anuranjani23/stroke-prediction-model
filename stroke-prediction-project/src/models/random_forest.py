import pandas as pd
import numpy as np
import os
import pickle
from time import time
import logging
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE, VarianceThreshold, f_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, make_scorer, recall_score, f1_score, precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, OneHotEncoder

# Setting up the logging to display important information during execution:
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Define the paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
X_TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'X_train_resampled.csv')
Y_TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'y_train_resampled.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'src', 'models') 
MODEL_PATH = os.path.join(MODELS_DIR, 'rf_model.pkl')  


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

def get_feature_selector(method='mutual_info', n_features=20):
    if method == 'mutual_info':
        return SelectKBest(mutual_info_classif, k=n_features)
    elif method == 'rfe':
        base_estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        return RFE(estimator=base_estimator, n_features_to_select=n_features)
    elif method == 'pca':
        return PCA(n_components=n_features)
    elif method == 'variance':
        return VarianceThreshold(threshold=0.01)
    else:
        logger.warning(f"Unknown feature selection method: {method}. Using mutual_info instead.")
        return SelectKBest(mutual_info_classif, k=n_features)

def extract_feature_importances(model, feature_names):
    try:
        if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            rf = model.named_steps['classifier']
            preprocessor = model.named_steps['preprocessor']
            if 'feature_selection' in model.named_steps:
                feature_selector = model.named_steps['feature_selection']
                if hasattr(feature_selector, 'get_support'):
                    mask = feature_selector.get_support()
                    if hasattr(preprocessor, 'get_feature_names_out'):
                        all_features = preprocessor.get_feature_names_out()
                        selected_features = all_features[mask]
                    else:
                        selected_features = np.array(feature_names)[mask]
                else:
                    selected_features = preprocessor.get_feature_names_out() if hasattr(preprocessor, 'get_feature_names_out') else feature_names
            else:
                selected_features = preprocessor.get_feature_names_out() if hasattr(preprocessor, 'get_feature_names_out') else feature_names

            importances = rf.feature_importances_
            if len(importances) == len(selected_features):
                feature_importances = list(zip(selected_features, importances))
                return sorted(feature_importances, key=lambda x: x[1], reverse=True)
            else:
                logger.warning("Mismatch between importances and selected features")
                return None
        else:
            logger.warning("Classifier not found in pipeline")
            return None
    except Exception as e:
        logger.error(f"Error extracting feature importances: {str(e)}")
        return None

def perform_hyperparameter_tuning(X, y):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    logger.info(f"Numeric features: {list(numeric_features)}")
    logger.info(f"Categorical features: {list(categorical_features)}")

    numeric_transformer = Pipeline([('scaler', PowerTransformer(method='yeo-johnson'))])
    categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(mutual_info_classif)),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])

    param_dist = {
        'feature_selection__k': [10, 20, 30, 'all'],
        'feature_selection__score_func': [mutual_info_classif, f_classif],
        'classifier__n_estimators': [100, 200, 300, 500],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2', None],
        'classifier__bootstrap': [True, False],
        'classifier__criterion': ['gini', 'entropy']
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    start_time = time()
    logger.info("Starting hyperparameter optimization...")

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        cv=cv,
        scoring=['recall', 'roc_auc', 'f1'],
        refit='recall',
        n_jobs=-1,
        n_iter=50,
        verbose=2,
        random_state=42,
        return_train_score=True
    )
    random_search.fit(X, y)

    logger.info(f"Random search completed in {time() - start_time:.2f} seconds")
    logger.info(f"Best Parameters: {random_search.best_params_}")
    logger.info(f"Best Score: {random_search.best_score_:.4f}")

    # Log top 3 models for comparison
    results = pd.DataFrame(random_search.cv_results_)
    top_results = results.sort_values('rank_test_recall').head(3)
    for i, (params, score) in enumerate(zip(top_results['params'], top_results['mean_test_recall'])):
        logger.info(f"Rank {i+1} - Score: {score:.4f} - Params: {params}")

    return random_search.best_estimator_

def train_and_save_model():
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        X, y = load_and_prepare_data()
        
        logger.info("Starting model training...")
        best_model = perform_hyperparameter_tuning(X, y)
        
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(best_model, f)
        logger.info(f"Optimized Random Forest model saved to {MODEL_PATH}")
        
        # Log feature importances if available
        feature_names = list(X.columns)
        importances = extract_feature_importances(best_model, feature_names)
        if importances:
            logger.info("Top 10 most important features:")
            for feature, importance in importances[:10]:
                logger.info(f"{feature}: {importance:.4f}")
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_save_model()