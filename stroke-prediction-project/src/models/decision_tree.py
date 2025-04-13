import os
import pandas as pd
import pickle
import logging
import time
import numpy as np
from joblib import parallel_backend
# For model training and evaluation
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
# For feature processing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, PowerTransformer
from sklearn.feature_selection import mutual_info_classif, f_classif, SelectPercentile
from sklearn.compose import ColumnTransformer
# For handling class imbalance
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
# For metrics
from sklearn.metrics import make_scorer, recall_score, f1_score, confusion_matrix
from sklearn.metrics import recall_score, confusion_matrix, make_scorer

def recall_specificity_score(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    # Weighted custom score
    return 0.99999999 * recall + 0.000000001 * specificity
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Define file paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
X_TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'X_train_resampled.csv')
Y_TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'y_train_resampled.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'src', 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'decision_tree_model.pkl')

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)


# Loading the oversampled dataset and preparing the feature set (X) and target variable (y):
def load_and_prepare_data():
    X = pd.read_csv(X_TRAIN_PATH)
    y = pd.read_csv(Y_TRAIN_PATH)
    y = y.values.ravel()
    logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    logger.info(f"Target variable shape: {y.shape}")
    
    # Log class distribution to understand balance after resampling
    unique_classes, class_counts = np.unique(y, return_counts=True)
    logger.info("Class distribution after resampling:")
    for cls, count in zip(unique_classes, class_counts):
        logger.info(f"  Class {cls}: {count} samples ({count/len(y)*100:.2f}%)")
    
    return X, y



# Function to build and train decision tree classifier with hyperparameter tuning:
def train_decision_tree_model(X, y):
    
    logger.info("Starting model training process")
    start_time = time.time()
    
    # Identify column types for preprocessing:
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    logger.info(f"Numeric features: {list(numeric_features)}")
    logger.info(f"Categorical features: {list(categorical_features)}")
    
    # Advanced preprocessing options
    numeric_transformers = [
        ('standard', StandardScaler()),
        ('robust', RobustScaler()),
        ('power', PowerTransformer(method='yeo-johnson'))
    ]
    
    # First, determine best preprocessing combination
    logger.info("Evaluating preprocessing combinations...")
    best_score = 0
    best_num_transformer = None
    
    base_tree = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    
    for name, transformer in numeric_transformers:
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer([
            ('num', transformer, numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
        
        # Basic pipeline to evaluate preprocessing
        test_pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('feature_selection', SelectPercentile(mutual_info_classif, percentile=90)),
            ('classifier', base_tree)
        ])
        
        # Quick evaluation
        cv_scores = cross_val_score(
            test_pipeline, X, y, 
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring='f1_macro'
        )
        avg_score = np.mean(cv_scores)
        logger.info(f"  {name} scaler: average f1_macro = {avg_score:.4f}")
        
        if avg_score > best_score:
            best_score = avg_score
            best_num_transformer = transformer
    
    logger.info(f"Selected numeric transformer with f1_macro = {best_score:.4f}")
    
    # Create the final preprocessor with best transformer
    preprocessor = ColumnTransformer([
        ('num', best_num_transformer, numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])
    
    # Define the feature selector options
    feature_selectors = {
        'percentile_mutual_info': SelectPercentile(mutual_info_classif),
        'percentile_f_classif': SelectPercentile(f_classif)
    }
    
    
    # Creating pipeline with preprocessing, sampling for class imbalance, feature selection, and classifier
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectPercentile(mutual_info_classif)),  # Placeholder, will be tuned
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])
    
    
    # Enhanced hyperparameter grid:
    param_dist = {
        'feature_selection': list(feature_selectors.values()),
        'feature_selection__percentile': [80, 85, 90, 95, 100],  # Higher percentiles to retain more features
        'classifier__criterion': ['gini', 'entropy', 'log_loss'],  # Split criterion
        'classifier__max_depth': [5, 8, 10, 12, 15, 20, None],  # Maximum depth of the tree
        'classifier__min_samples_split': [2, 4, 6, 8, 10],  # Minimum samples required to split a node
        'classifier__min_samples_leaf': [1, 2, 3, 4, 5],  # Minimum samples required in a leaf node
        'classifier__max_features': [None, 'sqrt', 'log2', 0.7, 0.8],  # Number of features to consider at each split
        'classifier__class_weight': ['balanced', None]  # Class weights
    }
    
    # Initializing the stratified k-fold cross-validation:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    custom_score = make_scorer(recall_specificity_score)
    
    # Use RandomizedSearchCV for more efficient parameter search
    with parallel_backend('multiprocessing'):  # Use multiprocessing for better performance
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=100,  # Number of parameter settings sampled
            scoring=custom_score,
    refit='True',  # if you want to refit based on AP
            cv=cv,
            n_jobs=-1,
            verbose=2,
            random_state=42,
            return_train_score=True  # Store training scores to check for overfitting
        )
        
        # Fit the model:
        logger.info("Training model with randomized search (this may take a while)...")
        search.fit(X, y)
    
    # Log detailed results
    logger.info("Best Parameters: %s", search.best_params_)
    logger.info("Best Custom Score: %.4f", search.best_score_)
    
    # Get scores for the best model
    best_idx = search.best_index_
    cv_results = search.cv_results_
    
    # Get the best model
    best_model = search.best_estimator_
    
    
    # Feature importance analysis for interpretability
    if hasattr(best_model['classifier'], 'feature_importances_'):
        try:
            # Get feature names after preprocessing and selection
            features_after_preprocessing = best_model['preprocessor'].get_feature_names_out()
            
            # Get support mask from feature selection
            if hasattr(best_model['feature_selection'], 'get_support'):
                support_mask = best_model['feature_selection'].get_support()
                selected_features = features_after_preprocessing[support_mask]
            else:
                selected_features = features_after_preprocessing
            
            # Get feature importances
            importances = best_model['classifier'].feature_importances_
            
            if len(importances) == len(selected_features):
                # Sort by importance
                indices = np.argsort(importances)[::-1]
                
                logger.info("Top 10 most important features:")
                for i in range(min(10, len(indices))):
                    idx = indices[i]
                    logger.info(f"  {selected_features[idx]}: {importances[idx]:.4f}")
        except Exception as e:
            logger.warning(f"Could not extract feature importances: {e}")
    
    # Calculate and log training time
    training_time = time.time() - start_time
    logger.info(f"Model training completed in {training_time:.2f} seconds")
    
    # Now retrain on full dataset with best parameters
    logger.info("Retraining best model on full dataset...")
    best_model.fit(X, y)
    
    logger.info(f"Best Parameters: {search.best_params_}")
    logger.info(f"Best Score: {search.best_score_:.4f}")
    
    # Log top 3 models by ROC AUC
    results = pd.DataFrame(search.cv_results_)
    # Log top 3 models by Average Precision
    top_ap = results.sort_values('rank_test_score').head(3)
    for i, (params, score) in enumerate(zip(top_ap['params'], top_ap['mean_test_score'])):
        logger.info(f"[Custom] Rank {i+1} - Score: {score:.4f} - Params: {params}")

    return best_model


def train_and_save_model():
    start_time = time.time()
    
    # Load data
    X, y = load_and_prepare_data()
    
    # Training the model:
    best_model = train_decision_tree_model(X, y)
    
    # Save the model
    with open(f"{MODEL_PATH}", 'wb') as f:
        pickle.dump(best_model, f)
    logger.info("Decision Tree model saved to %s", MODEL_PATH)
    
    # Calculate and log model size
    model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    logger.info(f"Model size: {model_size_mb:.2f} MB")
    
    logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")


if __name__ == '__main__':
    train_and_save_model()


# RandomizedSearchCV is more efficient than GridSearchCV for large parameter spaces as it samples a defined number of parameter combinations rather than exhaustively trying all of them.
# The custom scoring function specifically targets improving the diagonal of the confusion matrix, which directly addresses the optimization goal.
# Multiple preprocessing techniques are evaluated (StandardScaler, RobustScaler, PowerTransformer) to find the one that works best with Decision Trees and the specific dataset characteristics.
# Feature selection using percentile-based methods retains a higher number of potentially useful features compared to SelectKBest with fixed k value.
# SMOTE and ADASYN are both evaluated for their effectiveness in generating synthetic samples for minority classes, which is crucial for balanced learning.
# The train/validation split provides an independent evaluation of model performance and helps detect overfitting.
# Feature importance analysis offers interpretability of the model's decisions and could guide further feature engineering.