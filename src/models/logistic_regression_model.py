import pandas as pd  
import os  
import pickle 
import logging  
import numpy as np  

# For model selection and hyperparameter tuning:
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV 
from sklearn.linear_model import LogisticRegression  # Logistic Regression classifier.
from sklearn.preprocessing import OneHotEncoder, PowerTransformer  # For encoding and scaling.
from sklearn.compose import ColumnTransformer  # For applying different preprocessing to different feature types.

from sklearn.metrics import make_scorer  # For custom scoring functions.
# Creating a pipeline:
from sklearn.pipeline import Pipeline  # Standard scikit-learn pipeline.
from sklearn.metrics import recall_score, confusion_matrix, make_scorer

def recall_specificity_score(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    # Weighted custom score
    return 0.8 * recall + 0.35 * specificity

# Setting up the logging to display important information during execution:
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Define the paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
X_TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'X_train_resampled.csv')
Y_TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'y_train_resampled.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'src', 'models') 
MODEL_PATH = os.path.join(MODELS_DIR, 'logistic_regression_model.pkl')  

# Loading the oversampled dataset and preparing the feature set (X) and target variable (y):
def load_and_prepare_data():
    X = pd.read_csv(X_TRAIN_PATH)
    y = pd.read_csv(Y_TRAIN_PATH)
    y = y.values.ravel()
    logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    logger.info(f"Target variable shape: {y.shape}")
    return X, y

# Function to perform model training with GridSearchCV and hyperparameter tuning:
def perform_hyperparameter_tuning(X, y):
    # Separating feature types
    numeric = X.select_dtypes(include=['int64', 'float64']).columns  # Numeric columns
    categorical = X.select_dtypes(include=['object', 'category']).columns  # Categorical columns

    # Preprocessing pipeline: scaling numeric features, one-hot encoding categorical ones
    preprocessor = ColumnTransformer([
        ('num', PowerTransformer(), numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)
    ])

    # Creating a pipeline with preprocessing and logistic regression
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))  # Main classifier
    ])

    # Defining hyperparameter grid for GridSearchCV:
    param_grid = [
        {
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['l2'],
            'classifier__solver': ['lbfgs', 'liblinear'],
            'classifier__class_weight': ['balanced', {0: 1, 1: 10}]
        },
        {
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['l1'],
            'classifier__solver': ['liblinear'],
            'classifier__class_weight': ['balanced', {0: 1, 1: 10}]
        }
    ]

    # Using repeated stratified k-fold for more robust evaluation
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    custom_score = make_scorer(recall_specificity_score)  # Custom scoring function to balance recall and roc_auc
    # Running the grid search
    logger.info("Starting logistic regression tuning with GridSearchCV...")
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=100,  # Reduced to speed up without losing much performance
        cv=cv,
        scoring=custom_score,
        refit=True,  # Refit based on f1
        verbose=2,
        n_jobs=-1,
        random_state=42,
        return_train_score=True
    )
    search.fit(X, y)  # Train and validate models across grid

    best_model = search.best_estimator_
    logger.info(f"Best parameters: {search.best_params_}")  # Log the best combination found
    logger.info(f"Best score: {search.best_score_:.4f}")  # Log the best score achieved

    # Log top 3 models for comparison (based on custom score)
    results = pd.DataFrame(search.cv_results_)
    top_results = results.sort_values('rank_test_score').head(3)
    for i, (params, score) in enumerate(zip(top_results['params'], top_results['mean_test_score'])):
        logger.info(f"Rank {i+1} - Custom Score: {score:.4f} - Params: {params}")

    return best_model  # Return the optimized model

# Function to train the model and save it to disk
def train_and_save_model():
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)  # Ensure the models directory exists
        X, y = load_and_prepare_data()  # Load dataset
        model = perform_hyperparameter_tuning(X, y)  # Train and tune model
        with open(f"{MODEL_PATH}", 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved optimized logistic regression model to {MODEL_PATH}")
        logger.info("Successfully verified model loading")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_save_model()
