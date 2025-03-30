import pandas as pd
import os
# For saving and loading the trained models:
import joblib
import logging

# For the cross-validation and hyperparameter tuning of the model:
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB
# For the data scaling and encoding of the categorical features in dataset:
from sklearn.preprocessing import OneHotEncoder
# For preprocessing the pipeline for different data types:
from sklearn.compose import ColumnTransformer
# Using the Synthetic Minority Over-sampling Technique (SMOTE) to handle the class imbalance:
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # Pipeline that supports imbalanced learning methods
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Setting up the logging to display important information during execution:
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_dataset.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'src', 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'bayesian_model.pkl')


# Loading the dataset and preparing the feature set (X) and target variable (y):
def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['stroke'])  # Features (all the columns except 'stroke').
    y = df['stroke']  # Target variable (stroke prediction).
    return X, y


# Function for Hyperparameter Tuning using GridSearchCV:
def perform_hyperparameter_tuning(X, y):
    # Identifying the column types for preprocessing:
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    # Applying different preprocessing techniques (Scaling and One-hot encoding respectively) to the numeric and categorical features:
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Using ImbPipeline to incorporate SMOTE for handling class imbalance:
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(sampling_strategy=0.5, random_state=42)),  # SMOTE to oversample the minority class (stroke) during training
        ('feature_selection', SelectKBest(mutual_info_classif)),
        ('gnb', GaussianNB())
    ])

    # Grid definition for Gaussian Naive Bayes:
    param_grid = {
        'feature_selection__k': [5, 8, 10],  # Removed 'all' as it's invalid for SelectKBest; specifies number of features
        'gnb__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]  # Smoothing parameter for handling zero variance
    }

    # 5-fold stratified cross-validation:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1  # Using the ROC AUC as an evaluation metric.
    )
    # Fit the model using grid search:
    grid_search.fit(X, y)

    # Logging the best parameters and the performance:
    logger.info(f"Best Parameters: {grid_search.best_params_}")
    logger.info(f"Best f1 Score: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


def train_and_save_model():
    os.makedirs(MODELS_DIR, exist_ok=True)
    X, y = load_and_prepare_data()
    best_model = perform_hyperparameter_tuning(X, y)
    joblib.dump(best_model, MODEL_PATH)
    logger.info(f"Bayesian model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()


# Gaussian Naive Bayes (GNB) is a probabilistic classifier based on Bayes' theorem with strong (naive) independence assumptions
# between the features. It assumes that all features are normally distributed, which is where the "Gaussian" part comes from.

# var_smoothing is a stability parameter for GNB that adds a small amount of variance to all features to prevent division by zero,
# especially when a feature has zero variance. This can improve model performance by handling rare events more gracefully.

# Unlike KNN, Gaussian Naive Bayes is a parametric model that makes assumptions about the underlying data distribution.
# It works well with small datasets and is computationally efficient, making it suitable for real-time predictions.

# The model computes the probability of each class and the conditional probability of each feature given each class.
# Classification is done by selecting the class with the highest posterior probability.