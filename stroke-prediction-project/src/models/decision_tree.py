import os
import pandas as pd
import joblib
import logging

# For model training and evaluation
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# For feature processing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.compose import ColumnTransformer

# For handling class imbalance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Define file paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_dataset.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'src', 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'decision_tree_model.pkl')


def load_and_prepare_data():
    """
    Load the dataset and separate features from target variable.
    
    Returns:
        tuple: (X, y) where X contains features and y contains target labels
    """
    logger.info("Loading data from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    
    # Separate features and target
    X = df.drop(columns=['stroke'])  # Features (all columns except 'stroke')
    y = df['stroke']                 # Target variable (stroke prediction)
    
    logger.info("Data loaded: %d samples, %d features", X.shape[0], X.shape[1])
    return X, y


# Function to build and train decision tree classifier with hyperparameter tuning:
def train_decision_tree_model(X, y):
    
    logger.info("Starting model training process")
    
    # Identify column types for preprocessing:
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    # Create preprocessing pipeline for different data types:
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),  # Scaling of numerical features.
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # Encoding of categorical features.
        ])
    
    # Creating pipeline with preprocessing, SMOTE for class imbalance, feature selection, and classifier:
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),  # Handling class imbalance.
        ('feature_selection', SelectKBest(mutual_info_classif)),  # Selecting the most informative features, (ANOVA F-test).
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])
    
    # Define hyperparameter grid to search:
    param_grid = {
        'feature_selection__k': [5, 8, 9, 10, 12],  # Number of top features to select.
        'classifier__max_depth': [3, 5, 7, 10],  # Maximum depth of the tree.
        'classifier__criterion': ['gini', 'entropy'],  # Split criterion.
        'classifier__min_samples_split': [2, 5, 10]  # Minimum samples required to split a node.
    }
    
    # Initializing the stratified k-fold cross-validation:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Performing grid search with cross-validation:
    grid_search = GridSearchCV(
        pipeline, param_grid, 
        scoring='f1',  # Optimize for F1 score (balances precision and recall).
        cv=cv, 
        n_jobs=-1,  
        verbose=1
    )
    
    # Fit the model:
    logger.info("Training model with grid search (this may take a while)...")
    grid_search.fit(X, y)
    
    logger.info("Best Parameters: %s", grid_search.best_params_)
    logger.info("Best F1 Score: %.4f", grid_search.best_score_)
    
    return grid_search.best_estimator_


def train_and_save_model():
    X, y = load_and_prepare_data()
    
    # Training the model:
    best_model = train_decision_tree_model(X, y)
    joblib.dump(best_model, MODEL_PATH)
    logger.info("Decision Tree model saved to %s", MODEL_PATH)


if __name__ == '__main__':
    train_and_save_model()


# GridSearchCV is a method in scikit-learn which is used for hyperparameter tuning by searching through a grid of parameters, by using cross-validation.
# Mutual information is a feature selection method that measures the mutual dependence between two variables. It helps select features that have strong relationships with the target variable.
# SMOTE or Synthetic Minority Over-sampling Technique helps with class imbalancing by generating synthetic samples for the minority class, preventing the decision tree from being biased toward the majority class.
# Decision trees are prone to overfitting, which is why parameters like max_depth and min_samples_split are important to tune - they control the complexity of the tree and help find the right balance between bias and variance.