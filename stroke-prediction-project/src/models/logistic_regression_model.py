import pandas as pd  
import os  
import joblib  
import logging  
import numpy as np  

# For model selection and hyperparameter tuning:
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV  
from sklearn.linear_model import LogisticRegression  # Logistic Regression classifier.
from sklearn.preprocessing import OneHotEncoder, PowerTransformer  # For encoding and scaling.
from sklearn.compose import ColumnTransformer  # For applying different preprocessing to different feature types.
from sklearn.metrics import make_scorer, recall_score, precision_score  

# Handling class imbalance and creating a pipeline that supports imbalanced learning:
from imblearn.combine import SMOTETomek  # SMOTE + Tomek Links for resampling.
from imblearn.pipeline import Pipeline as ImbPipeline  # Pipeline compatible with imbalanced-learn.

# Setting up logging configuration:
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)  

# Defining paths for dataset and saving models:
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_dataset.csv')  
MODELS_DIR = os.path.join(BASE_DIR, 'src', 'models')  
MODEL_PATH = os.path.join(MODELS_DIR, 'logistic_regression_model.pkl')  

# Function to load dataset and split it into features (X) and target (y):
def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)  
    X = df.drop(columns=['stroke'])  
    y = df['stroke']  
    logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    return X, y


# Function to perform model training with GridSearchCV and hyperparameter tuning:
def perform_hyperparameter_tuning(X, y):
    # Separating feature types
    numeric = X.select_dtypes(include=['int64', 'float64']).columns  # Numeric columns
    categorical = X.select_dtypes(include=['object', 'category']).columns  # Categorical columns

    # Preprocessing pipeline: scaling numeric features, one-hot encoding categorical ones
    preprocessor = ColumnTransformer([
        ('num', PowerTransformer(), numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
    ])

    # Creating a pipeline with preprocessing, SMOTETomek resampling, and logistic regression
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('sampling', SMOTETomek(random_state=42)),  # Resample data to handle class imbalance
        ('classifier', LogisticRegression(max_iter=6000, random_state=42, tol=1e-5))  # Main classifier
    ])

    # Defining hyperparameter grid for GridSearchCV:
    param_grid = [
        {
            'classifier__C': np.logspace(-3, 3, 4),  # Inverse regularization strength
            'classifier__penalty': ['l2'],  # Regularization type
            'classifier__solver': ['lbfgs', 'liblinear'],  # Solvers that support L2
            'classifier__class_weight': ['balanced', {0: 1, 1: 10}]  # Class weight options
        },
        {
            'classifier__C': np.logspace(-3, 3, 4),
            'classifier__penalty': ['l1'],  # L1 penalty requires liblinear solver
            'classifier__solver': ['liblinear'],
            'classifier__class_weight': ['balanced', {0: 1, 1: 10}]
        }
    ]

    # Using repeated stratified k-fold for more robust evaluation
    cv = RepeatedStratifiedKFold(n_splits=7, n_repeats=3, random_state=42)

    # Running the grid search
    logger.info("Starting logistic regression tuning with GridSearchCV...")
    search = GridSearchCV(pipeline, param_grid, scoring='f1', cv=cv, n_jobs=-1, verbose=1)
    search.fit(X, y)  # Train and validate models across grid

    best_model = search.best_estimator_
    logger.info(f"Best parameters: {search.best_params_}")  # Log the best combination found

    return best_model  # Return the optimized model

# Function to train the model and save it to disk
def train_and_save_model():
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)  # Ensure the models directory exists
        X, y = load_and_prepare_data()  # Load dataset
        model = perform_hyperparameter_tuning(X, y)  # Train and tune model
        joblib.dump(model, MODEL_PATH)  # Save the trained model
        logger.info(f"Saved optimized logistic regression model to {MODEL_PATH}")
        loaded_model = joblib.load(MODEL_PATH)  # Load back to confirm it was saved correctly
        logger.info("Successfully verified model loading")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

# Function to load the saved model for inference or further evaluation:
def load_model():
    try:
        model = joblib.load(MODEL_PATH)  # Load the saved model
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_save_model()

# SMOTETomek is a combination of SMOTE (oversampling the minority class) and Tomek Links (undersampling the majority class by removing borderline examples). Together, they balance the dataset more effectively for imbalanced classification problems.
# PowerTransformer is a preprocessing method that applies a power transformation to make data more Gaussian-like. This can improve the performance of linear models like logistic regression by stabilizing variance and minimizing skewness.