import pandas as pd  
import os  
import joblib  
import logging  
import numpy as np  
# For statistical transformations (e.g., skewness):
from scipy import stats  

# For cross-validation and hyperparameter tuning:
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV  
from sklearn.linear_model import LogisticRegression  
# For feature scaling and encoding:
from sklearn.preprocessing import OneHotEncoder, PowerTransformer 
# To combine transformers for numeric and categorical features
from sklearn.compose import ColumnTransformer  
from sklearn.metrics import make_scorer, recall_score, precision_score 
# For handling imbalanced classes:
from imblearn.combine import SMOTETomek  
# Pipeline supporting imbalanced-learn transformers:
from imblearn.pipeline import Pipeline as ImbPipeline  
# Dimensionality reduction:
from sklearn.decomposition import PCA  

# Setting up the logging to track model training:
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Wrap model with custom threshold classifier
class ThresholdClassifier:
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return (self.model.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def fit(self, X, y):
        return self.model.fit(X, y)

# Set important directory paths:
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_dataset.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'src', 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'logistic_regression_model.pkl')

# Loading and splitting the dataset into features (X) and target (y):
def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)  
    X = df.drop(columns=['stroke'])  # Features.
    y = df['stroke']  # Target variable.
    logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    return X, y

def engineer_features(X):
    key_cols = [col for col in ['age', 'bmi', 'avg_glucose_level'] if col in X.columns]  # Important continuous features

    # Add interaction terms (product of feature pairs)
    for i in range(len(key_cols)):
        for j in range(i+1, len(key_cols)):
            X[f"{key_cols[i]}_x_{key_cols[j]}"] = X[key_cols[i]] * X[key_cols[j]]

    # Add squared terms and log-transformed features (if skewed)
    for col in key_cols:
        X[f"{col}_squared"] = X[col] ** 2  # Quadratic feature
        if abs(stats.skew(X[col].dropna())) > 1:  # Check for skewness
            offset = 1 if X[col].min() >= 0 else -X[col].min() + 1  # Offset to avoid log(0)
            X[f"{col}_log"] = np.log(X[col] + offset)  # Log transformation

    logger.info(f"Feature engineering completed. Total features: {X.shape[1]}")
    return X

# Custom scoring function prioritizing recall with minimum acceptable precision
def custom_stroke_score(y_true, y_pred):
    recall = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    precision = precision_score(y_true, y_pred, zero_division=0)  # Positive predictive value
    penalty = (0.4 - precision) * 2 if precision < 0.4 else 0  # Penalize low precision
    return max(0, 0.8 * recall + 0.2 * precision - penalty)  # Weighted score

# Tune hyperparameters using GridSearchCV and build final model with optimal threshold
def perform_hyperparameter_tuning(X, y):
    X = engineer_features(X)  # Add more predictive features
    numeric = X.select_dtypes(include=['int64', 'float64']).columns  # Numerical columns
    categorical = X.select_dtypes(include=['object', 'category']).columns  # Categorical columns

    # Data preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', PowerTransformer(), numeric),  # Normalize numerical features
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)  # One-hot encode categorical features
    ])

    # Full pipeline: preprocessing → resampling → PCA → classifier
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('sampling', SMOTETomek(random_state=42)),  # Address class imbalance
        ('dim_reduction', PCA(random_state=42)),  # Reduce dimensionality
        ('classifier', LogisticRegression(max_iter=5000, random_state=42, tol=1e-4))  # Increased max_iter and adjusted tolerance
    ])

    # Grid of hyperparameters for tuning - Modified to reduce complexity and improve convergence
    param_grid = [
        {
            'dim_reduction__n_components': [0.95, 0.99],
            'classifier__C': np.logspace(-3, 3, 4),  # Reduced number of steps
            'classifier__penalty': ['l2'],  # Focus on l2 which converges better
            'classifier__solver': ['liblinear', 'lbfgs'],  # Better solvers for convergence
            'classifier__class_weight': ['balanced', {0:1, 1:10}]
        },
        {
            'dim_reduction__n_components': [0.95, 0.99],
            'classifier__C': np.logspace(-3, 3, 4),  # Reduced number of steps
            'classifier__penalty': ['l1'],
            'classifier__solver': ['liblinear'],  # liblinear supports l1
            'classifier__class_weight': ['balanced', {0:1, 1:10}]
        }
    ]

    # Cross-validation setup
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)  # Reduced repeats to save computation
    scorer = make_scorer(custom_stroke_score)  # Use custom score

    logger.info("Starting logistic regression tuning with GridSearchCV...")
    search = GridSearchCV(pipeline, param_grid, scoring=scorer, cv=cv, n_jobs=-1, verbose=1)  # Start grid search
    search.fit(X, y)  # Fit models

    best_model = search.best_estimator_  # Retrieve best model
    logger.info(f"Best parameters: {search.best_params_}")

    # Find the optimal classification threshold using F2 score
    y_probs = best_model.predict_proba(X)[:, 1]  # Predicted probabilities for positive class
    thresholds = np.arange(0.1, 0.5, 0.01)  # Try multiple thresholds
    best_threshold = 0.5  # Default threshold
    best_f2 = 0  # Initialize best F2 score

    for t in thresholds:
        preds = (y_probs >= t).astype(int)  # Binarize using threshold
        recall = recall_score(y, preds, zero_division=0)  # Recall
        precision = precision_score(y, preds, zero_division=0)  # Precision
        f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) else 0  # F2 score
        if precision >= 0.4 and f2 > best_f2:  # Only accept threshold if precision acceptable
            best_f2, best_threshold = f2, t

    logger.info(f"Optimal classification threshold: {best_threshold:.2f}")
    
    # Return the ThresholdClassifier instance to be saved
    return ThresholdClassifier(best_model, best_threshold)

# Train model and save to disk
def train_and_save_model():
    os.makedirs(MODELS_DIR, exist_ok=True)  # Ensure model directory exists
    X, y = load_and_prepare_data()  # Load data
    model = perform_hyperparameter_tuning(X, y)  # Train model
    joblib.dump(model, MODEL_PATH)  # Save model directly
    logger.info(f"Saved optimized logistic regression model to {MODEL_PATH}")

# Main script entrypoint
if __name__ == "__main__":
    train_and_save_model()
