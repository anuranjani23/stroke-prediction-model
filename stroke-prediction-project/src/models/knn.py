import pandas as pd  
import os  
# For saving and loading the trained models:
import joblib  
import logging  

# For the cross-validation and hyperparameter tuning of the model:
from sklearn.model_selection import StratifiedKFold, GridSearchCV  
from sklearn.neighbors import KNeighborsClassifier 
# For the data scaling and encoding of the categorical features in dataset:
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
# For preprocessing the pipeline for different data types: 
from sklearn.compose import ColumnTransformer 
# Using the Synthetic Minority Over-sampling Technique (SMOTE) to handle the class imbalance:
from imblearn.over_sampling import SMOTE 
from imblearn.pipeline import Pipeline as ImbPipeline  # Pipeline that supports the imbalanced learning methods.
from sklearn.feature_selection import SelectKBest, f_classif  # Feature selection using ANOVA F-test.


# Setting up the logging to display important information during execution:
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)  

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_dataset.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'src', 'models')  
MODEL_PATH = os.path.join(MODELS_DIR, 'knn_model.pkl')  


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
            ('num', StandardScaler(), numeric_features),  
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) 
        ])

    
    imbalanced_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),  # Applying the feature scaling and encoding.
        ('smote', SMOTE(random_state=42)),  # Applying the SMOTE to handle the class imbalance.
        ('feature_selection', SelectKBest(f_classif)),  # Selecting the best features using ANOVA F-test.
        ('knn', KNeighborsClassifier())  
    ])

    # Grid definition:
    param_grid = {
        'feature_selection__k': [5, 8, 10, 'all'],  
        'knn__n_neighbors': [3, 5, 7, 9, 11],  
        'knn__weights': ['uniform', 'distance'],  
        'knn__metric': ['euclidean', 'manhattan']  
    }

    # 5-fold stratified cross-validation:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  
    grid_search = GridSearchCV(
        imbalanced_pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1  # Using the ROC AUC as an evaluation metric.
    )
    
    # Fit the model using grid search:
    grid_search.fit(X, y)

    # Logging the best parameters and the performance:
    logger.info(f"Best Parameters: {grid_search.best_params_}")  
    logger.info(f"Best ROC AUC Score: {grid_search.best_score_:.4f}")  
    return grid_search.best_estimator_  


def train_and_save_model():
    os.makedirs(MODELS_DIR, exist_ok=True)
    X, y = load_and_prepare_data()
    best_model = perform_hyperparameter_tuning(X, y)
    joblib.dump(best_model, MODEL_PATH)
    logger.info(f"KNN model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save_model()  






# GridSearchCV is a methd in scikit-learn which is used for hyperparameter tuning by searching through a grid of parameters, by using cross-validation.
# ANOVA F-test or analysis of variance f-test is a statistical method which helps in selecting relevant features by measuring their variance across different classes,
# it helps improve k-NN's performance by reducing the noise. A higher f-score means that feature is more important.
# SMOTE or synthetic minority over-sampling technique helps with class imbalancing, it prevents k-NN from being biased towards the majority class. It is a resampling 
# method that generates synthetic samples for the minority class to balance the dataset.






