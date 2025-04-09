# src/models/decision_tree.py

import os
import pandas as pd
import joblib
import logging

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'src', 'models', 'decision_tree_model.pkl')

def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['stroke'])
    y = df['stroke']
    X = pd.get_dummies(X)
    return X, y

def train_model(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    dtree = DecisionTreeClassifier(random_state=42)

    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('feature_selection', SelectKBest(mutual_info_classif, k=8)),
        ('classifier', dtree)
    ])

    param_grid = {
        'classifier__max_depth': [3, 5, 7],
        'classifier__criterion': ['gini', 'entropy']
    }

    grid = GridSearchCV(pipeline, param_grid, scoring='f1', cv=cv, n_jobs=-1)
    grid.fit(X, y)

    logger.info(f"Best Params: {grid.best_params_}")
    logger.info(f"Best F1 Score: {grid.best_score_:.4f}")
    return grid.best_estimator_

def save_model(model):
    joblib.dump(model, MODEL_PATH)
    logger.info(f"Model saved at {MODEL_PATH}")

def main():
    X, y = load_data()
    model = train_model(X, y)
    save_model(model)

if __name__ == '__main__':
    main()
