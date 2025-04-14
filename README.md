# 🧠 Stroke Prediction Using Machine Learning

A project for CSL2050 - Pattern Recognition and Machine Learning at IIT Jodhpur that develops machine learning models to **predict stroke occurrences** based on patient demographics, lifestyle, and health features. Our analysis across multiple clinical scenarios highlights the importance of context-specific model selection, with different algorithms showing strengths in different healthcare settings.

---

## 💡 Problem Statement

Stroke is a leading cause of death and disability worldwide. Early detection is critical for timely intervention. However, predicting stroke is challenging due to the **severe class imbalance**—stroke cases are rare (19.44:1 ratio in our test set). Our goal is to build a system that:
- Handles **imbalanced data** using techniques like **SMOTE**
- Uses **feature engineering**, **selection**, and **classification models**
- Evaluates performance across different clinical scenarios
- Recommends optimal models for specific healthcare contexts

---

## 📁 Dataset

- **Source**: [Healthcare Dataset Stroke Data on Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Raw Data**: 5,110 patient records
- **Target Column**: `stroke` (1 = stroke, 0 = no stroke)

### 📊 Features
- `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`,
  `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`
- Engineered Features: `age_group`, `bmi_category`, `glucose_category`, `age_bmi_interaction`, `hypertension_heart_disease`

---

## ⚙️ Data Preprocessing

- **Cleaning**: Dropped `id`, imputed `bmi` with median
- **Encoding**: LabelEncoded categorical features
- **Feature Engineering**:
  - `age_group`: Young, Middle-Aged, Senior, Elderly
  - `bmi_category`: Underweight, Normal, Overweight, Obese
  - `glucose_category`: Low, Normal, Prediabetes, Diabetes, High Risk
  - Interaction terms
- **Scaling**: Standardized numerical features
- **Class Imbalance Handling**: Applied SMOTE to 70% training data while preserving natural distribution in 30% test set

### ⚠️ Imbalance
- Stroke: **249 cases (4.87%)** in full dataset
- No Stroke: **4,861 cases (95.13%)** in full dataset
- Test set imbalance ratio: **19.44:1**

---

## 📊 Exploratory Data Analysis

- Stroke cases more common among:
  - Married individuals (5.97%)
  - Self-employed (7.8%)
  - Former smokers (7.63%)
- Positive correlations:
  - `age` & stroke: 0.25
  - `hypertension` & `heart_disease`: 0.13

Notebooks available:
- `01_exploratory_data_analysis.ipynb`
- `02_data_preprocessing.ipynb`
- `03_model_comparison.ipynb`

---

## 🧠 Models Implemented and Analyzed

We implemented and optimized seven machine learning algorithms:

1. **Logistic Regression**:
   - PowerTransformer normalization, L1 regularization, 'liblinear' solver
   - Best parameters: `{'classifier__solver': 'liblinear', 'classifier__penalty': 'l1', 'classifier__class_weight': 'balanced', 'classifier__C': 0.1}`
   - Best overall performer (3 out of 4 scenarios)

2. **Random Forest**:
   - PowerTransformer, SelectKBest feature selection
   - Best parameters: `{'feature_selection__score_func': f_classif, 'classifier__n_estimators': 100, 'classifier__criterion': 'entropy', 'classifier__max_depth': 10}`
   - Consistent second-best performer

3. **Artificial Neural Network (ANN)**:
   - MLPClassifier with SMOTE balancing
   - Best parameters: `{'feature_selection__k': 'all', 'ann__hidden_layer_sizes': (256,), 'ann__activation': 'relu', 'ann__alpha': 0.01, 'ann__learning_rate': 'constant'}`
   - Strong performance in limited resource settings

4. **Support Vector Machine (SVM)**:
   - RBF kernel, feature selection (k=25)
   - Best parameters: `{'feature_selection__k': 25, 'classifier__kernel': 'rbf', 'classifier__gamma': 0.1, 'classifier__C': 100}`
   - Good generalization but lower recall

5. **K-Nearest Neighbors (KNN)**:
   - RobustScaler, mutual information feature selection
   - Best parameters: `{'knn__weights': 'distance', 'knn__n_neighbors': 11, 'knn__metric': 'minkowski'}`
   - Moderate performance across scenarios

6. **Decision Tree**:
   - Mutual information feature selection
   - Best parameters: `{'feature_selection__percentile': 85, 'classifier__max_depth': 5, 'classifier__criterion': 'log_loss'}`
   - Second-best for high-risk screening

7. **Gaussian Naïve Bayes (GNB)**:
   - StandardScaler, mutual information feature selection
   - Best parameters: `{'preprocessor__num__scaler': StandardScaler(), 'gnb__var_smoothing': 1e-11}`
   - Highest sensitivity (0.80), ideal for high-risk screening

---

## 📊 Model Performance Across Clinical Scenarios

### General Hospital Screening
1. Logistic Regression (Score: 0.7086)
2. Random Forest (Score: 0.6888)
3. Bayesian (Score: 0.6158)
4. Decision Tree (Score: 0.6119)
5. ANN (Score: 0.6040)
6. KNN (Score: 0.5881)
7. SVM (Score: 0.5668)

### High-Risk Patient Screening
1. Bayesian (Score: 0.8128)
2. Decision Tree (Score: 0.8092)
3. Logistic Regression (Score: 0.7479)
4. Random Forest (Score: 0.7100)
5. KNN (Score: 0.6296)
6. ANN (Score: 0.5436)
7. SVM (Score: 0.4805)

### Limited Resource Setting
1. Logistic Regression (Score: 0.5941)
2. Random Forest (Score: 0.5916)
3. ANN (Score: 0.5852)
4. SVM (Score: 0.5722)
5. KNN (Score: 0.4960)
6. Bayesian (Score: 0.4198)
7. Decision Tree (Score: 0.4154)

### Balanced Clinical Decision Support
1. Logistic Regression (Score: 0.5532)
2. Random Forest (Score: 0.5339)
3. ANN (Score: 0.4770)
4. Bayesian (Score: 0.4404)
5. SVM (Score: 0.4335)
6. KNN (Score: 0.4197)
7. Decision Tree (Score: 0.4102)

---

## 🧪 Key Performance Metrics

Due to the class imbalance, we focused on balanced metrics:

**Logistic Regression**:
- Accuracy: 0.8584
- Sensitivity: 0.6000
- ROC AUC: 0.8332
- Precision: 0.1940
- Specificity: 0.8717

**Bayesian Model**:
- Sensitivity: 0.8000
- NPV: 0.9833
- Highest recall for high-risk scenarios

All model metrics are saved in:
- `model_metrics.json`
- Prediction history in `prediction_history.json`

---

## 📌 Recommendations

Based on our analysis, we recommend:

1. **For Emergency Screening (High Sensitivity Required)**:
   → Use Bayesian Model (0.80 sensitivity, 0.9833 NPV)

2. **For General Hospital Screening**:
   → Use Logistic Regression (balanced sensitivity/specificity)

3. **For Limited Resource Settings**:
   → Use Logistic Regression (efficient, good specificity)

4. **For Balanced Clinical Decision Support**:
   → Use Logistic Regression (good ROC AUC, interpretable)

---

## 🚀 Running the Web App

### 💻 Local Setup

```bash
git clone <repo-url>
cd stroke-prediction-project/demo
pip install -r requirements.txt
python app.py
```

Then go to `http://localhost:5000`

### 🧩 App Features
- Model selection based on clinical scenario
- Home page for input
- Displays top risk factors
- Visual results and prediction history
- Robust error handling and logging (`app.log`)

Templates used:
- `index.html`, `result.html`, `about.html`, `error.html`

---

## 🛠️ Project Structure
```
stroke-prediction-project
├── data/
│   ├── raw/healthcare-dataset-stroke-data.csv
│   └── processed/
│       ├── cleaned_dataset.csv
│       ├── X_test_processed.csv
│       ├── X_train_resampled.csv
│       ├── y_test.csv
│       ├── y_trained_resampled.csv
│       └── feature_statistics.txt
├── demo/
|   └── data/
│       ├── prediction_history.json
|   └── model/
│       ├── model_metrics.json
│       └── stroke_model.pkl
|   └── templates/
│       ├── about.html
│       ├── error.html
│       ├── index.html
│       └── result.html
│   └── app.py
│   └── requirements.txt
├── notebooks/
|   ├── 01_exploratory_data_analysis.ipynb
|   ├── 02_data_preprocessing.ipynb
│   └── 02_model_comparison.ipynb
├── reports/
│   │   ├── mid_reports
|   |   ├── final_report
│   │   └── images....
├── src/
│   ├── models/
│   │   ├── ann.py
|   |   ├── ann_model.pkl
│   │   ├── bayesian.py
|   │   ├── bayesian_model.pkl
│   │   ├── decision_tree.py
|   │   ├── decision_tree_model.pkl
│   │   ├── knn.py
|   │   ├── knn_model.pkl
│   │   ├── logistic_regression.py
|   │   ├── logistic_regression_model.pkl
│   │   ├── random_forest.py
|   │   ├── rf_model.pkl
|   │   ├── svm_model.pkl
│   │   └── svm.py
│   └── data_processing/
│       └── cleaning.py
└── README.md
```

---

## 📌 Future Improvements

- Develop ensemble systems combining logistic regression and Bayesian models
- Implement adaptive threshold selection based on clinical context
- Explore deep learning models with attention mechanisms
- Conduct external validation across diverse patient populations
- Integrate with electronic health record systems
- Perform prospective clinical trials to assess real-world impact

---

## 👥 Contributors

- **Anuranjani** (B23EE1083)  
- **Pratheeksha Bangarapu** (B23EE1058)  
- **Polimetla Eshikha** (B23CS1053)  
- **Sai Pragathi Lagudu** (B23CM1021)  
- **Pragna Sree Muvva** (B23CM1025)

---

## 📚 References

- [Scikit-learn](https://scikit-learn.org/)
- [Imbalanced-learn](https://imbalanced-learn.org/)
- [Kaggle Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- [MLPClassifier - sklearn docs](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- [PowerTransformer - sklearn docs](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html)
- [StandardScaler - sklearn docs][ ]
- [For understanding Pipeline With Hyperparameter Tuning][https://www.kaggle.com/code/hkalita/pipeline-with-hyperparameter-tuning]
- [https://scikit-learn.org/stable/modules/grid_search.html]
- [ColumnTransformer][https://machinelearningmastery.com/columntransformer-for-numerical-and-categorical-data/]