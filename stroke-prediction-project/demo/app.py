from flask import Flask, render_template, request, flash, redirect, url_for, session, jsonify
import pickle
import numpy as np
import os
import json
from datetime import datetime
import logging
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For flash messages

# Configuration
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODELS = {
    'knn': {
        'path': os.path.join(MODEL_DIR, 'knn_model.pkl'),
        'name': 'K-Nearest Neighbors',
        'metrics': {
            'accuracy': 0.833,
            'precision': 0.64,  # Calculated based on high sensitivity but class imbalance
            'recall': 0.947,    # Sensitivity
            'specificity': 0.76,
            'f1_score': 0.76,
            'auc': 0.959,
        },
        'best_for': 'High-risk screening where catching all potential stroke cases is critical',
        'strengths': 'Highest sensitivity/recall and ROC AUC, indicating excellent ability to identify true stroke cases',
        'limitations': 'Lower precision means more false positives'
    },
    'logistic': {
        'path': os.path.join(MODEL_DIR, 'logistic_model.pkl'),
        'name': 'Logistic Regression',
        'metrics': {
            'accuracy': 0.73,
            'precision': 0.53,
            'recall': 0.827,    # Sensitivity
            'specificity': 0.728,
            'f1_score': 0.65,
            'auc': 0.84,
        },
        'best_for': 'Clinical settings requiring good balance between sensitivity and interpretability',
        'strengths': 'Good sensitivity while maintaining interpretable results',
        'limitations': 'Lower overall accuracy compared to some models'
    },
    'random_forest': {
        'path': os.path.join(MODEL_DIR, 'rf_model.pkl'),
        'name': 'Random Forest',
        'metrics': {
            'accuracy': 0.755,
            'precision': 0.58,
            'recall': 0.74,     # Sensitivity
            'specificity': 0.756,
            'f1_score': 0.65,
            'auc': 0.85,
        },
        'best_for': 'Balanced approach when both prediction accuracy and false positive reduction are important',
        'strengths': 'Good overall performance with strong specificity',
        'limitations': 'Slightly lower sensitivity than KNN and Logistic Regression'
    },
    'ann': {
        'path': os.path.join(MODEL_DIR, 'ann_model.pkl'),
        'name': 'Artificial Neural Network',
        'metrics': {
            'accuracy': 0.74,
            'precision': 0.56,
            'recall': 0.81,     # Sensitivity
            'specificity': 0.73,
            'f1_score': 0.66,
            'auc': 0.84,
        },
        'best_for': 'Personalized risk assessment with good overall performance',
        'strengths': 'Good sensitivity with reasonable specificity',
        'limitations': 'Less interpretable than simpler models'
    }
}

DEFAULT_MODEL = 'knn'  # Set KNN as default based on analysis
HISTORY_FILE = os.path.join(os.path.dirname(__file__), 'data', 'prediction_history.json')

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)

# Load models
models = {}
for model_id, model_info in MODELS.items():
    model_path = model_info['path']
    # For demonstration, fall back to single model if specific models don't exist
    if not os.path.exists(model_path) and os.path.exists(os.path.join(os.path.dirname(__file__), 'model', 'stroke_model.pkl')):
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'stroke_model.pkl')
        logger.warning(f"Model {model_id} not found. Using default model.")
    
    try:
        if os.path.exists(model_path):
            models[model_id] = pickle.load(open(model_path, 'rb'))
            logger.info(f"Model {model_id} loaded successfully")
        else:
            models[model_id] = None
            logger.warning(f"Model {model_id} file not found")
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {e}")
        models[model_id] = None

# Clinical contexts with recommended models
CLINICAL_CONTEXTS = {
    'emergency': {
        'name': 'Emergency Screening',
        'description': 'High sensitivity required to ensure minimal missed stroke cases',
        'recommended_model': 'knn',
        'explanation': 'KNN has the highest sensitivity (94.7%) and ROC AUC (95.9%), ensuring minimal missed stroke cases. This is crucial in emergency settings where missing a stroke case could be fatal.'
    },
    'general': {
        'name': 'General Hospital Screening',
        'description': 'Balanced approach for routine clinical screening',
        'recommended_model': 'knn',
        'explanation': 'KNN offers the highest overall accuracy (83.3%) while maintaining excellent sensitivity.'
    },
    'limited': {
        'name': 'Limited Resource Setting',
        'description': 'Optimized for settings with limited follow-up capacity',
        'recommended_model': 'random_forest',
        'explanation': 'Random Forest offers the best balance of metrics with good specificity (75.6%), helping reduce unnecessary referrals when resources are constrained.'
    },
    'personalized': {
        'name': 'Personalized Risk Assessment',
        'description': 'Detailed risk assessment for individual patient counseling',
        'recommended_model': 'ann',
        'explanation': 'ANN provides good sensitivity (81%) with reasonable specificity (73%), well-suited for integration into personalized health monitoring systems.'
    }
}

# Feature engineering function
def perform_feature_engineering(df):
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], labels=['Young', 'Middle-Aged', 'Senior', 'Elderly'])
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    df['glucose_category'] = pd.cut(df['avg_glucose_level'], bins=[0, 70, 100, 125, 200, 1000], labels=['Low', 'Normal', 'Prediabetes', 'Diabetes', 'High Risk'])
    df['age_bmi_interaction'] = df['age'] * df['bmi']
    df['hypertension_heart_disease'] = df['hypertension'] * df['heart_disease']
    return df

# Feature descriptions
FEATURE_INFO = {
    'age': 'Age of the patient in years',
    'avg_glucose_level': 'Average glucose level in blood (mg/dL)',
    'bmi': 'Body Mass Index, weight(kg)/(height(m))²',
    'hypertension': 'Whether the patient has hypertension (0: No, 1: Yes)',
    'heart_disease': 'Whether the patient has heart disease (0: No, 1: Yes)',
    'gender': 'Gender of the patient',
    'ever_married': 'Whether the patient has ever been married',
    'work_type': 'Type of work/occupation',
    'residence_type': 'Type of residence area',
    'smoking_status': 'Smoking history of the patient'
}

VALID_RANGES = {
    'age': (0, 120),
    'avg_glucose_level': (50, 300),
    'bmi': (10, 60)
}

GENDER_OPTIONS = {'Female': 0, 'Male': 1, 'Other': 2}
MARRIED_OPTIONS = {'No': 0, 'Yes': 1}
WORK_TYPE_OPTIONS = {'Govt_job': 0, 'Never_worked': 1, 'Private': 2, 'Self-employed': 3, 'children': 4}
RESIDENCE_OPTIONS = {'Rural': 0, 'Urban': 1}
SMOKING_OPTIONS = {'Unknown': 0, 'formerly smoked': 1, 'never smoked': 2, 'smokes': 3}

REVERSE_GENDER = {v: k for k, v in GENDER_OPTIONS.items()}
REVERSE_MARRIED = {v: k for k, v in MARRIED_OPTIONS.items()}
REVERSE_WORK_TYPE = {v: k for k, v in WORK_TYPE_OPTIONS.items()}
REVERSE_RESIDENCE = {v: k for k, v in RESIDENCE_OPTIONS.items()}
REVERSE_SMOKING = {v: k for k, v in SMOKING_OPTIONS.items()}

# Risk categories based on probability
def get_risk_category(probability):
    if probability < 20:
        return "Very Low", "text-success"
    elif probability < 40:
        return "Low", "text-info"
    elif probability < 60:
        return "Moderate", "text-warning"
    elif probability < 80:
        return "High", "text-danger"
    else:
        return "Very High", "text-danger font-weight-bold"

df = pd.read_csv("../data/raw/healthcare-dataset-stroke-data.csv")
# Population averages for comparison (these would be calculated from your dataset)
POPULATION_AVERAGES = {
    'age': df['age'].mean(),
    'avg_glucose_level': df['avg_glucose_level'].mean(),
    'bmi': df['bmi'].mean(),
    'hypertension_rate': df['hypertension'].mean(),      # if 1 means yes, 0 means no
    'heart_disease_rate': df['heart_disease'].mean(),    # same assumption
    'stroke_risk': df['stroke'].mean()                   # assuming binary 1/0
}

def load_prediction_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading prediction history: {e}")
            return []
    return []

def save_prediction(features, prediction, probability, model_id, context_id):
    history = load_prediction_history()
    feature_dict = {
        'age': features[0],
        'avg_glucose_level': features[1],
        'bmi': features[2],
        'hypertension': 'Yes' if features[3] == 1 else 'No',
        'heart_disease': 'Yes' if features[4] == 1 else 'No',
        'gender': REVERSE_GENDER.get(features[5], 'Unknown'),
        'ever_married': 'Yes' if features[6] == 1 else 'No',
        'work_type': REVERSE_WORK_TYPE.get(features[7], 'Unknown'),
        'residence_type': REVERSE_RESIDENCE.get(features[8], 'Unknown'),
        'smoking_status': REVERSE_SMOKING.get(features[9], 'Unknown')
    }
    
    risk_category, _ = get_risk_category(probability * 100)
    
    entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'features': feature_dict,
        'prediction': int(prediction),
        'probability': round(float(probability), 3),
        'risk_category': risk_category,
        'model_used': MODELS[model_id]['name'] if model_id in MODELS else 'Unknown Model',
        'clinical_context': CLINICAL_CONTEXTS[context_id]['name'] if context_id in CLINICAL_CONTEXTS else 'General Screening'
    }
    history.append(entry)
    if len(history) > 100:
        history = history[-100:]
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving prediction: {e}")

def get_risk_factors(features):
    risk_factors = []
    if features[0] > 65:
        risk_factors.append({"factor": "Advanced age", "detail": f"{features[0]} years (risk increases with age)", "severity": "high" if features[0] > 75 else "moderate"})
    if features[1] > 150:
        risk_factors.append({"factor": "High blood glucose", "detail": f"{features[1]} mg/dL (target: <100 mg/dL fasting)", "severity": "high" if features[1] > 200 else "moderate"})
    if features[2] > 30:
        risk_factors.append({"factor": "Obesity", "detail": f"BMI {features[2]:.1f} (target: 18.5-24.9)", "severity": "high" if features[2] > 35 else "moderate"})
    if features[3] == 1:
        risk_factors.append({"factor": "Hypertension", "detail": "Present (increases stroke risk by 4-6 times)", "severity": "high"})
    if features[4] == 1:
        risk_factors.append({"factor": "Heart disease", "detail": "Present (increases stroke risk by 2-3 times)", "severity": "high"})
    if features[9] == 3:  # Currently smoking
        risk_factors.append({"factor": "Smoking", "detail": "Current smoker (doubles stroke risk)", "severity": "high"})
    elif features[9] == 1:  # Formerly smoked
        risk_factors.append({"factor": "Former smoker", "detail": "Risk decreases over time but remains elevated", "severity": "moderate"})
    
    # Add combined risk factors
    if features[3] == 1 and features[4] == 1:
        risk_factors.append({"factor": "Hypertension + Heart disease", "detail": "Combined conditions significantly increase risk", "severity": "very high"})
    
    return risk_factors

def create_risk_comparison_chart(user_risk, population_avg=POPULATION_AVERAGES['stroke_risk']):
    plt.figure(figsize=(8, 4))
    labels = ['Your Risk', 'Population Average']
    values = [user_risk, population_avg]
    colors = ['#ff9999' if user_risk > population_avg else '#66b3ff', '#99ff99']
    
    bars = plt.bar(labels, values, color=colors)
    plt.ylabel('Stroke Risk Probability')
    plt.title('Your Risk Compared to Population Average')
    plt.ylim(0, max(max(values) + 0.1, 0.5))  # Ensure y-axis shows at least up to 0.5 (50%)
    
    # Add text labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.1%}', ha='center', va='bottom')
    
    # Save plot to a temporary bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    # Encode the bytes to base64
    encoded = base64.b64encode(image_png).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

def create_model_comparison_chart(user_risk, model_metrics):
    plt.figure(figsize=(10, 6))
    
    # Prepare data
    metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'AUC']
    values = [model_metrics['accuracy'], model_metrics['recall'], 
              model_metrics['specificity'], model_metrics['auc']]
    
    # Create the bar chart
    bars = plt.bar(metrics, values, color=['#4361ee', '#3a0ca3', '#4895ef', '#4cc9f0'])
    
    # Add title and labels
    plt.title('Model Performance Metrics')
    plt.ylabel('Score (0-1)')
    plt.ylim(0, 1.1)  # Set y-axis from 0 to 1
    
    # Add text labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    # Save plot to a temporary bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    # Encode the bytes to base64
    encoded = base64.b64encode(image_png).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

def generate_personalized_recommendations(risk_factors, probability):
    general_recs = [
        "Schedule regular check-ups with your healthcare provider",
        "Maintain a healthy weight",
        "Exercise regularly (aim for 150 minutes of moderate activity weekly)",
        "Follow a balanced diet rich in fruits, vegetables, and whole grains",
        "Limit alcohol consumption"
    ]
    
    specific_recs = []
    
    # Add specific recommendations based on risk factors
    for factor in risk_factors:
        if factor["factor"] == "Advanced age":
            specific_recs.append("Consider additional cardiovascular screenings given your age")
        elif factor["factor"] == "High blood glucose":
            specific_recs.append("Monitor your blood glucose levels regularly")
            specific_recs.append("Consider consulting with an endocrinologist")
            specific_recs.append("Follow a low-glycemic diet plan")
        elif factor["factor"] == "Obesity":
            specific_recs.append("Work with a healthcare provider on a weight management plan")
            specific_recs.append("Aim for 5-10% weight loss initially to improve health markers")
        elif factor["factor"] == "Hypertension":
            specific_recs.append("Monitor your blood pressure regularly")
            specific_recs.append("Take blood pressure medications as prescribed")
            specific_recs.append("Reduce sodium intake to less than 1,500mg daily")
        elif factor["factor"] == "Heart disease":
            specific_recs.append("Follow your cardiologist's recommendations closely")
            specific_recs.append("Take medications exactly as prescribed")
            specific_recs.append("Learn the warning signs of heart attack and stroke")
        elif factor["factor"] == "Smoking" or factor["factor"] == "Former smoker":
            specific_recs.append("Quit smoking or avoid relapse if you've quit")
            specific_recs.append("Consider smoking cessation programs or medications")
    
    # Add urgent recommendations for high risk
    if probability > 0.5:
        urgent_recs = [
            "Consult with a healthcare provider as soon as possible",
            "Discuss stroke prevention strategies with a neurologist",
            "Learn the FAST signs of stroke: Face drooping, Arm weakness, Speech difficulty, Time to call emergency services"
        ]
        return {"general": general_recs, "specific": specific_recs, "urgent": urgent_recs}
    
    return {"general": general_recs, "specific": specific_recs}

@app.route('/')
def index():
    return render_template('index.html', 
                          feature_info=FEATURE_INFO,
                          gender_options=GENDER_OPTIONS,
                          married_options=MARRIED_OPTIONS,
                          work_options=WORK_TYPE_OPTIONS,
                          residence_options=RESIDENCE_OPTIONS,
                          smoking_options=SMOKING_OPTIONS,
                          clinical_contexts=CLINICAL_CONTEXTS,
                          models=MODELS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get selected model and clinical context
        model_id = request.form.get('model_id', DEFAULT_MODEL)
        context_id = request.form.get('context_id', 'general')
        
        # Use selected model or fall back to default
        selected_model = models.get(model_id)
        if selected_model is None:
            selected_model = models.get(DEFAULT_MODEL)
            model_id = DEFAULT_MODEL
            if selected_model is None:
                flash("No models available. Please contact administrator.", "danger")
                return redirect(url_for('index'))
        
        logger.info(f"Using model: {model_id}, Context: {context_id}")
        logger.info(f"Form data received: {request.form}")

        input_data = {
            'gender': int(request.form['gender']),
            'age': float(request.form['age']),
            'hypertension': int(request.form['hypertension']),
            'heart_disease': int(request.form['heart_disease']),
            'ever_married': int(request.form['ever_married']),
            'work_type': int(request.form['work_type']),
            'Residence_type': int(request.form['residence_type']),
            'avg_glucose_level': float(request.form['avg_glucose_level']),
            'bmi': float(request.form['bmi']),
            'smoking_status': int(request.form['smoking_status']),
            'stroke': 0
        }

        for key in ['age', 'avg_glucose_level', 'bmi']:
            min_val, max_val = VALID_RANGES[key]
            if not (min_val <= input_data[key] <= max_val):
                flash(f"{key.replace('_', ' ').title()} should be between {min_val} and {max_val}", "danger")
                return redirect(url_for('index'))

        input_df = pd.DataFrame([input_data])
        input_df = perform_feature_engineering(input_df)

        # Get prediction and probability
        prediction = selected_model.predict(input_df)[0]
        probabilities = selected_model.predict_proba(input_df)[0]
        stroke_probability = probabilities[1]  # Probability of positive class (stroke)
        
        logger.info(f"Prediction: {prediction}, Probability: {stroke_probability}")

        # Feature list for risk factor analysis
        features_list = [
            input_data['age'],
            input_data['avg_glucose_level'],
            input_data['bmi'],
            input_data['hypertension'],
            input_data['heart_disease'],
            input_data['gender'],
            input_data['ever_married'],
            input_data['work_type'],
            input_data['Residence_type'],
            input_data['smoking_status']
        ]
        
        # Get risk factors and save prediction
        risk_factors = get_risk_factors(features_list)
        save_prediction(features_list, prediction, stroke_probability, model_id, context_id)
        
        # Generate risk comparison chart
        risk_chart = create_risk_comparison_chart(stroke_probability)
        
        # Generate model performance chart
        model_chart = create_model_comparison_chart(stroke_probability, MODELS[model_id]['metrics'])
        
        # Get personalized recommendations
        recommendations = generate_personalized_recommendations(risk_factors, stroke_probability)
        
        # Determine risk category
        risk_category, category_class = get_risk_category(stroke_probability * 100)

        # Store everything in session for the result page
        session['prediction'] = int(prediction)
        session['probability'] = round(stroke_probability * 100, 1)
        session['probability_no_stroke'] = round(probabilities[0] * 100, 1)
        session['risk_factors'] = risk_factors
        session['features'] = features_list
        session['risk_category'] = risk_category
        session['category_class'] = category_class
        session['risk_chart'] = risk_chart
        session['model_chart'] = model_chart
        session['recommendations'] = recommendations
        session['model_id'] = model_id
        session['context_id'] = context_id
        session['model_info'] = MODELS[model_id]
        session['context_info'] = CLINICAL_CONTEXTS[context_id]

        return redirect(url_for('result'))

    except ValueError as e:
        flash("Invalid input: Please check that all fields contain valid numbers", "danger")
        logger.error(f"Value error in prediction: {e}")
        logger.error(f"Form data was: {request.form}")
        return redirect(url_for('index'))
    except Exception as e:
        flash(f"An error occurred: {str(e)}", "danger")
        logger.error(f"Error in prediction: {e}")
        return redirect(url_for('index'))

@app.route('/result')
def result():
    # Retrieve all stored session data
    prediction = session.get('prediction')
    probability = session.get('probability')
    probability_no_stroke = session.get('probability_no_stroke')
    risk_factors = session.get('risk_factors', [])
    risk_category = session.get('risk_category', 'Unknown')
    category_class = session.get('category_class', 'text-secondary')
    risk_chart = session.get('risk_chart', '')
    model_chart = session.get('model_chart', '')
    recommendations = session.get('recommendations', {'general': [], 'specific': []})
    model_id = session.get('model_id', DEFAULT_MODEL)
    context_id = session.get('context_id', 'general')
    model_info = session.get('model_info', MODELS.get(DEFAULT_MODEL, {}))
    context_info = session.get('context_info', CLINICAL_CONTEXTS.get('general', {}))
    
    # Convert model metrics to percentage for display
    display_metrics = {k: f"{v*100:.1f}%" for k, v in model_info.get('metrics', {}).items()}
    
    # Render the enhanced result template
    return render_template('result.html', 
                          prediction=prediction, 
                          probability=probability,
                          probability_no_stroke=probability_no_stroke,
                          risk_factors=risk_factors,
                          risk_category=risk_category,
                          category_class=category_class,
                          risk_chart=risk_chart,
                          model_chart=model_chart,
                          recommendations=recommendations,
                          model_info=model_info,
                          context_info=context_info,
                          model_metrics=display_metrics)

@app.route('/history')
def history():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                predictions = json.load(f)
        else:
            predictions = []

        return render_template('history.html', predictions=predictions)
    except Exception as e:
        app.logger.error(f"Error loading history: {e}")
        return f"<h1>Something went wrong: {e}</h1>", 500
    
@app.route('/about')
def about():
    return render_template('about.html', 
                          models=MODELS, 
                          clinical_contexts=CLINICAL_CONTEXTS,
                          class_imbalance_ratio="19.44:1")

@app.route('/api/context_recommendation', methods=['GET'])
def get_context_recommendation():
    context_id = request.args.get('context_id', 'general')
    
    if context_id in CLINICAL_CONTEXTS:
        context = CLINICAL_CONTEXTS[context_id]
        recommended_model = context['recommended_model']
        return jsonify({
            'recommended_model': recommended_model,
            'explanation': context['explanation']
        })
    else:
        return jsonify({'error': 'Context not found'}), 404

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error_code=404, message="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', error_code=500, message="Internal server error"), 500

if __name__ == '__main__':
    app.run(debug=True)