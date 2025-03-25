import os  
import logging  
import pandas as pd 

# For feature scaling and encoding using the preprocessing package from scikit-learn:
from sklearn.preprocessing import LabelEncoder, StandardScaler  
# For handling/imputing the missing values using the impute package from scikit-learn:
from sklearn.impute import SimpleImputer 


# Configuring the logging to track execution details and errors:
logging.basicConfig(
    level=logging.INFO,  # Set logging level to INFO to capture key events.
    format='%(asctime)s - %(levelname)s: %(message)s',  
    datefmt='%Y-%m-%d %H:%M:%S'  
)
# Creating a logger instance:
logger = logging.getLogger(__name__)  


# Defining the paths for data files using absolute paths to avoid issues:

# Base directory of the project:
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  
# Raw dataset path:
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'healthcare-dataset-stroke-data.csv') 
# Processed dataset path:
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_dataset.csv')  
# Path for feature statistics report (initial):
FEATURE_STATS_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'feature_statistics.txt')  

# Function to validate the file_path:
def validate_data_path(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

# Function to perform advanced categorisation of already present features to have a better analysis:
def perform_feature_engineering(df):

    # Categorizing the age feature into groups of Young, Senior etc.:
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], labels=['Young', 'Middle-Aged', 'Senior', 'Elderly'])
    
    # Categorizing the BMI into groupsof Underweight, Normal etc.:
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    # Categorizing the glucose levels into risk categories of Low, Normal, Prediabetes etc.:
    df['glucose_category'] = pd.cut(df['avg_glucose_level'], bins=[0, 70, 100, 125, 200, 1000], labels=['Low', 'Normal', 'Prediabetes', 'Diabetes', 'High Risk'])
    
    # Creating interaction features:
    df['age_bmi_interaction'] = df['age'] * df['bmi']  # Interaction term between age and BMI.
    df['hypertension_heart_disease'] = df['hypertension'] * df['heart_disease']  # Interaction between hypertension and heart disease.
    
    return df

# Function to get label encoding mapping for categorical features:
def get_label_encoding_mapping(categorical_columns):
    
    label_mappings = {}
    
    # Read the original dataset
    original_df = pd.read_csv(RAW_DATA_PATH)
    
    # Create a label encoder
    le = LabelEncoder()
    
    for col in categorical_columns:
        # Fit the encoder to the original data
        original_data = original_df[col]
        le.fit(original_data.astype(str))
        
        # Create a mapping dictionary
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        label_mappings[col] = mapping
    
    return label_mappings

def preprocess_data():
    try:
        # Validating the input file path:
        validate_data_path(RAW_DATA_PATH)
        logger.info(f"Loading dataset from {RAW_DATA_PATH}")
        df = pd.read_csv(RAW_DATA_PATH)
        
        # Logging the dataset dimensions and missing value summary:
        logger.info(f"Dataset shape: {df.shape}")  
        logger.info(f"Missing values:\n{df.isnull().sum()}")  
        
        # Creating the necessary directories if they don't exist:
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(FEATURE_STATS_PATH), exist_ok=True)
        
        # Dropping/Removing unnecessary columns:
        df.drop(columns=['id'], inplace=True, errors='ignore')
        
        # Handling the missing values in numerical columns using the median imputation (replace it with the median of that feature column):
        numerical_columns = ['age', 'bmi', 'avg_glucose_level']
        imputer = SimpleImputer(strategy='median')
        df[numerical_columns] = imputer.fit_transform(df[numerical_columns])
        
        # Defining the categorical columns for encoding:
        categorical_columns = ['gender', 'smoking_status', 'ever_married', 'work_type', 'Residence_type']
        
        # Get label encoding mapping before transformation
        label_encoding_mapping = get_label_encoding_mapping(categorical_columns)
        
        # Converting the categorical columns into numerical values using Label Encoding:
        label_encoder = LabelEncoder()
        for col in categorical_columns:
            df[col] = label_encoder.fit_transform(df[col].astype(str))
        
        # Applying the advanced feature categorization:
        df = perform_feature_engineering(df)
        
        # Scaling the numerical features using StandardScaler:
        standard_scaler = StandardScaler()
        df[numerical_columns] = standard_scaler.fit_transform(df[numerical_columns])
        
        # Ensuring that the target variable ('stroke') remains as an integer:
        df['stroke'] = df['stroke'].astype(int)
        
        # Generate feature statistics report (after the processing and analysis):
        with open(FEATURE_STATS_PATH, 'w') as f:
            # Report Header
            f.write("=" * 50 + "\n")
            f.write("     COMPREHENSIVE FEATURE STATISTICS REPORT     \n")
            f.write("=" * 50 + "\n\n")

            # 1. Numerical Features Distribution
            f.write("1. NUMERICAL FEATURES DISTRIBUTION\n")
            f.write("-" * 40 + "\n")
            # Convert describe() output to a more readable format
            desc_stats = df[numerical_columns].describe()
            for col in numerical_columns:
                f.write(f"\n{col.upper()} Statistics:\n")
                for stat, value in desc_stats.loc[:, col].items():
                    f.write(f"  {stat.capitalize()}: {value:.4f}\n")
            f.write("\n")

            # 2. Categorical Features Distribution
            f.write("2. CATEGORICAL FEATURES DISTRIBUTION\n")
            f.write("-" * 40 + "\n")
            for col in categorical_columns:
                f.write(f"\n{col.upper()} Distribution:\n")
                value_counts = df[col].value_counts(normalize=True)
                for category, percentage in value_counts.items():
                    f.write(f"  {category}: {percentage:.2%}\n")
            f.write("\n")

            # 3. Target Variable Distribution
            f.write("3. TARGET VARIABLE DISTRIBUTION\n")
            f.write("-" * 40 + "\n")
            target_dist = df['stroke'].value_counts(normalize=True)
            for category, percentage in target_dist.items():
                f.write(f"  {category}: {percentage:.2%}\n")
            f.write("\n")

            # 4. Label Encoding Mapping
            f.write("4. LABEL ENCODING MAPPING\n")
            f.write("-" * 40 + "\n")
            for col, mapping in label_encoding_mapping.items():
                f.write(f"\n{col.upper()} Mapping:\n")
                for original, encoded in sorted(mapping.items(), key=lambda x: x[1]):
                    f.write(f"  {original:<15} -> {encoded}\n")

            # Footer
            f.write("\n" + "=" * 50 + "\n")
            f.write("     END OF FEATURE STATISTICS REPORT     \n")
            f.write("=" * 50 + "\n")
        
        # Saving the processed data to CSV:
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        
        # Logging the completion messages:
        logger.info(f"Processed data saved at {PROCESSED_DATA_PATH}")
        logger.info(f"Processed dataset shape: {df.shape}")
        logger.info(f"Feature statistics saved at {FEATURE_STATS_PATH}")
        
        return df  
    
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")  
        raise  

def main():
    try:
        processed_data = preprocess_data()  
        
        # Display processed dataset preview and structure:
        print("\nProcessed Data Overview:")
        print(processed_data.head())  # Showing the first 5 rows of processed data.
        print("\nData Info:")
        processed_data.info()  # Display the info like data type and non-null counts.
    
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")  

if __name__ == "__main__":  
    main()