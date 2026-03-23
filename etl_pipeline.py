import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# 1. Extract Data
def extract_data(file_path):
    print("Extracting data...")
    df = pd.read_csv(file_path)
    print(f"Data shape: {df.shape}")
    return df

# 2. Transform Data
def transform_data(df, target_column):
    print("Transforming data...")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object','string']).columns
    
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    X_processed = preprocessor.fit_transform(X)
    print("Transformation complete.")
    return X_processed, y, preprocessor

# 3. Load Data (Save)
def load_data(X, y, preprocessor, output_prefix="processed"):
    print("Loading (saving) processed data...")
    X_df = pd.DataFrame(X.toarray() if hasattr(X, 'toarray') else X)
    y_df = pd.DataFrame(y)
    
    X_df.to_csv(f"{output_prefix}_features.csv", index=False)
    y_df.to_csv(f"{output_prefix}_target.csv", index=False)
    joblib.dump(preprocessor, f"{output_prefix}_pipeline.pkl")
    print("Data saved successfully.")

# 4. Full Pipeline Execution
def run_pipeline(file_path, target_column):
    df = extract_data(file_path)
    X, y, preprocessor = transform_data(df, target_column)
    load_data(X, y, preprocessor)

# 5. Execution
if __name__ == "__main__":
    FILE_PATH = "data.csv"  # Replace with your dataset
    TARGET_COLUMN = "Target"  # Replace with your target column name
    
    run_pipeline(FILE_PATH, TARGET_COLUMN)