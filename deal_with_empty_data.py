import os
import pandas as pd
from sklearn.impute import KNNImputer

# Directory path containing CSV files
directory_path = r'./ML$'

# List all CSV files in the directory
csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

# Create a function to fill NaN values with multiple methods
def fill_nan_values(df):
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Interpolation (linear interpolation)
    df.interpolate(method='linear', limit_direction='forward', inplace=True)

    # KNN Imputation (K-Nearest Neighbors)
    knn_imputer = KNNImputer(n_neighbors=5)
    df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])

    # Handle categorical columns (impute using the mode or other strategies)
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Backward Fill
    df.fillna(method='bfill', inplace=True)
    
    # Forward Fill
    df.fillna(method='ffill', inplace=True)
    
    return df

# Process each CSV file
for csv_file in csv_files:
    # Load CSV file into pandas DataFrame
    file_path = os.path.join(directory_path, csv_file)
    df = pd.read_csv(file_path)

    # Fill missing values using different methods
    df_filled = fill_nan_values(df)

    # Save the processed DataFrame back to the same file (overwrite the original)
    df_filled.to_csv(file_path, index=False)

    print(f'Processed and overwritten {csv_file}')
