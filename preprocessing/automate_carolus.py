import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def preprocess_heart_data(df_input):
    """
    Automated preprocessing pipeline for the heart disease dataset.
    Applies the following steps:
    1. Removes rows with implausible zero values for 'Cholesterol' and 'RestingBP'.
    2. Applies Label Encoding to 'Sex' and 'ExerciseAngina'.
    3. Applies One-Hot Encoding to 'ChestPainType', 'RestingECG', and 'ST_Slope'.
    4. Detects and removes outliers from specified numerical columns using the IQR method.
    """
    df_prep = df_input.copy() # Work on a copy

    # Step 1: Filter 0-values
    print(f"Shape before 0-value filter (Cholesterol/RestingBP): {df_prep.shape}")
    df_prep = df_prep[~((df_prep['Cholesterol'] == 0) | (df_prep['RestingBP'] == 0))]
    print(f"Shape after 0-value filter: {df_prep.shape}")

    # Step 2: Label Encoding
    label_cols = ['Sex', 'ExerciseAngina']
    label_encoders = {}
    print("\n--- Applying Label Encoding ---")
    for col in label_cols:
        if col in df_prep.columns:
            le = LabelEncoder()
            df_prep[col] = le.fit_transform(df_prep[col])
            label_encoders[col] = le
            print(f"Label Encoded: '{col}'")
        else:
            print(f"Warning: Column '{col}' for Label Encoding not found.")

    # Step 3: One-Hot Encoding
    one_hot_encode_cols = ['ChestPainType', 'RestingECG', 'ST_Slope']
    print("\n--- Applying One-Hot Encoding ---")
    existing_one_hot_cols = [col for col in one_hot_encode_cols if col in df_prep.columns]
    if existing_one_hot_cols:
        df_prep = pd.get_dummies(df_prep, columns=existing_one_hot_cols, drop_first=True)
        print(f"One-Hot Encoded: {existing_one_hot_cols}")
        print(f"Shape after One-Hot Encoding: {df_prep.shape}")
    else:
        print("No specified columns found for One-Hot Encoding.")

    # Step 4: Outlier Removal
    def detect_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_indices = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
        return outliers_indices

    outlier_cols = ['RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    print("\n--- Applying Outlier Removal (IQR method) ---")
    for col in outlier_cols:
        if col in df_prep.columns:
            outlier_indices = detect_outliers_iqr(df_prep, col)
            if not outlier_indices.empty:
                df_prep = df_prep.drop(outlier_indices)
                print(f"Outliers removed from '{col}' ({len(outlier_indices)}). New shape: {df_prep.shape}")
            else:
                print(f"No outliers found in '{col}'.")
        else:
            print(f"Warning: Column '{col}' for outlier removal not found.")

    print("\nPreprocessing steps completed.")
    print(f"Final shape of processed data: {df_prep.shape}")
    return df_prep, label_encoders

# --- Main execution block ---
if __name__ == '__main__':
    script_dir = os.path.dirname(__file__) # Directory of this script (preprocessing/)

    # Input CSV is at the project root (one level up from this script)
    INPUT_CSV_PATH = os.path.join(script_dir, '..', 'heart.csv')

    # Output CSV will be in the same directory as this script (preprocessing/)
    OUTPUT_CSV_PATH = os.path.join(script_dir, 'processed_heart_data.csv')

    # The 'preprocessing' directory (script_dir) already exists, so no need to create it.
    # os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True) # This line is removed/not needed

    try:
        raw_df = pd.read_csv(INPUT_CSV_PATH)
        print(f"Original data loaded from '{INPUT_CSV_PATH}'. Shape: {raw_df.shape}")

        df_ready, _ = preprocess_heart_data(raw_df)

        # Save the processed DataFrame to the 'preprocessing' directory
        df_ready.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"\nProcessed data saved to '{OUTPUT_CSV_PATH}'")
        print("--- Head of Preprocessed DataFrame ---")
        print(df_ready.head())

    except FileNotFoundError:
        print(f"Error: Input file not found at '{INPUT_CSV_PATH}'. Please ensure 'heart.csv' is in the project root directory.")
    except Exception as e:
        print(f"An error occurred: {e}")