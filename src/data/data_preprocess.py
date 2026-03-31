import pandas as pd

def preprocess_data(df: pd.DataFrame, target_column: str = "Churn") -> pd.DataFrame:
    """
    Basic cleaning for Telco churn.
    - trim column names
    - drop obvious ID cols
    - fix TotalCharges to numeric
    - map target Churn to 0/1 if needed
    - simple NA handling
    """
    # trim column names
    df.columns = df.columns.str.strip() #remove leading/trailing whitespace from column names

    # drop obvious ID cols
    for col in ['customerID', 'CustomerID', 'customer_id', 'id']:
        if col in df.columns:
             df = df.drop(columns=[col])

    # fix TotalCharges to numeric
    # TotalCharges often has blanks in this dataset -> coerce to float
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # SeniorCitizen should be 0/1 ints if present
    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = df['SeniorCitizen'].fillna(0).astype(int)

    # simple NA handling:
    # - numeric: fill with 0
    # - others: leave for encoders to handle (get_dummies ignores NaN safely)
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(0)

    return df