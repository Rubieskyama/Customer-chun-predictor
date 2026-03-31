import pandas as pd

def _map_binary_series(s: pd.Series) -> pd.Series:
    """
    Apply deterministic binary encoding to 2-category features.
    
    This function implements the core binary encoding logic that converts
    categorical features with exactly 2 values into 0/1 integers. The mappings
    are deterministic and must be consistent between training and serving.
    """
    # Get unique values and remove NaN
    vals = list(pd.Series(s.dropna().unique()))
    valset = set(vals)

    # === DETERMINISTIC BINARY MAPPINGS ===
    # Map 'Yes'/'No' to 1/0
    if valset == {'Yes', 'No'}:
        return s.map({'Yes': 1, 'No': 0}).astype("int64")
    
    # Gender mapping
    if valset == {'Male', 'Female'}:
        return s.map({'Male': 1, 'Female': 0}).astype("int64")
    
    # === GENERIC BINARY MAPPING ===
    # For any other 2-category feature, use stable alphabetical ordering
    if len(valset) == 2:
        sorted_vals = sorted(valset)
        return s.map({sorted_vals[0]: 0, sorted_vals[1]: 1}).astype("int64")

    # === NON-BINARY FEATURES ===
    # Return unchanged - will be handled by one-hot encoding
    return s

def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Apply complete feature engineering pipeline for training data.
    
    This is the main feature engineering function that transforms raw customer data
    into ML-ready features. The transformations must be exactly replicated in the
    serving pipeline to ensure prediction accuracy.

    """
    df = df.copy()  # Avoid modifying original DataFrame
    print(f"Starting feature engineering on {df.shape[1]} columns...")

    # === STEP 1: Identify Feature Types ===
    # Find categorical columns (object dtype) excluding the target variable
    obj_cols = [col for col in df.select_dtypes(include="object") if col != target_col]
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    print(f"Identified {len(obj_cols)} categorical columns and {len(numeric_cols)} numeric columns.")

    # === STEP 2: Split Categorical by Cardinality ===
    # Binary features (exactly 2 unique values) get binary encoding
    # Multi-category features (>2 unique values) get one-hot encoding
    binary_cols = [col for col in obj_cols if df[col].nunique() == 2]
    multi_cols = [col for col in obj_cols if df[col].nunique() > 2]

    print(f"Binary features: {len(binary_cols)} | Multi-category features: {len(multi_cols)}")
    if binary_cols:
        print(f"Binary:{binary_cols}")
    if multi_cols:
        print(f"Multi-category:{multi_cols}")

    # === STEP 3: Apply Binary Encoding ===
    # Convert 2-category features to 0/1 using deterministic mappings
    for col in binary_cols:
        original_dtype = df[col].dtype
        df[col] = _map_binary_series(df[col].astype(str))  # Ensure string type for mapping
        print(f"{col}': {original_dtype} to binary (0/1)")

    # === STEP 4: Convert Boolean Columns ===
    # XGBoost requires integer inputs, not boolean
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype(int)
        print(f"Converted {len(bool_cols)} boolean columns to int (0/1): {bool_cols}")
    
    # === STEP 5: One-Hot Encoding for Multi-Category Features ===
    # CRITICAL: drop_first=True prevents multicollinearity
    if multi_cols:
        print(f"Applying one-hot encoding to {len(multi_cols)} multi-category features...")
        original_shape = df.shape

        # Apply one-hot encoding with drop_first=True (same as serving)
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

        new_shape = df.shape
        new_features = new_shape[1] - original_shape[1] + len(multi_cols)  # New features added by get_dummies
        print(f"One-hot encoding added {new_features} new features from {len(multi_cols)} categorical columns.")

    # === STEP 6: Data Type Cleanup ===
    # Convert nullable integers (Int64) to standard integers for XGBoost
    for col in binary_cols:
        if pd.api.types.is_integer_dtype(df[col]):
            # Fill any NaN values with 0 and convert to int
            df[col] = df[col].fillna(0).astype(int)

    print(f"Completed feature engineering. Final dataset has {df.shape[1]} features.")
    return df