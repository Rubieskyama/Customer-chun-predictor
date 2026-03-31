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