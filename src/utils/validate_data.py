import great_expectations as ge
from great_expectations.datasource.fluent import PandasDatasource

def validate_telco_data(df):
    # 1. Get the context
    context = ge.get_context()
    
    # 2. Add the datasource using the 'add_pandas' method directly on the context
    # (If 'sources' fails, this method is usually the fallback)
    try:
        datasource = context.sources.add_pandas(name="my_datasource")
    except AttributeError:
        # Fallback for versions where 'sources' is 'data_sources'
        datasource = context.data_sources.add_pandas(name="my_datasource")

    # 3. Create the asset and validator
    asset = datasource.add_dataframe_asset(name="my_asset")
    validator = asset.get_validator(batch_request=asset.build_batch_request(dataframe=df))
    
    # Now your validator.expect_... calls will work!

    # === SCHEMA VALIDATION ===
    # (Note: We use 'validator' instead of 'ge_df')
    validator.expect_column_to_exist("customerID")
    validator.expect_column_values_to_not_be_null("customerID")
    
    validator.expect_column_to_exist("gender") 
    validator.expect_column_to_exist("Partner")
    validator.expect_column_to_exist("tenure")
    validator.expect_column_to_exist("MonthlyCharges")
    validator.expect_column_to_exist("TotalCharges")
    
    # === BUSINESS LOGIC VALIDATION ===
    validator.expect_column_values_to_be_in_set("gender", ["Male", "Female"])
    validator.expect_column_values_to_be_in_set("Contract", ["Month-to-month", "One year", "Two year"])
    
    # === NUMERIC RANGE VALIDATION ===
    validator.expect_column_values_to_be_between("tenure", min_value=0, max_value=120)
    validator.expect_column_values_to_be_between("MonthlyCharges", min_value=0, max_value=200)

    # === DATA CONSISTENCY CHECKS ===
    # TotalCharges should be >= MonthlyCharges
    validator.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        mostly=0.95
    )
    
    # === RUN VALIDATION & PROCESS RESULTS ===
    # In v1.x, we validate the Validator's current state
    results = validator.validate()
    
    failed_expectations = [
        r["expectation_config"]["expectation_type"] 
        for r in results["results"] if not r["success"]
    ]
    
    if results["success"]:
        print(f"✅ Data validation PASSED")
    else:
        print(f"❌ Data validation FAILED: {len(failed_expectations)} checks failed")
    
    return results["success"], failed_expectations