import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

def train_model(df: pd.DataFrame, target_col: str):
    """
    Trains an XGBoost model and logs with MLflow.

    Args:
        df (pd.DataFrame): Feature dataset.
        target_col (str): Name of the target column.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    with mlflow.start_run():
        # Train the model
        model.fit(X_train, y_train)
        # Predict and evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        # Log parameters and metrics
        mlflow.log_params({
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1
        })
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        # Log the model
        mlflow.xgboost.log_model(model, "xgb_model")
        # Log dataset so it shows up in MLflow UI for reference
        train_data = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_data(train_data, context="training")

        print(f"Model trained with accuracy: {acc:.4f} and recall: {rec:.4f}.")