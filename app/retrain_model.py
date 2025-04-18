import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

# Set MLflow to use local directory
mlflow.set_tracking_uri("file:./mlruns")

def check_and_retrain():
    print("🔍 Checking if retraining needed...")

    # Azure DB connection
    engine = create_engine(
        "postgresql://pgadmin:MyPass06@ml-pipeline-pg-server.postgres.database.azure.com:5432/ml_pipeline_db"
    )

    # Load data from DB
    df = pd.read_sql("SELECT * FROM admission_data", engine)

    if len(df) % 10 == 0:
        print("🔁 Retraining model on {} rows...".format(len(df)))

        X = df.drop("Predicted_Probability", axis=1)
        y = df["Predicted_Probability"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Save new model to file
        joblib.dump(model, "model.pkl")
        print("✅ model.pkl updated!")

        # Log to MLflow
        with mlflow.start_run(run_name="auto-retrain") as run:
            mlflow.log_param("rows_used", len(df))
            mlflow.log_metric("score", model.score(X_test, y_test))
            mlflow.sklearn.log_model(model, "model")
            print(f"📦 Model logged in MLflow: run_id={run.info.run_id}")

    else:
        print("⏳ Not enough rows yet. Skipping retrain.")
