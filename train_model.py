import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Connect to Azure PostgreSQL
engine = create_engine(
    "postgresql://pgadmin:MyPass06@ml-pipeline-pg-server.postgres.database.azure.com:5432/ml_pipeline_db"
)

# Load data from DB
query = "SELECT * FROM admission_data"
df = pd.read_sql(query, engine)

# Confirm column names
print("📊 Columns in DB:", df.columns.tolist())

# Train/test split
X = df.drop("Predicted_Probability", axis=1)
y = df["Predicted_Probability"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "model.pkl")
print("✅ Model trained on DB data and saved as model.pkl")


import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("file:./mlruns")

with mlflow.start_run():
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_metric("score", model.score(X_test, y_test))
