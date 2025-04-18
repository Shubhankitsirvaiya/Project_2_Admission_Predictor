from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine
import pandas as pd
import joblib
from app.retrain_model import check_and_retrain

# Initialize FastAPI
app = FastAPI()

# Load trained model
model = joblib.load("model.pkl")

# Azure PostgreSQL connection
engine = create_engine(
    "postgresql://pgadmin:MyPass06@ml-pipeline-pg-server.postgres.database.azure.com:5432/ml_pipeline_db"
)

# Input schema using snake_case
class AdmissionInput(BaseModel):
    GRE: float
    TOEFL: float
    University_Rating: float
    SOP: float
    LOR: float
    CGPA: float
    Research: int

@app.post("/predict")
def predict(data: AdmissionInput):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data.dict()])

        # Predict using model
        prediction = model.predict(input_df)[0]
        input_df["Predicted_Probability"] = prediction

        # Insert into Azure DB
        input_df.to_sql("admission_data", engine, if_exists="append", index=False)

        # Call retrain logic
        check_and_retrain()

        return {"prediction": round(prediction, 4)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
