import pandas as pd
from sqlalchemy import create_engine

# Load and clean data
df = pd.read_csv("data/admission_predict.csv")
df = df.rename(columns={
    'GRE Score': 'GRE',
    'TOEFL Score': 'TOEFL',
    'University Rating': 'University_Rating',
    'SOP': 'SOP',
    'LOR ': 'LOR',
    'CGPA': 'CGPA',
    'Research': 'Research',
    'Chance of Admit ': 'Predicted_Probability'
})
df.drop('Serial No.', axis=1, inplace=True)

# Create DB engine
engine = create_engine(
    "postgresql://pgadmin:MyPass06@ml-pipeline-pg-server.postgres.database.azure.com:5432/ml_pipeline_db"
)

# Upload with cleaned snake_case columns
df.to_sql("admission_data", engine, if_exists="replace", index=False)
print("✅ Data uploaded to Azure DB with snake_case columns.")
