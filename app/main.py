from fastapi import FastAPI
from pydantic import BaseModel
import json
from pyspark.sql import SparkSession
import sparknlp

# --------- Initialize FastAPI ---------
app = FastAPI(
    title="Contract Classification API",
    description="API for classifying contract text using Spark NLP + ML models",
    version="1.0.0",
)

# --------- Load Models ---------
spark = sparknlp.start()

# TODO: adjust model load paths if needed
classical_nlp_pipeline = sparknlp.pretrained.PipelineModel.load("models/classical_nlp_pipeline")
feature_pipeline = sparknlp.pretrained.PipelineModel.load("models/feature_pipeline")
lr_model = sparknlp.pretrained.PipelineModel.load("models/lr_model")

with open("models/label_mappings.json", "r") as f:
    label_mappings = json.load(f)


# --------- Input Schema ---------
class PredictRequest(BaseModel):
    text: str


# --------- Endpoints ---------

@app.get("/")
def read_root():
    return {"message": "Welcome to the Contract Classification API"}


@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": True}


@app.post("/predict")
def predict(request: PredictRequest):
    text = request.text

    # Convert text into Spark DataFrame
    data = spark.createDataFrame([[text]]).toDF("text")

    # Apply pipelines
    processed = classical_nlp_pipeline.transform(data)
    features = feature_pipeline.transform(processed)
    predictions = lr_model.transform(features)

    # Extract prediction + probabilities
    result = predictions.collect()[0]
    prediction_idx = int(result.prediction)
    prediction_label = label_mappings[str(prediction_idx)]
    probabilities = {label_mappings[str(i)]: float(prob) for i, prob in enumerate(result.probability)}

    return {
        "predicted_type": prediction_label,
        "confidence": float(result.probability[prediction_idx]),
        "probabilities": probabilities,
    }


@app.get("/model-info")
def model_info():
    return {"labels": label_mappings, "model": "Logistic Regression"}
