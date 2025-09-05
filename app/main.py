from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
import json
import os
import sparknlp
from pyspark.ml import PipelineModel
from io import BytesIO
import PyPDF2
from bs4 import BeautifulSoup

# --------- Initialize FastAPI ---------
app = FastAPI(
    title="Contract Classification API",
    description="API for classifying contract text using Spark NLP + Logistic Regression",
    version="1.0.0",
)

# --------- Load Models ---------
spark = sparknlp.start()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

FULL_PIPELINE_PATH = os.path.join(MODEL_DIR, "full_pipeline")
LABEL_MAPPING_PATH = os.path.join(MODEL_DIR, "label_mappings.json")

# Load full pipeline
full_pipeline = PipelineModel.load(FULL_PIPELINE_PATH)

# Load label mappings
with open(LABEL_MAPPING_PATH, "r") as f:
    label_mappings = json.load(f)


# --------- Input Schema ---------
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        example="This Employment Agreement is made between Company X and John Doe..."
    )


# --------- Helper: Run prediction ---------
def run_prediction(text: str):
    # Convert input into Spark DataFrame
    data = spark.createDataFrame([[text]]).toDF("description")

    # Run through pipeline
    predictions = full_pipeline.transform(data)

    # Extract prediction and probability
    result = predictions.collect()[0]
    prediction_idx = int(result.prediction)
    prediction_label = label_mappings[str(prediction_idx)]
    probabilities = {
        label_mappings[str(i)]: float(prob)
        for i, prob in enumerate(result.probability)
    }

    return {
        "predicted_type": prediction_label,
        "confidence": float(result.probability[prediction_idx]),
        "probabilities": probabilities,
    }


# --------- Endpoints ---------
@app.get("/")
def read_root():
    return {"message": "Welcome to the Contract Classification API"}


@app.get("/health")
def health_check():
    return {"status": "ok", "pipeline_loaded": True}


@app.post("/predict", response_model=dict, summary="Predict from raw text")
def predict(request: PredictRequest):
    """
    Submit raw contract text and get the predicted contract type.
    """
    return run_prediction(request.text)


@app.post("/predict-file", summary="Predict from file upload (PDF, TXT, HTML)")
def predict_file(file: UploadFile = File(..., description="Upload a PDF, TXT, or HTML file")):
    """
    Upload a contract in PDF, TXT, or HTML format. 
    The text will be extracted and classified.
    """
    try:
        ext = os.path.splitext(file.filename)[1].lower()

        if ext == ".pdf":
            pdf_reader = PyPDF2.PdfReader(BytesIO(file.file.read()))
            text = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        elif ext == ".txt":
            text = file.file.read().decode("utf-8")
        elif ext in [".html", ".htm"]:
            html_content = file.file.read().decode("utf-8")
            soup = BeautifulSoup(html_content, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
        else:
            return JSONResponse(status_code=400, content={"error": f"Unsupported file type: {ext}"})

        if not text.strip():
            return JSONResponse(status_code=400, content={"error": "No extractable text found in file."})

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed to process file: {str(e)}"})

    return run_prediction(text)


@app.get("/model-info")
def model_info():
    return {
        "labels": label_mappings,
        "pipeline": "Full Pipeline (NLP + Features + Logistic Regression)"
    }


# --------- Custom OpenAPI (adds examples for file upload) ---------
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Update /predict-file requestBody
    if "/predict-file" in openapi_schema["paths"]:
        openapi_schema["paths"]["/predict-file"]["post"]["requestBody"] = {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "file": {
                                "type": "string",
                                "format": "binary",
                                "description": "Upload a contract (PDF, TXT, HTML)"
                            }
                        }
                    },
                    "examples": {
                        "samplePDF": {
                            "summary": "Sample PDF contract",
                            "description": "Upload a sample employment agreement in PDF format",
                            "value": {"file": "sample_contract.pdf"}
                        },
                        "sampleTXT": {
                            "summary": "Sample TXT contract",
                            "description": "Upload a plain text contract",
                            "value": {"file": "sample_contract.txt"}
                        },
                        "sampleHTML": {
                            "summary": "Sample HTML contract",
                            "description": "Upload a contract embedded in HTML",
                            "value": {"file": "sample_contract.html"}
                        },
                    }
                }
            },
            "required": True,
        }

    app.openapi_schema = openapi_schema
    return app.openapi_schema
