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
from typing import List
from pdf2image import convert_from_bytes
import pytesseract

# --------- Initialize FastAPI ---------
app = FastAPI(
    title="Contract Classification API",
    description="API for classifying contract text using Spark NLP + Logistic Regression with fallback and OCR handling",
    version="1.3.0",
)

# --------- Load Spark NLP & Models ---------
spark = sparknlp.start()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
FULL_PIPELINE_PATH = os.path.join(MODEL_DIR, "full_pipeline")
LABEL_MAPPING_PATH = os.path.join(MODEL_DIR, "label_mappings.json")

# Load trained pipeline
full_pipeline = PipelineModel.load(FULL_PIPELINE_PATH)

# Load label mappings
with open(LABEL_MAPPING_PATH, "r") as f:
    label_mappings = json.load(f)

# --------- Fallback Configuration ---------
FALLBACK_LABEL = "na"
AMBIG_CLASSES = [0, 3, 4, 6]
CONF_THRESHOLD_AMBIG = 0.5
CONF_THRESHOLD_OTHERS = 0.2
MARGIN_AMBIG = 0.10
MARGIN_OTHERS = 0.05
CHUNK_SIZE = 3000  # characters per chunk for long documents

# --------- Input Schema ---------
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        example="This Employment Agreement is made between Company X and John Doe..."
    )

# --------- Fallback Logic ---------
def apply_fallback_api(prediction_probs: List[float], predicted_idx: int) -> (str, float):
    """
    Apply fallback logic to a single prediction.
    Returns label and confidence.
    """
    sorted_probs = sorted(prediction_probs)
    top1_prob = sorted_probs[-1]
    top2_prob = sorted_probs[-2] if len(sorted_probs) > 1 else 0.0
    margin = top1_prob - top2_prob

    if (predicted_idx in AMBIG_CLASSES and (top1_prob < CONF_THRESHOLD_AMBIG or margin < MARGIN_AMBIG)) \
        or (predicted_idx not in AMBIG_CLASSES and (top1_prob < CONF_THRESHOLD_OTHERS or margin < MARGIN_OTHERS)):
        fallback_conf = max(prediction_probs)
        return FALLBACK_LABEL, fallback_conf
    else:
        return label_mappings[str(predicted_idx)], top1_prob

# --------- Run Prediction with Chunking & Logging ---------
def run_prediction(text: str, chunk_size: int = CHUNK_SIZE) -> dict:
    """
    Process text: split into chunks, run through Spark NLP pipeline,
    apply fallback logic, and return top prediction, top-3 probabilities,
    chunk info, and fallback triggers.
    """
    text = " ".join(text.split())
    doc_len = len(text)
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    num_chunks = len(chunks)
    agg_probs = [0.0] * len(label_mappings)
    fallback_chunks = []

    # Process each chunk
    for idx, chunk in enumerate(chunks):
        df_chunk = spark.createDataFrame([[chunk]]).toDF("description")
        pred = full_pipeline.transform(df_chunk).collect()[0]

        chunk_probs = [float(p) for p in pred.probability]
        top_idx = chunk_probs.index(max(chunk_probs))
        top_label, top_conf = apply_fallback_api(chunk_probs, top_idx)

        if top_label == FALLBACK_LABEL:
            fallback_chunks.append(idx)

        # Aggregate probabilities across chunks
        agg_probs = [agg_probs[i] + chunk_probs[i] for i in range(len(agg_probs))]

    # Average probabilities
    agg_probs = [p / num_chunks for p in agg_probs]
    top_idx = agg_probs.index(max(agg_probs))
    top_label, top_conf = apply_fallback_api(agg_probs, top_idx)

    # Top 3 predictions
    top3 = sorted(
        [(label_mappings[str(i)], p) for i, p in enumerate(agg_probs)],
        key=lambda x: x[1], reverse=True
    )[:3]

    return {
        "predicted_type": top_label,
        "confidence": top_conf,
        "top3_predictions": [{"label": label, "probability": prob} for label, prob in top3],
        "probabilities": {label_mappings[str(i)]: p for i, p in enumerate(agg_probs)},
        "document_length": doc_len,
        "num_chunks": num_chunks,
        "fallback_chunks": fallback_chunks,
    }

# --------- OCR for PDFs ---------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from PDF using PyPDF2 first, then fall back to OCR if needed.
    """
    text = ""
    pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "

    # If no text extracted, apply OCR
    if not text.strip():
        images = convert_from_bytes(file_bytes)
        for img in images:
            text += pytesseract.image_to_string(img) + " "

    return text.strip()

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
    Predict contract type from raw text input.
    """
    return run_prediction(request.text)

@app.post("/predict-file", response_model=dict, summary="Predict from uploaded file (PDF, TXT, HTML)")
def predict_file(file: UploadFile = File(...)):
    """
    This endpoint can extract text from file (PDF with OCR if needed, TXT, or HTML),
    then run classification with chunking and fallback.
    """
    try:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext == ".pdf":
            text = extract_text_from_pdf(file.file.read())
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
        "pipeline": "Full Pipeline (NLP + Features + Logistic Regression with fallback and OCR)"
    }

# --------- Custom OpenAPI ---------
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    if "/predict-file" in openapi_schema["paths"]:
        openapi_schema["paths"]["/predict-file"]["post"]["requestBody"] = {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string", "format": "binary", "description": "Upload a contract (PDF, TXT, HTML)"}
                        }
                    },
                    "examples": {
                        "samplePDF": {"summary": "Sample PDF contract", "description": "Upload a sample employment agreement in PDF format", "value": {"file": "sample_contract.pdf"}},
                        "sampleTXT": {"summary": "Sample TXT contract", "description": "Upload a plain text contract", "value": {"file": "sample_contract.txt"}},
                        "sampleHTML": {"summary": "Sample HTML contract", "description": "Upload a contract embedded in HTML", "value": {"file": "sample_contract.html"}},
                    }
                }
            },
            "required": True,
        }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
