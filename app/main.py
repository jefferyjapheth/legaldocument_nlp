import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
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

# --------- Configure Logging ---------
# Logs include timestamp, log level, logger name, and message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("contract-classifier")

# --------- Initialize FastAPI ---------
app = FastAPI(
    title="Contract Classification API",
    description="API for classifying contract text using Spark NLP + Logistic Regression with fallback, OCR, explainability, and logging",
    version="1.5.0",
)

# --------- Load Spark NLP & Models ---------
try:
    spark = sparknlp.start()
    logger.info("Spark NLP session started successfully.")
except Exception as e:
    logger.exception("Failed to start Spark NLP session.")
    raise RuntimeError("Spark NLP initialization failed.") from e

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
FULL_PIPELINE_PATH = os.path.join(MODEL_DIR, "full_pipeline")
LABEL_MAPPING_PATH = os.path.join(MODEL_DIR, "label_mappings.json")

try:
    full_pipeline = PipelineModel.load(FULL_PIPELINE_PATH)
    logger.info(f"Loaded Spark NLP pipeline from {FULL_PIPELINE_PATH}")
except Exception as e:
    logger.exception("Failed to load trained Spark NLP pipeline.")
    raise RuntimeError("Pipeline loading failed.") from e

try:
    with open(LABEL_MAPPING_PATH, "r") as f:
        label_mappings = json.load(f)
    logger.info("Loaded label mappings.")
except Exception as e:
    logger.exception("Failed to load label mappings.")
    raise RuntimeError("Label mappings loading failed.") from e

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
def apply_fallback_api(prediction_probs: list[float], predicted_idx: int) -> tuple[str, float]:
    """
    Apply fallback logic when confidence is low or class is ambiguous.
    """
    sorted_probs = sorted(prediction_probs)
    top1_prob = sorted_probs[-1]
    top2_prob = sorted_probs[-2] if len(sorted_probs) > 1 else 0.0
    margin = top1_prob - top2_prob

    if (predicted_idx in AMBIG_CLASSES and (top1_prob < CONF_THRESHOLD_AMBIG or margin < MARGIN_AMBIG)) \
        or (predicted_idx not in AMBIG_CLASSES and (top1_prob < CONF_THRESHOLD_OTHERS or margin < MARGIN_OTHERS)):
        fallback_conf = max(prediction_probs)
        logger.warning(f"Fallback applied for index={predicted_idx}, probs={prediction_probs}")
        return FALLBACK_LABEL, fallback_conf
    else:
        return label_mappings[str(predicted_idx)], top1_prob

# --------- Run Prediction ---------
def run_prediction(text: str, chunk_size: int = CHUNK_SIZE) -> dict:
    """
    Splits text into chunks, runs classification with Spark NLP,
    aggregates results, and logs structured request/response info.
    """
    try:
        # Normalize text and split into chunks
        text = " ".join(text.split())
        doc_len = len(text)
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        num_chunks = len(chunks)
        agg_probs = [0.0] * len(label_mappings)
        fallback_chunks = []
        chunk_details = []

        logger.info(f"Running prediction: doc_length={doc_len}, num_chunks={num_chunks}")

        # Process each chunk separately
        for idx, chunk in enumerate(chunks):
            df_chunk = spark.createDataFrame([[chunk]]).toDF("description")
            pred = full_pipeline.transform(df_chunk).collect()[0]

            chunk_probs = [float(p) for p in pred.probability]
            top_idx = chunk_probs.index(max(chunk_probs))
            top_label, top_conf = apply_fallback_api(chunk_probs, top_idx)

            if top_label == FALLBACK_LABEL:
                fallback_chunks.append(idx)

            # Aggregate probabilities
            agg_probs = [agg_probs[i] + chunk_probs[i] for i in range(len(agg_probs))]

            # Save chunk-level explainability
            chunk_details.append({
                "chunk_index": idx,
                "text_snippet": chunk[:200] + ("..." if len(chunk) > 200 else ""),
                "predicted_label": top_label,
                "confidence": top_conf,
                "probabilities": {label_mappings[str(i)]: p for i, p in enumerate(chunk_probs)}
            })

        # Average probabilities across chunks
        agg_probs = [p / num_chunks for p in agg_probs]
        top_idx = agg_probs.index(max(agg_probs))
        top_label, top_conf = apply_fallback_api(agg_probs, top_idx)

        # Get top-3 predictions
        top3 = sorted(
            [(label_mappings[str(i)], p) for i, p in enumerate(agg_probs)],
            key=lambda x: x[1], reverse=True
        )[:3]

        result = {
            "predicted_type": top_label,
            "confidence": top_conf,
            "top3_predictions": [{"label": label, "probability": prob} for label, prob in top3],
            "probabilities": {label_mappings[str(i)]: p for i, p in enumerate(agg_probs)},
            "document_length": doc_len,
            "num_chunks": num_chunks,
            "fallback_chunks": fallback_chunks,
            "chunk_details": chunk_details
        }

        # Log structured response summary
        logger.info(f"Prediction completed: label={top_label}, confidence={top_conf:.2f}, doc_length={doc_len}")

        return result
    except Exception:
        logger.exception("Error during prediction.")
        raise

# --------- OCR for PDFs ---------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts text from PDF using PyPDF2. Falls back to OCR if needed.
    """
    try:
        text = ""
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "

        if not text.strip():
            logger.info("No text found in PDF, falling back to OCR.")
            images = convert_from_bytes(file_bytes)
            for img in images:
                text += pytesseract.image_to_string(img) + " "

        return text.strip()
    except Exception :
        logger.exception("Failed to extract text from PDF.")
        raise

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
    Endpoint for raw text classification with logging.
    """
    try:
        logger.info(f"/predict request received: text_length={len(request.text)}")
        result = run_prediction(request.text)
        logger.info(f"/predict response: predicted_type={result['predicted_type']}, confidence={result['confidence']:.2f}")
        return result
    except Exception:
        logger.exception("/predict failed.")
        raise HTTPException(status_code=500, detail="Prediction failed.")

@app.post("/predict-file", response_model=dict, summary="Predict from uploaded file (PDF, TXT, HTML)")
def predict_file(file: UploadFile = File(...)):
    """
    Endpoint for file classification with OCR + logging.
    """
    try:
        logger.info(f"/predict-file request: filename={file.filename}")
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
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

        if not text.strip():
            raise HTTPException(status_code=400, detail="No extractable text found in file.")

        result = run_prediction(text)
        logger.info(f"/predict-file response: predicted_type={result['predicted_type']}, confidence={result['confidence']:.2f}")
        return result
    except HTTPException:
        raise
    except Exception:
        logger.exception("/predict-file failed.")
        raise HTTPException(status_code=500, detail="Failed to process file.")

@app.get("/model-info")
def model_info():
    return {
        "labels": label_mappings,
        "pipeline": "Full Pipeline (NLP + Features + Logistic Regression with fallback, OCR, explainability, logging)"
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
