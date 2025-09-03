from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import json
import os
import time
import logging
import sys
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
import sparknlp

# ========== LOGGING CONFIGURATION ==========

def setup_logging():
    """Configure logging with multiple handlers and formatters"""
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create custom logger
    logger = logging.getLogger("contract_api")
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate logs if logger already exists
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (for development)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # File handler for all logs
    file_handler = logging.FileHandler("logs/contract_api.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    
    # File handler for errors only
    error_handler = logging.FileHandler("logs/contract_api_errors.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# ========== REQUEST LOGGING MIDDLEWARE ==========

class RequestLoggingMiddleware:
    """Middleware to log all HTTP requests and responses"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Extract request info
        request = Request(scope, receive)
        start_time = time.time()
        
        # Log incoming request
        logger.info(f" INCOMING REQUEST | {request.method} {request.url.path} | Client: {request.client.host}")
        
        # Process request
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code = message["status"]
                processing_time = round((time.time() - start_time) * 1000, 2)
                
                # Log response
                status_emoji = "🟢" if status_code < 400 else "🔴" if status_code >= 500 else "🟡"
                logger.info(f"{status_emoji} RESPONSE | {request.method} {request.url.path} | Status: {status_code} | Time: {processing_time}ms")
                
            await send(message)
        
        await self.app(scope, receive, send_wrapper)

# ========== FASTAPI APPLICATION ==========

# Initialize FastAPI
app = FastAPI(
    title="Contract Classification API",
    description="API for classifying contract text using Spark NLP + TF-IDF + Logistic Regression",
    version="1.0.0",
)

# Add logging middleware
app.add_middleware(RequestLoggingMiddleware)

# ========== MODEL LOADING ==========

def load_models():
    """Load all models with proper error handling and logging"""
    logger.info(" Starting model loading process...")
    
    # Model paths
    MODEL_DIR = "models"
    CLASSICAL_NLP_PATH = os.path.join(MODEL_DIR, "classical_nlp_pipeline")
    FEATURE_PIPELINE_PATH = os.path.join(MODEL_DIR, "feature_pipeline")
    LR_MODEL_PATH = os.path.join(MODEL_DIR, "lr_model")
    LABEL_MAPPING_PATH = os.path.join(MODEL_DIR, "label_mappings.json")
    
    models = {}
    
    try:
        # Initialize Spark session
        logger.info("🔧 Initializing Spark session...")
        spark = sparknlp.start()
        models["spark"] = spark
        logger.info(" Spark session initialized successfully")
        
        # Load classical NLP pipeline
        logger.info(f" Loading classical NLP pipeline from {CLASSICAL_NLP_PATH}")
        models["classical_nlp_pipeline"] = PipelineModel.load(CLASSICAL_NLP_PATH)
        logger.info(" Classical NLP pipeline loaded successfully")
        
        # Load feature pipeline
        logger.info(f" Loading feature pipeline from {FEATURE_PIPELINE_PATH}")
        models["feature_pipeline"] = PipelineModel.load(FEATURE_PIPELINE_PATH)
        logger.info(" Feature pipeline loaded successfully")
        
        # Load logistic regression model
        logger.info(f" Loading LR model from {LR_MODEL_PATH}")
        models["lr_model"] = LogisticRegressionModel.load(LR_MODEL_PATH)
        logger.info(" Logistic regression model loaded successfully")
        
        # Load label mappings
        logger.info(f" Loading label mappings from {LABEL_MAPPING_PATH}")
        with open(LABEL_MAPPING_PATH, "r") as f:
            models["label_mappings"] = json.load(f)
        logger.info(f" Label mappings loaded: {list(models['label_mappings'].values())}")
        
        logger.info(" All models loaded successfully!")
        return models
        
    except FileNotFoundError as e:
        logger.error(f" Model file not found: {e}")
        raise HTTPException(status_code=503, detail=f"Model files not found: {e}")
    except Exception as e:
        logger.error(f" Error loading models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Failed to load models: {e}")

# Load models on startup
models = load_models()
spark = models["spark"]
classical_nlp_pipeline = models["classical_nlp_pipeline"]
feature_pipeline = models["feature_pipeline"]
lr_model = models["lr_model"]
label_mappings = models["label_mappings"]

# ========== REQUEST/RESPONSE MODELS ==========

class PredictRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    predicted_type: str
    confidence: float
    probabilities: dict
    processing_time_ms: float
    text_length: int

# ========== UTILITY FUNCTIONS ==========

def preprocess_text(text: str, request_id: str = None):
    """Apply the same preprocessing pipeline as training with detailed logging"""
    
    log_prefix = f"[REQ-{request_id}]" if request_id else ""
    
    try:
        logger.info(f"{log_prefix}  Starting text preprocessing | Length: {len(text)} chars")
        
        # Step 1: Create DataFrame
        logger.debug(f"{log_prefix} Creating DataFrame with training schema")
        input_df = spark.createDataFrame(
            [[text, "unknown", "unknown"]], 
            ["description", "contract", "agreement_type"]
        )
        
        # Step 2: Apply classical NLP pipeline
        logger.debug(f"{log_prefix} Applying classical NLP pipeline (tokenization, normalization, etc.)")
        start_time = time.time()
        tokens_df = classical_nlp_pipeline.transform(input_df)
        nlp_time = round((time.time() - start_time) * 1000, 2)
        logger.debug(f"{log_prefix} Classical NLP completed in {nlp_time}ms")
        
        # Step 3: Filter tokens (length > 2)
        logger.debug(f"{log_prefix} Filtering tokens (length > 2)")
        tokens_df = tokens_df.withColumn(
            "finished_tokens",
            expr("filter(finished_tokens, x -> length(x) > 2)")
        )
        
        # Step 4: Apply feature pipeline
        logger.debug(f"{log_prefix} Applying feature pipeline (n-grams, TF-IDF)")
        start_time = time.time()
        features_df = feature_pipeline.transform(tokens_df)
        feature_time = round((time.time() - start_time) * 1000, 2)
        logger.debug(f"{log_prefix} Feature pipeline completed in {feature_time}ms")
        
        logger.info(f"{log_prefix}  Text preprocessing completed | NLP: {nlp_time}ms, Features: {feature_time}ms")
        return features_df
        
    except Exception as e:
        logger.error(f"{log_prefix}  Preprocessing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

# ========== API ENDPOINTS ==========

@app.get("/")
def read_root():
    logger.info(" Root endpoint accessed")
    return {
        "message": "Welcome to the Contract Classification API",
        "model_type": "Spark NLP + TF-IDF + Logistic Regression",
        "available_labels": list(label_mappings.values()),
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
def health_check():
    logger.info(" Health check requested")
    
    # Check model availability
    models_available = {
        "classical_nlp_pipeline": classical_nlp_pipeline is not None,
        "feature_pipeline": feature_pipeline is not None,
        "lr_model": lr_model is not None,
        "label_mappings": label_mappings is not None,
        "spark_session": spark is not None
    }
    
    all_models_ok = all(models_available.values())
    
    health_status = {
        "status": "healthy" if all_models_ok else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": models_available,
        "available_labels": len(label_mappings) if label_mappings else 0,
        "spark_version": spark.version if spark else None
    }
    
    if all_models_ok:
        logger.info(" Health check passed - all systems operational")
    else:
        logger.warning(f" Health check failed - missing models: {[k for k, v in models_available.items() if not v]}")
    
    return health_status

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictRequest):
    # Generate unique request ID for tracking
    request_id = f"{int(time.time())}-{hash(request.text) % 10000}"
    start_time = time.time()
    
    logger.info(f"[REQ-{request_id}]  Prediction request received | Text length: {len(request.text)} chars")
    
    try:
        # Input validation
        if not request.text or len(request.text.strip()) < 10:
            logger.warning(f"[REQ-{request_id}]  Invalid input: text too short ({len(request.text)} chars)")
            raise HTTPException(status_code=400, detail="Text must be at least 10 characters long")
        
        if len(request.text) > 100000:  # 100KB limit
            logger.warning(f"[REQ-{request_id}]  Invalid input: text too long ({len(request.text)} chars)")
            raise HTTPException(status_code=400, detail="Text is too long (max 100,000 characters)")
        
        # Log text preview (first 100 chars)
        text_preview = request.text.replace('\n', ' ').replace('\r', '')[:100]
        logger.info(f"[REQ-{request_id}]  Text preview: {text_preview}...")
        
        # Preprocess text
        features_df = preprocess_text(request.text, request_id)
        
        # Make prediction
        logger.info(f"[REQ-{request_id}]  Making prediction...")
        pred_start = time.time()
        predictions_df = lr_model.transform(features_df)
        pred_time = round((time.time() - pred_start) * 1000, 2)
        
        # Extract results
        logger.debug(f"[REQ-{request_id}] Extracting prediction results")
        result = predictions_df.select("prediction", "probability").collect()[0]
        
        # Process results
        prediction_idx = str(int(result["prediction"]))
        predicted_label = label_mappings.get(prediction_idx, f"UNKNOWN_{prediction_idx}")
        
        # Extract probabilities
        prob_array = result["probability"].toArray()
        confidence = float(max(prob_array))
        
        probabilities = {
            label_mappings.get(str(i), f"LABEL_{i}"): float(prob)
            for i, prob in enumerate(prob_array)
        }
        
        # Calculate total processing time
        total_time = round((time.time() - start_time) * 1000, 2)
        
        # Log successful prediction
        logger.info(f"[REQ-{request_id}]  Prediction completed | Result: {predicted_label} | Confidence: {confidence:.3f} | Total time: {total_time}ms")
        
        # Log detailed probabilities
        top_3_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
        logger.info(f"[REQ-{request_id}]  Top predictions: {', '.join([f'{label}: {prob:.3f}' for label, prob in top_3_probs])}")
        
        return PredictionResponse(
            predicted_type=predicted_label,
            confidence=confidence,
            probabilities=probabilities,
            processing_time_ms=total_time,
            text_length=len(request.text)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        total_time = round((time.time() - start_time) * 1000, 2)
        logger.error(f"[REQ-{request_id}]  Prediction failed after {total_time}ms: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model-info")
def model_info():
    logger.info(" Model info requested")
    
    return {
        "model_type": "Logistic Regression",
        "features": "TF-IDF (unigrams + bigrams)",
        "preprocessing_steps": [
            "document_assembly",
            "tokenization", 
            "normalization",
            "stemming", 
            "stopword_removal",
            "token_filtering",
            "n-gram_generation", 
            "count_vectorization",
            "tf-idf_transformation"
        ],
        "labels": label_mappings,
        "num_classes": len(label_mappings),
        "spark_version": spark.version if spark else None,
        "model_loaded_at": datetime.now().isoformat()
    }

@app.get("/logs")
def get_recent_logs():
    """Get recent log entries (last 100 lines)"""
    logger.info(" Recent logs requested")
    
    try:
        log_file = "logs/contract_api.log"
        if not os.path.exists(log_file):
            return {"message": "No log file found"}
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Return last 100 lines
        recent_logs = lines[-100:] if len(lines) > 100 else lines
        
        return {
            "total_lines": len(lines),
            "recent_lines": len(recent_logs),
            "logs": [line.strip() for line in recent_logs]
        }
        
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        return {"error": f"Could not read logs: {e}"}

# ========== STARTUP/SHUTDOWN EVENTS ==========

@app.on_event("startup")
async def startup_event():
    logger.info(" CONTRACT CLASSIFICATION API STARTING UP")
    logger.info(f" Startup time: {datetime.now().isoformat()}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info(" CONTRACT CLASSIFICATION API SHUTTING DOWN")
    if spark:
        logger.info("🔧 Stopping Spark session...")
        spark.stop()
        logger.info(" Spark session stopped")
    logger.info(f" Shutdown time: {datetime.now().isoformat()}")

# ========== MAIN ==========

if __name__ == "__main__":
    import uvicorn
    
    logger.info(" Starting FastAPI server...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s | UVICORN | %(levelname)s | %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )