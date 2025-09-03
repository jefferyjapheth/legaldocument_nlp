import logging
import logging.config
import os
from datetime import datetime

def setup_logging(log_level: str = "INFO"):
    """
    Set up comprehensive logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Generate log filename with date
    log_date = datetime.now().strftime("%Y-%m-%d")
    
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "simple": {
                "format": "%(asctime)s | %(levelname)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "uvicorn": {
                "format": "%(asctime)s | UVICORN | %(levelname)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            },
            "file_all": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "detailed",
                "filename": f"logs/contract_api_{log_date}.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8"
            },
            "file_errors": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": f"logs/contract_api_errors_{log_date}.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8"
            },
            "file_predictions": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "detailed",
                "filename": f"logs/predictions_{log_date}.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 10,
                "encoding": "utf8"
            }
        },
        "loggers": {
            "contract_api": {
                "level": log_level,
                "handlers": ["console", "file_all", "file_errors"],
                "propagate": False
            },
            "predictions": {
                "level": "INFO",
                "handlers": ["file_predictions"],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False
            },
            "uvicorn.error": {
                "level": "INFO",
                "handlers": ["console", "file_errors"],
                "propagate": False
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False
            }
        },
        "root": {
            "level": log_level,
            "handlers": ["console", "file_all"]
        }
    }
    
    logging.config.dictConfig(logging_config)
    
    # Create specialized loggers
    api_logger = logging.getLogger("contract_api")
    prediction_logger = logging.getLogger("predictions")
    
    api_logger.info(" Logging system initialized")
    api_logger.info(f" Log level set to: {log_level}")
    api_logger.info(f" Log files created in: logs/")
    
    return api_logger, prediction_logger

class PredictionLogger:
    """Specialized logger for prediction events"""
    
    def __init__(self):
        self.logger = logging.getLogger("predictions")
    
    def log_prediction(self, request_id: str, text_length: int, predicted_type: str, 
                      confidence: float, processing_time: float, client_ip: str = None):
        """Log prediction event in structured format"""
        
        log_data = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "text_length": text_length,
            "predicted_type": predicted_type,
            "confidence": round(confidence, 4),
            "processing_time_ms": processing_time,
            "client_ip": client_ip or "unknown"
        }
        
        self.logger.info(f"PREDICTION | {log_data}")
    
    def log_error(self, request_id: str, error_type: str, error_message: str, 
                  text_length: int = None, processing_time: float = None):
        """Log prediction error in structured format"""
        
        log_data = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "text_length": text_length,
            "processing_time_ms": processing_time
        }
        
        self.logger.error(f"PREDICTION_ERROR | {log_data}")

def get_loggers():
    """Get the configured loggers"""
    return (
        logging.getLogger("contract_api"),
        PredictionLogger()
    )