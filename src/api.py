"""
Production API for sentiment analysis
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import json
import logging
from datetime import datetime
from pathlib import Path
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
prediction_counter = Counter('sentiment_predictions_total', 'Total predictions', ['sentiment'])
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction latency')
api_requests = Counter('api_requests_total', 'Total API requests', ['endpoint', 'status'])

# Initialize FastAPI
app = FastAPI(
    title="Sentiment Analysis API",
    description="Production sentiment analysis with MLOps pipeline",
    version="2.0.0"
)

# Load model
MODEL_PATH = "models/sentiment_model"

try:
    logger.info(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    
    # Create pipeline for easier inference
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Load metrics
    with open(Path(MODEL_PATH) / 'metrics.json', 'r') as f:
        model_metrics = json.load(f)
    
    logger.info("✅ Model loaded successfully")
    logger.info(f"Test Accuracy: {model_metrics['test_accuracy']:.4f}")
    
    # Initialize W&B for inference logging (optional)
    try:
        wandb.init(
            project="sentiment-analysis",
            job_type="inference",
            config=model_metrics
        )
    except:
        logger.warning("W&B initialization failed, continuing without it")
    
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    raise

# Request/Response models
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    scores: Dict[str, float]
    processing_time_ms: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_accuracy: float
    uptime_seconds: float

start_time = time.time()

# Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    
    api_requests.labels(
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    response.headers["X-Process-Time"] = str(duration)
    return response

# Endpoints
@app.get("/")
def root():
    return {
        "service": "Sentiment Analysis API",
        "version": "2.0.0",
        "model": "DistilBERT",
        "accuracy": f"{model_metrics['test_accuracy']*100:.2f}%",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "health": "/health",
            "metrics": "/metrics",
            "model_info": "/model-info"
        }
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    uptime = time.time() - start_time
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_accuracy": model_metrics['test_accuracy'],
        "uptime_seconds": uptime
    }

@app.get("/model-info")
def model_info():
    """Get model information"""
    return {
        "model_name": model_metrics['model_name'],
        "test_accuracy": model_metrics['test_accuracy'],
        "test_f1": model_metrics['test_f1'],
        "training_date": model_metrics['training_date'],
        "train_samples": model_metrics['train_samples'],
        "test_samples": model_metrics['test_samples']
    }

@app.post("/predict", response_model=SentimentResponse)
def predict(input: TextInput):
    """Predict sentiment for single text"""
    start_pred = time.time()
    
    try:
        with prediction_duration.time():
            # Get prediction
            result = sentiment_pipeline(input.text)[0]
            
            # Map label to sentiment
            sentiment = "positive" if result['label'] == 'LABEL_1' else "negative"
            confidence = result['score']
            
            # Calculate both scores
            pos_score = confidence if sentiment == "positive" else 1 - confidence
            neg_score = 1 - pos_score
            
            # Update metrics
            prediction_counter.labels(sentiment=sentiment).inc()
            
            # Log to W&B (optional)
            try:
                wandb.log({
                    "prediction": sentiment,
                    "confidence": confidence,
                    "text_length": len(input.text)
                })
            except:
                pass
            
            processing_time = (time.time() - start_pred) * 1000
            
            logger.info(f"Prediction: {sentiment} (confidence: {confidence:.4f})")
            
            return {
                "text": input.text[:100],  # Truncate for response
                "sentiment": sentiment,
                "confidence": confidence,
                "scores": {
                    "positive": pos_score,
                    "negative": neg_score
                },
                "processing_time_ms": processing_time,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict")
def batch_predict(input: BatchTextInput):
    """Predict sentiment for multiple texts"""
    start_batch = time.time()
    
    try:
        results = []
        
        for text in input.texts:
            if not text.strip():
                results.append({"error": "Empty text"})
                continue
            
            result = sentiment_pipeline(text)[0]
            sentiment = "positive" if result['label'] == 'LABEL_1' else "negative"
            
            prediction_counter.labels(sentiment=sentiment).inc()
            
            results.append({
                "text": text[:50],
                "sentiment": sentiment,
                "confidence": result['score']
            })
        
        processing_time = (time.time() - start_batch) * 1000
        
        return {
            "predictions": results,
            "count": len(results),
            "processing_time_ms": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    """Prometheus metrics"""
    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000))
    )
