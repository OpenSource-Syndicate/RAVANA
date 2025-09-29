import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from event_detector import process_data_for_events, load_models

# ---------- Logging Setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("EventDetectionAPI")

# ---------- FastAPI App ----------
app = FastAPI(
    title="Event Detection API",
    description="API for detecting emerging events from text data.",
    version="0.1.0"
)


class ProcessRequest(BaseModel):
    texts: List[str]


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    try:
        logger.info("Starting up... Loading models.")
        load_models()
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.exception("Error loading models during startup.")
        raise


@app.post("/process/", response_model=Dict[str, Any])
async def process_texts(request: ProcessRequest):
    """
    Process a list of texts to detect events.
    """
    logger.info(f"Received request to process {len(request.texts)} texts.")

    if not request.texts:
        logger.warning("No texts provided in request.")
        raise HTTPException(status_code=400, detail="No texts provided.")

    try:
        results = process_data_for_events(request.texts)
        logger.info("Event detection completed successfully.")
        return results
    except Exception as e:
        logger.exception("Error while processing texts for event detection.")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Running development server with Uvicorn.")
    uvicorn.run(app, host="0.0.0.0", port=8001)
