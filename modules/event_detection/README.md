# Event Detection Module (module-2)

This module provides an API for detecting emerging events and trending topics from a stream of text data.

## Features

- **Topic Detection**: Uses sentence embeddings and clustering to group related documents into events.
- **Content Filtering**: Filters content based on sentiment analysis to remove irrelevant or low-quality data.
- **Event Alerting**: Identifies significant clusters and flags them as events.

## How to Run

1.  **Install Dependencies**: Make sure you have Python 3.9+ installed. Then, install the required packages:

    ```bash
    pip install -r requirements.txt
    ```
    (Note: You'll need to generate a `requirements.txt` from `pyproject.toml` if you prefer that over using a tool like Poetry or PDM).

2.  **Run the API Server**:

    ```bash
    uvicorn main:app --reload --port 8001
    ```

3.  **Send a Request**:

    You can send a POST request to the `/process/` endpoint with a list of texts:

    ```bash
    curl -X POST "http://127.0.0.1:8001/process/" -H "Content-Type: application/json" -d '{
      "texts": [
        "Massive solar flare expected to hit Earth tomorrow.",
        "New study shows coffee can improve memory.",
        "Scientists are amazed by the recent solar activity."
      ]
    }'
    ```