
---

# Contract Classification API – Startup Guide

## 1. Prerequisites

Ensure the following are installed in your environment:

* **Python 3.10+**
* **pip** (Python package manager)
* Required Python dependencies:

  ```bash
  pip install -r requirements.txt
  ```

  *(Ensure `fastapi`, `uvicorn`, `pyspark`, `sparknlp`, `PyPDF2`, and `python-multipart` are included in `requirements.txt`.)*

---

## 2. Project Structure

The project should be organized as follows:

```
project/
│── app/
│    └── main.py
│── models/
│    ├── full_pipeline/
│    ├── label_mappings.json
│── requirements.txt
```

---

## 3. Running the API Server

From the project root directory, start the API server with:

```bash
uvicorn app.main:app --reload
```

* `app.main` → path to the `main.py` file inside the `app/` folder
* `app` → the FastAPI application instance defined in `main.py`

---

## 4. Verifying the Server

If successful, you will see logs similar to:

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

## 5. Accessing the API Documentation

Once the server is running, open the following URL in your browser:

* **Swagger UI** (interactive documentation):
  [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

* **OpenAPI schema (JSON format):**
  [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

---

## 6. Available Endpoints

* `GET /` → Welcome message
* `GET /health` → Health check (verifies pipeline is loaded)
* `POST /predict` → Classify contract text (raw text input)
* `POST /predict-file` → Classify uploaded contract file (PDF, TXT, or HTML)
* `GET /model-info` → Model labels and pipeline details

---

## 7. Example Requests

### Raw Text Prediction

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "This Employment Agreement is made between Company X and John Doe..."
}'
```

### File Upload Prediction (PDF/TXT/HTML)

Using Swagger UI, upload a file directly.
Or with `curl`:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict-file' \
  -H 'accept: application/json' \
  -F 'file=@sample_contract.pdf'
```

---

## 8. Stopping the Server

Press **CTRL+C** in the terminal to stop the server.

---
