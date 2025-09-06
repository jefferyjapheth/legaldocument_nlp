
---

# Contract Classification API – Startup Guide (with Conda)

## 0. Installing Conda (if not already installed)

If you don’t have Conda installed, download and install **Miniconda**:

1. Visit the official Miniconda download page: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
2. Choose the installer for your operating system (Windows, macOS, or Linux).
3. Follow the instructions on the page to complete the installation.

Verify installation:

```bash
conda --version
```

---

## 1. Prerequisites

* **Python 3.10+**
* System dependencies for OCR:

  * **Tesseract** (for `pytesseract`)

    * Linux: `sudo apt install tesseract-ocr`
    * macOS: `brew install tesseract`
    * Windows: download installer and add to PATH
  * **Poppler** (for `pdf2image`)

    * Linux: `sudo apt install poppler-utils`
    * macOS: `brew install poppler`
    * Windows: download binaries and add to PATH

---

## 2. Creating the Conda Environment

From your project directory:

```bash
conda env create -f environment.yml
conda activate contract-classification
```

Verify the environment:

```bash
conda list
```

---

## 3. Project Structure

```
project/
│── app/
│    └── main.py
│── models/
│    ├── full_pipeline/
│    ├── label_mappings.json
│── environment.yml
│── sample_data/
│    ├── sample_contract.pdf
│    ├── sample_contract.txt
│    └── sample_contract.html
```

---

## 4. Running the API Server
* Run this command from the root of the project dir

```bash
uvicorn app.main:app --reload
```

* `app.main` → path to your `main.py`
* `app` → FastAPI application instance

---

## 5. API Verification

Logs should show:

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

---

## 6. Accessing API Documentation

* Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* OpenAPI JSON: [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

---

## 7. Endpoints

* `GET /` → Welcome message
* `GET /health` → Health check
* `POST /predict` → Predict contract type from raw text
* `POST /predict-file` → Predict contract type from PDF/TXT/HTML
* `GET /model-info` → View labels & pipeline info

---

## 8. Example Requests

### Raw Text

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "This Employment Agreement is made between Company X and John Doe..."}'
```

### File Upload (PDF/TXT/HTML)

```bash
curl -X POST http://127.0.0.1:8000/predict-file \
  -F 'file=@sample_data/sample_contract.pdf'
```

**Example API Response**

```json
{
  "predicted_type": "employment",
  "confidence": 0.87,
  "top3_predictions": [
    {
      "label": "employment",
      "probability": 0.87
    },
    {
      "label": "security",
      "probability": 0.08
    },
    {
      "label": "purchase&ma",
      "probability": 0.05
    }
  ],
  "probabilities": {
    "employment": 0.87,
    "security": 0.08,
    "purchase&ma": 0.05,
    "services&supply": 0.0,
    "shareholder": 0.0,
    "other": 0.0,
    "lease": 0.0,
    "na": 0.0
  },
  "document_length": 5230,
  "num_chunks": 2,
  "fallback_chunks": [
    1
  ]
}
```

> **Tip:** The `na` label appears when the model is unsure, using your fallback logic.
---

## 9. Stop the Server

Press **CTRL+C** in the terminal.

---

## 10. Sample Data for Testing

To quickly test the API, the project includes sample files:

```
project/
│── sample_data/
│    ├── sample_contract.pdf
│    ├── sample_contract.txt
│    └── sample_contract.html
```

Upload these files via the `/predict-file` endpoint to see predictions.

**Full dataset for research/testing:** [Stanford Contract Dataset](https://mcc.law.stanford.edu/download/contracts/)

---

## 11. Prediction Labels and Mapping

The API returns predictions as **labels** with confidence scores. The mapping of `type_label` → human-readable contract type is:

| Label Name (`type_label`) | Agreement Type    | Description/Notes                  |
| ------------------------- | ----------------- | ---------------------------------- |
| `LABEL_0`                 | Security          | Contracts related to securities    |
| `LABEL_1`                 | Employment        | Employment agreements              |
| `LABEL_2`                 | Lease             | Lease or rental contracts          |
| `LABEL_3`                 | Services & Supply | Service or supply agreements       |
| `LABEL_4`                 | Purchase & M\&A   | Purchase or merger/acquisition     |
| `LABEL_5`                 | Shareholder       | Shareholder agreements             |
| `LABEL_6`                 | Other             | Miscellaneous/other contracts      |
| `LABEL_7`                 | na                | Fallback label for uncertain cases |

---

### 12.3 Accessing Labels Programmatically

You can get the label mapping via the API:

```bash
curl -X GET http://127.0.0.1:8000/model-info
```

