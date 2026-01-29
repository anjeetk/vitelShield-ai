# Sepsis Prediction Backend API

A complete, production-ready Python FastAPI backend that serves pre-trained LightGBM clinical prediction models for sepsis risk assessment.

## Project Overview

This API accepts patient clinical data and performs the exact same preprocessing (MICE imputation and Standard Scaling) used during model training. It runs predictions using LightGBM for horizons of 6h, 12h, 18h, and 24h, and can explain its predictions using SHAP (SHapley Additive exPlanations) values to identify the most significant clinical features contributing to risk.

## Architecture

- **FastAPI**: Provides high-performance, asynchronous REST API endpoints.
- **Pydantic**: Enforces strict schema validation for clinical inputs.
- **LightGBM & SHAP**: Fast inference and explainability.
- **Singleton Model Loader**: Models are loaded into memory exactly once during application startup for high concurrency and low latency.
- **Pandas/Scikit-Learn**: Vectorized batch processing and exact preprocessing replication.

## Installation & Setup

1. **Clone/Navigate to Backend**:
   ```bash
   cd backend
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Setup**:
   Copy `.env.example` to `.env` and adjust the paths if needed.
   ```bash
   cp .env.example .env
   ```

5. **Model Placement**:
   Ensure `trained_models.pkl` and `preprocessing.pkl` are located in the `models/` directory (one level above `backend/`, as configured by default).

## How to Run Locally

Start the Uvicorn server:
```bash
uvicorn app.main:app --reload
```
The API will be available at `http://127.0.0.1:8000`.
Swagger UI documentation is automatically generated at: `http://127.0.0.1:8000/docs`

## Docker Deployment

1. **Build Image**:
   ```bash
   docker build -t sepsis-backend .
   ```

2. **Run Container**:
   Map the models directory into the container.
   ```bash
   docker run -p 8000:8000 -v $(pwd)/../models:/models sepsis-backend
   ```

## Scaling Approach

For high traffic, this API is designed to be **stateless**.
- Deploy behind a load balancer (e.g., Nginx, AWS ALB).
- Run multiple Uvicorn workers using Gunicorn:
  ```bash
  gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
  ```
- Models are loaded efficiently in each worker process during startup.

## Testing

Run tests with `pytest`:
```bash
pytest
```

## Example API Requests

### 1. Health Check
```bash
curl -X GET http://localhost:8000/health
```

### 2. Single Prediction (6h horizon by default)
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
           "patient_id": "P001",
           "age": 65,
           "gender": "M",
           "lactate_mean": 2.8,
           "sbp_mean": 90
         }'
```

### 3. Explainability
```bash
curl -X POST http://localhost:8000/predict/explain \
     -H "Content-Type: application/json" \
     -d '{
           "patient_id": "P001",
           "age": 65,
           "lactate_mean": 2.8
         }'
```

*(Note: Any missing clinical features are automatically imputed based on the patient's existing features, using the pre-trained MICE imputer).*
