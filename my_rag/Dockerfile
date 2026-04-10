FROM python:3.9-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY eu_ai_act.pdf .
COPY api.py .
COPY app.py .
COPY improved_rag.py .
COPY models/ ./models/

# Expose ports
EXPOSE 8000
EXPOSE 8501

# Default: run the API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]