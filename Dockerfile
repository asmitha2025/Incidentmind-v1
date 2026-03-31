FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose HuggingFace Spaces port
EXPOSE 7860

# Start the environment server
CMD ["uvicorn", "incident_env:app", "--host", "0.0.0.0", "--port", "7860"]
