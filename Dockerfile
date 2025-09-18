# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install dependencies (includes fastapi + uvicorn)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repo
COPY . .

# Expose port 8000 for the API
EXPOSE 8000

# Default command: start FastAPI with uvicorn
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
