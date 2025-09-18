FROM python:3.11-slim

# 1) Workdir inside the container
WORKDIR /app

# 2) Install deps (cache-friendly)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copy the rest of your code
COPY . /app

# 4) Default command: run your RL script
CMD ["python", "main.py"]
