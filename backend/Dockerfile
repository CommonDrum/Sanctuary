# Backend Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System dependencies for Chroma and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["flask", "run", "--host=0.0.0.0"]