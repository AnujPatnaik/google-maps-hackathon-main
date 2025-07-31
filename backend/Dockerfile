FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copy the rest of your app
COPY . .

# Tell Flask/Gunicorn to listen on Render's port
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
