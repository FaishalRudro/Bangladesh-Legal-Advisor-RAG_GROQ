FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build React frontend
COPY frontend/package*.json ./frontend/
RUN cd frontend && npm ci

COPY frontend/ ./frontend/
RUN cd frontend && npm run build

# Copy backend
COPY backend/ ./backend/

# Move built frontend to backend static folder
RUN mv frontend/dist backend/static

EXPOSE 7860

ENV PORT=7860

WORKDIR /app/backend

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]