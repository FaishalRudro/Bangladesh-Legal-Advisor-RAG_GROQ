FROM python:3.11-slim

# Install Node.js 20 + supervisor + build tools
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    supervisor \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Backend: Python dependencies ─────────────────────────────────────────────
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# ── Frontend: Node dependencies + build ──────────────────────────────────────
COPY frontend/package*.json ./frontend/
RUN cd frontend && npm ci

COPY frontend/ ./frontend/

# Build Next.js (standalone output for production)
# BACKEND_API_BASE_URL is set at runtime via HF secrets/vars, not needed at build time
RUN cd frontend && npm run build

# ── Copy backend source ───────────────────────────────────────────────────────
COPY backend/ ./backend/

# ── Supervisor config: run both processes ─────────────────────────────────────
RUN mkdir -p /var/log/supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# HuggingFace exposes port 7860 (Next.js frontend)
# Backend runs internally on port 8000
EXPOSE 7860

CMD ["/bin/bash", "-c", "\
  echo 'BACKEND_API_BASE_URL=http://localhost:8000' > /app/frontend/.env.local && \
  /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf\
"]
