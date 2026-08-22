# syntax=docker/dockerfile:1.7

FROM node:22-alpine AS frontend-build
WORKDIR /app/frontend
ARG VITE_API_BEARER_TOKEN=local-development-admin-token
ENV VITE_API_BEARER_TOKEN=${VITE_API_BEARER_TOKEN}
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production
WORKDIR /app

COPY backend/pyproject.toml backend/README.md ./backend/
COPY backend/src ./backend/src
RUN pip install --no-cache-dir ./backend

COPY --from=frontend-build /app/frontend/dist ./frontend/dist
RUN mkdir -p /app/data/attachments

EXPOSE 8000
CMD ["sh", "-c", "uvicorn multimedia_intelligence.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
