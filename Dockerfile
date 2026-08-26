# syntax=docker/dockerfile:1.7

FROM node:22-alpine AS frontend-build
WORKDIR /app/frontend
ARG VITE_CLERK_PUBLISHABLE_KEY
ARG VITE_CHATKIT_API_DOMAIN_KEY=""
ARG VITE_CHATKIT_API_URL="/chatkit"
ENV VITE_CLERK_PUBLISHABLE_KEY=${VITE_CLERK_PUBLISHABLE_KEY} \
    VITE_CHATKIT_API_DOMAIN_KEY=${VITE_CHATKIT_API_DOMAIN_KEY} \
    VITE_CHATKIT_API_URL=${VITE_CHATKIT_API_URL}
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
RUN mkdir -p /app/data

EXPOSE 8000
CMD ["sh", "-c", "uvicorn multimedia_intelligence.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
