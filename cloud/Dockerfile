# One Dockerfile for all Python services: the pip layer is shared, only the last COPY differs.
# docker-compose passes SERVICE=<folder in services/>.
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY common /app/common
ARG SERVICE
COPY services/${SERVICE} /app/service
# extra packages for one service only (e.g. cloud_inference needs the LiteRT runtime)
RUN if [ -f /app/service/requirements.txt ]; then pip install --no-cache-dir -r /app/service/requirements.txt; fi
ENV PYTHONPATH=/app
CMD ["python", "-m", "service.main"]
