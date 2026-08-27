# AdaptiveVision edge API container (Milestone M18).
#
# Builds a slim runtime image that serves the AdaptiveVision HTTP API
# (results, metrics, health, dashboard) for edge observability. The image is
# built from the repository root so the package can be installed editable-free.
#
#   docker build -f deploy/docker/Dockerfile -t adaptivevision:edge .
#   docker run --rm -p 8000:8000 adaptivevision:edge

FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install the package (non-editable) so the runtime image is self-contained.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# The API is served by uvicorn; expose the default port.
EXPOSE 8000

# Default command: serve the AdaptiveVision API on all interfaces.
CMD ["uvicorn", "adaptivevision.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
