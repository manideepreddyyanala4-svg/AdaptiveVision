# AdaptiveVision edge API container (Milestone M18).
#
# Builds a slim runtime image that serves the AdaptiveVision HTTP API
# (results, metrics, health, dashboard) for edge observability. The image is
# built from the repository root so the package can be installed editable-free.
#
#   docker build -f deploy/Dockerfile -t adaptivevision:edge .
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
COPY scripts ./scripts

# The API is served by uvicorn; expose the default port.
EXPOSE 8000

# Default command: serve the AdaptiveVision API on all interfaces.
#
# Deliberately NOT `uvicorn adaptivevision.api:create_app --factory`: that
# calls the raw factory with zero arguments, but create_app()'s `repository`
# parameter has no default, so the container crashed at startup every time
# (docs/architecture.md, Milestone M20). scripts/run_api.py is the one place that actually
# builds the repository/advisory/deployment-profiles composition create_app()
# needs, exactly like scripts/run_station.py is the real entry point for the
# station rather than calling app.app.build_station() directly.
CMD ["python", "scripts/run_api.py", "--host", "0.0.0.0", "--port", "8000"]
