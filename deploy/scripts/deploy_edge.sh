#!/usr/bin/env bash
# AdaptiveVision edge deployment (Milestone M18).
#
# Builds the AdaptiveVision edge image and brings up the observability stack
# (API + Prometheus + Grafana) via docker compose.
#
# Usage:
#   ./deploy/scripts/deploy_edge.sh [up|down|build|logs]
#
#   up      - build and start the edge stack (default)
#   down    - stop and remove the edge stack
#   build   - build the edge image only
#   logs    - tail the edge stack logs

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/deploy/compose/docker-compose.yml"

ACTION="${1:-up}"

case "${ACTION}" in
  up)
    docker compose -f "${COMPOSE_FILE}" up --build -d
    echo "AdaptiveVision edge stack is up."
    echo "  API:        http://localhost:8000"
    echo "  Prometheus: http://localhost:9090"
    echo "  Grafana:    http://localhost:3000 (admin/admin)"
    ;;
  down)
    docker compose -f "${COMPOSE_FILE}" down
    ;;
  build)
    docker build -f "${REPO_ROOT}/deploy/docker/Dockerfile" -t adaptivevision:edge "${REPO_ROOT}"
    ;;
  logs)
    docker compose -f "${COMPOSE_FILE}" logs -f
    ;;
  *)
    echo "Unknown action '${ACTION}'. Use up|down|build|logs." >&2
    exit 2
    ;;
esac
