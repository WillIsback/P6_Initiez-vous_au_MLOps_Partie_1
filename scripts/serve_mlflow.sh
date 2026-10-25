#!/usr/bin/env bash
set -euo pipefail

# Se place toujours à la racine du projet, d'où qu'on l'appelle
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Charge .env s'il existe
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

exec uvx mlflow server \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI:-sqlite:///mlflow.db}" \
  --default-artifact-root "${MLFLOW_ARTIFACT_ROOT:-./mlflow-artifacts}" \
  --host "${MLFLOW_HOST:-127.0.0.1}" \
  --port "${MLFLOW_PORT:-5000}"