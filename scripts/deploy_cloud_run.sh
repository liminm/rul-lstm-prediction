#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="ml-zoomcamp-cmapss" # replace with your GCP project ID
REGION="europe-west3"
REPO="rul-repo"
SERVICE="rul-app"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:latest"

# Build + push

docker build -t "${IMAGE}" .
docker push "${IMAGE}"

# Deploy

gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --allow-unauthenticated \
  --port 8080
