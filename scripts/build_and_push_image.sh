#!/usr/bin/env bash
# Build the homecredit-images container and push to ECR (us-west-2).
#
# Prereqs:
#   - Docker Desktop running.
#   - `source env.sh` already done (project AWS profile + region exported).
#   - HomeCreditTrainingStack already deployed so the ECR repo exists.
#
# Tags pushed per build:
#   - latest
#   - <git short SHA>     (so we can pin Pipeline runs to immutable digests later)
#
# We force linux/amd64 because SageMaker Processing/Training nodes are amd64
# regardless of the build host (M-series Macs are arm64 by default).

set -euo pipefail

REGION="${AWS_REGION:-us-west-2}"
REPO_NAME="homecredit-images"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
REPO_URI="${REGISTRY}/${REPO_NAME}"
SHA="$(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo dev)"

DOCKER_CONTEXT="$(dirname "$0")/../docker/processing"

echo "==> Building ${REPO_URI}:{latest,${SHA}} from ${DOCKER_CONTEXT}"
docker buildx build \
    --platform linux/amd64 \
    --tag "${REPO_URI}:latest" \
    --tag "${REPO_URI}:${SHA}" \
    --load \
    "${DOCKER_CONTEXT}"

echo "==> Logging in to ECR ${REGISTRY}"
aws ecr get-login-password --region "${REGION}" \
    | docker login --username AWS --password-stdin "${REGISTRY}"

echo "==> Pushing both tags"
docker push "${REPO_URI}:latest"
docker push "${REPO_URI}:${SHA}"

echo ""
echo "✓ Image pushed:"
echo "    ${REPO_URI}:latest"
echo "    ${REPO_URI}:${SHA}"
echo ""
echo "Use this URI in training_pipeline.py:"
aws ecr describe-images --repository-name "${REPO_NAME}" --region "${REGION}" \
    --query 'imageDetails[0].[imageTags[0], imageDigest, imageSizeInBytes]' \
    --output table
