#!/usr/bin/env bash
# Deploy all CDK stacks for Home Credit MLOps to us-west-2:
#   - HomeCreditBaseStack    (Phase 1: S3, IAM, budget)
#   - HomeCreditFeatureStack (Phase 2: Glue job, Feature Store grants)
set -euo pipefail

export CDK_DEFAULT_REGION=us-west-2
export CDK_DEFAULT_ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"

echo "==> Account: ${CDK_DEFAULT_ACCOUNT}  Region: ${CDK_DEFAULT_REGION}"

cd "$(dirname "$0")/../infra"

if ! aws cloudformation describe-stacks --stack-name CDKToolkit --region "${CDK_DEFAULT_REGION}" &>/dev/null; then
  echo "==> Bootstrapping CDK..."
  uv run cdk bootstrap "aws://${CDK_DEFAULT_ACCOUNT}/${CDK_DEFAULT_REGION}"
fi

echo "==> Synthesizing..."
uv run cdk synth --quiet

echo "==> Deploying all stacks..."
uv run cdk deploy --all --require-approval never
