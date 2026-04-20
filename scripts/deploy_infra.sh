#!/usr/bin/env bash
# Deploy Phase 1 CDK stack to us-west-2.
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

echo "==> Deploying HomeCreditBaseStack..."
uv run cdk deploy HomeCreditBaseStack --require-approval never
