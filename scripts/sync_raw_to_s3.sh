#!/usr/bin/env bash
# One-time sync of locally-unzipped Kaggle parquet files into the S3 raw bucket.
#
# Context: Phase 1 streamed homecredit.zip directly to S3. The zip was then
# copied back locally and unzipped so we could iterate on a LightGBM baseline.
# Phase 2's Glue job reads parquets, not a zip — so we mirror the unzipped
# tree up to S3 here. Idempotent: re-running only uploads changed files.
#
# Prereq: source env.sh first.
set -euo pipefail

RAW_BUCKET="$(aws cloudformation describe-stacks \
    --stack-name HomeCreditBaseStack \
    --query 'Stacks[0].Outputs[?OutputKey==`RawBucketName`].OutputValue' \
    --output text)"

if [[ -z "${RAW_BUCKET}" ]]; then
    echo "✗ Could not resolve RawBucketName from HomeCreditBaseStack" >&2
    exit 1
fi

LOCAL_DIR="data/raw/parquet_files"
if [[ ! -d "${LOCAL_DIR}" ]]; then
    echo "✗ ${LOCAL_DIR} not found. Unzip homecredit.zip first." >&2
    exit 1
fi

echo "==> Syncing ${LOCAL_DIR}/ → s3://${RAW_BUCKET}/parquet_files/"
aws s3 sync "${LOCAL_DIR}/" "s3://${RAW_BUCKET}/parquet_files/" \
    --exclude "*.zip" \
    --only-show-errors

echo "✓ Done. Summary:"
aws s3 ls "s3://${RAW_BUCKET}/parquet_files/train/" --summarize --human-readable \
    | tail -3
