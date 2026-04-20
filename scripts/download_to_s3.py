"""Stream Kaggle competition data directly to S3 — never touches local disk.

Usage:
    uv run python scripts/download_to_s3.py
    uv run python scripts/download_to_s3.py --key raw/homecredit.zip
    uv run python scripts/download_to_s3.py --bucket other-bucket

Prereqs (one of):
    (a) New-style API token  (preferred):
        export KAGGLE_API_TOKEN=KGAT_xxx          # from kaggle.com/settings/account
    (b) Legacy kaggle.json:
        ~/.kaggle/kaggle.json with chmod 600

    Plus:
    - Competition rules accepted:
      https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/rules
    - HomeCreditBaseStack deployed (script reads bucket name from CloudFormation output)

How it works:
    1. Hit Kaggle API, which 302-redirects to a pre-signed download URL.
    2. `requests.get(stream=True)` pulls chunks from the remote.
    3. boto3 multipart upload streams those chunks to S3 (64 MB parts).
    4. Nothing is ever written to local disk.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import boto3
import requests
from boto3.s3.transfer import TransferConfig
from tqdm import tqdm

COMPETITION = "home-credit-credit-risk-model-stability"
CFN_STACK = "HomeCreditBaseStack"
CFN_OUTPUT_KEY = "RawBucketName"
DEFAULT_S3_KEY = "homecredit.zip"
PART_SIZE = 64 * 1024 * 1024  # 64 MB


@dataclass(frozen=True)
class KaggleAuth:
    """Either a bearer token (new-style) or basic-auth credentials (legacy)."""
    bearer: str | None = None
    username: str | None = None
    api_key: str | None = None

    @property
    def display(self) -> str:
        if self.bearer:
            return f"bearer:{self.bearer[:8]}…"
        return f"basic:{self.username}"

    def apply(self, req: requests.Request | requests.PreparedRequest) -> None:
        if self.bearer:
            req.headers["Authorization"] = f"Bearer {self.bearer}"
        else:
            req.auth = (self.username, self.api_key)


def load_kaggle_creds() -> KaggleAuth:
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        return KaggleAuth(bearer=token)

    path = Path.home() / ".kaggle" / "kaggle.json"
    if path.exists():
        mode = path.stat().st_mode & 0o777
        if mode != 0o600:
            print(
                f"[warn] ~/.kaggle/kaggle.json mode is {oct(mode)}; run `chmod 600`.",
                file=sys.stderr,
            )
        data = json.loads(path.read_text())
        return KaggleAuth(username=data["username"], api_key=data["key"])

    sys.exit(
        "✗ No Kaggle credentials found.\n"
        "  Either: export KAGGLE_API_TOKEN=KGAT_xxx\n"
        "  Or:     put kaggle.json at ~/.kaggle/kaggle.json (chmod 600)\n"
        "  Get one at https://www.kaggle.com/settings/account → Create New Token"
    )


def resolve_raw_bucket() -> str:
    cfn = boto3.client("cloudformation")
    outputs = cfn.describe_stacks(StackName=CFN_STACK)["Stacks"][0]["Outputs"]
    for o in outputs:
        if o["OutputKey"] == CFN_OUTPUT_KEY:
            return o["OutputValue"]
    sys.exit(f"✗ {CFN_OUTPUT_KEY} not found in stack {CFN_STACK} outputs")


def s3_already_has(s3, bucket: str, key: str, expected_size: int) -> bool:
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except s3.exceptions.ClientError:
        return False
    return expected_size == 0 or head["ContentLength"] == expected_size


def stream_to_s3(auth: KaggleAuth, bucket: str, s3_key: str) -> None:
    url = f"https://www.kaggle.com/api/v1/competitions/data/download-all/{COMPETITION}"
    headers = {"Accept-Encoding": "identity"}  # keep the byte stream verbatim
    kwargs: dict = {"headers": headers, "stream": True, "timeout": 30}
    if auth.bearer:
        headers["Authorization"] = f"Bearer {auth.bearer}"
    else:
        kwargs["auth"] = (auth.username, auth.api_key)

    with requests.get(url, **kwargs) as resp:
        if resp.status_code == 403:
            sys.exit(
                "✗ 403 Forbidden — accept the competition rules first:\n"
                f"  https://www.kaggle.com/competitions/{COMPETITION}/rules"
            )
        resp.raise_for_status()

        total = int(resp.headers.get("Content-Length", "0"))
        s3 = boto3.client("s3")

        if s3_already_has(s3, bucket, s3_key, total):
            print(f"✓ already on S3: s3://{bucket}/{s3_key} ({total:,} bytes) — skipping")
            return

        resp.raw.decode_content = True  # decompress transparently if server compressed
        pbar = tqdm(total=total, unit="B", unit_scale=True, desc="Kaggle → S3")

        class _ProgressReader:
            """Wrap a file-like so boto3 streams through tqdm."""
            def __init__(self, fp):
                self._fp = fp
            def read(self, size: int = -1) -> bytes:
                chunk = self._fp.read(size)
                pbar.update(len(chunk))
                return chunk

        try:
            s3.upload_fileobj(
                _ProgressReader(resp.raw),
                Bucket=bucket,
                Key=s3_key,
                ExtraArgs={
                    "Metadata": {"source": "kaggle", "competition": COMPETITION},
                },
                Config=TransferConfig(
                    multipart_threshold=PART_SIZE,
                    multipart_chunksize=PART_SIZE,
                    use_threads=True,
                    max_concurrency=4,
                ),
            )
        finally:
            pbar.close()

    print(f"✓ uploaded s3://{bucket}/{s3_key}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", default=None, help="Override S3 bucket (default: resolve from CloudFormation)")
    ap.add_argument("--key", default=DEFAULT_S3_KEY, help=f"S3 object key (default: {DEFAULT_S3_KEY})")
    args = ap.parse_args()

    auth = load_kaggle_creds()
    bucket = args.bucket or resolve_raw_bucket()
    print(f"  competition : {COMPETITION}")
    print(f"  kaggle auth : {auth.display}")
    print(f"  destination : s3://{bucket}/{args.key}")
    print()

    stream_to_s3(auth, bucket, args.key)


if __name__ == "__main__":
    main()
