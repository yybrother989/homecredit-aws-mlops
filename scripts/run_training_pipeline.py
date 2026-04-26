"""Kickoff wrapper for the Phase 3 SageMaker Pipeline.

Usage:
    source env.sh
    # Upsert + start in one shot, with cheap 5-trial sanity run
    uv run python scripts/run_training_pipeline.py --max-jobs 5 --upsert --start

    # Full 20-trial production run
    uv run python scripts/run_training_pipeline.py --max-jobs 20 --max-parallel 4 --start

The script pulls the SageMaker execution role ARN from CloudFormation outputs,
so no hard-coded ARNs.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Scripts are launched from the project root; make `src` importable regardless
# of cwd so this works whether invoked as `python scripts/run_...py` or `-m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipelines.training_pipeline import build_pipeline, cfn_output  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-jobs", type=int, default=5)
    ap.add_argument("--max-parallel", type=int, default=2)
    ap.add_argument("--upsert", action="store_true", help="Create or update pipeline definition")
    ap.add_argument("--start", action="store_true", help="Kick off a new execution")
    ap.add_argument("--stability-threshold", type=float, default=None,
                    help="Override the stability gate (default 0.55 from pipeline params)")
    args = ap.parse_args()

    if not (args.upsert or args.start):
        log.error("Nothing to do — pass --upsert and/or --start")
        sys.exit(1)

    role_arn = cfn_output("HomeCreditBaseStack", "SageMakerRoleArn")
    log.info("Using SageMaker role %s", role_arn)

    pipeline = build_pipeline(role_arn, max_jobs=args.max_jobs, max_parallel_jobs=args.max_parallel)

    if args.upsert:
        log.info("Upserting pipeline %s (max_jobs=%d, parallel=%d)",
                 pipeline.name, args.max_jobs, args.max_parallel)
        pipeline.upsert(role_arn=role_arn)
        log.info("✓ upsert complete")

    if args.start:
        params = {}
        if args.stability_threshold is not None:
            params["StabilityThreshold"] = args.stability_threshold
        params["MaxJobs"] = args.max_jobs
        params["MaxParallelJobs"] = args.max_parallel

        log.info("Starting execution with params=%s", params)
        execution = pipeline.start(parameters=params)
        log.info("✓ execution started: %s", execution.arn)
        log.info("  Follow progress:")
        log.info("    aws sagemaker describe-pipeline-execution --pipeline-execution-arn %s",
                 execution.arn)


if __name__ == "__main__":
    main()
