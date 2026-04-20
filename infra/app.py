#!/usr/bin/env python3
"""CDK app entrypoint for Home Credit MLOps infra."""
import os
import aws_cdk as cdk
from stacks.base_stack import HomeCreditBaseStack

app = cdk.App()

env = cdk.Environment(
    account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
    region=os.environ.get("CDK_DEFAULT_REGION", "us-west-2"),
)

HomeCreditBaseStack(
    app,
    "HomeCreditBaseStack",
    env=env,
    description="Phase 1: S3 buckets, IAM roles, billing alarm for Home Credit MLOps project",
)

app.synth()
