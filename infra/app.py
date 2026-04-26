#!/usr/bin/env python3
"""CDK app entrypoint for Home Credit MLOps infra."""
import os

import aws_cdk as cdk
from stacks.base_stack import HomeCreditBaseStack
from stacks.feature_stack import HomeCreditFeatureStack
from stacks.training_stack import HomeCreditTrainingStack

app = cdk.App()

cdk.Tags.of(app).add("Project", "HomeCredit")
cdk.Tags.of(app).add("ManagedBy", "CDK")

# Static tag value for the AppRegistry-managed `awsApplication` tag. This
# must be a literal string because CloudFormation rejects unresolved tokens
# as stack-level tags (needed so Application Manager groups the stack, not
# just its resources). If the AppRegistry application is ever destroyed and
# recreated, the resource-groups ID suffix changes — update this constant
# from the latest `attr_application_tag_value` output after redeploy.
APPLICATION_TAG_VALUE = (
    "arn:aws:resource-groups:us-west-2:770826159657"
    ":group/HomeCredit/0df50q3hhq176ijs762e8jmofb"
)
cdk.Tags.of(app).add("awsApplication", APPLICATION_TAG_VALUE)

env = cdk.Environment(
    account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
    region=os.environ.get("CDK_DEFAULT_REGION", "us-west-2"),
)

base = HomeCreditBaseStack(
    app,
    "HomeCreditBaseStack",
    env=env,
    description="Phase 1: S3 buckets, IAM roles, billing alarm for Home Credit MLOps project",
)

HomeCreditFeatureStack(
    app,
    "HomeCreditFeatureStack",
    env=env,
    description="Phase 2: Glue feature engineering job + Feature Store permissions",
    raw_bucket=base.raw_bucket,
    processed_bucket=base.processed_bucket,
    artifacts_bucket=base.artifacts_bucket,
    glue_role=base.glue_role,
)

HomeCreditTrainingStack(
    app,
    "HomeCreditTrainingStack",
    env=env,
    description="Phase 3: Model Package Group + Athena/Glue grants for HPO pipeline",
    sm_role=base.sm_role,
)

app.synth()
