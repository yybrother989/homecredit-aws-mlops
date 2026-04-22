"""Phase 2 feature stack: Glue job + SageMaker Feature Store (offline).

What it does:
  - Uploads the PySpark feature script from `src/features/glue_feature_job.py`
    to `s3://<artifacts_bucket>/glue/` via BucketDeployment (re-deploys the
    script on every `cdk deploy`).
  - Declares the Glue 5.0 job `homecredit-features` bound to the script.
  - Grants the BaseStack Glue role permission to write to Feature Store
    (used by the downstream ingestion step, not the Glue job itself).

What it intentionally does NOT do:
  - Create the SageMaker Feature Group. The FG schema is derived from the
    parquet produced by the first Glue run; `src/features/ingest_to_feature_store.py`
    creates it at ingestion time so we don't hard-code ~500 feature names.
"""
from aws_cdk import (
    CfnOutput,
    Stack,
)
from aws_cdk import (
    aws_glue as glue,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_s3 as s3,
)
from aws_cdk import (
    aws_s3_deployment as s3deploy,
)
from constructs import Construct

GLUE_JOB_NAME = "homecredit-features"
GLUE_VERSION = "5.0"        # PySpark 3.5, Python 3.11
GLUE_DPU_WORKERS = 10       # G.1X — 4 vCPU / 16 GB each
GLUE_TIMEOUT_MIN = 30


class HomeCreditFeatureStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        raw_bucket: s3.IBucket,
        processed_bucket: s3.IBucket,
        artifacts_bucket: s3.IBucket,
        glue_role: iam.IRole,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # --- Upload PySpark script into artifacts bucket ---
        script_deploy = s3deploy.BucketDeployment(
            self, "GlueScriptDeployment",
            sources=[s3deploy.Source.asset("../src/features")],
            destination_bucket=artifacts_bucket,
            destination_key_prefix="glue/",
            prune=False,
            retain_on_delete=False,
        )
        script_s3_uri = f"s3://{artifacts_bucket.bucket_name}/glue/glue_feature_job.py"

        # --- Glue job definition ---
        glue_job = glue.CfnJob(
            self, "FeatureEngineeringJob",
            name=GLUE_JOB_NAME,
            role=glue_role.role_arn,
            glue_version=GLUE_VERSION,
            worker_type="G.1X",
            number_of_workers=GLUE_DPU_WORKERS,
            timeout=GLUE_TIMEOUT_MIN,
            command=glue.CfnJob.JobCommandProperty(
                name="glueetl",
                python_version="3",
                script_location=script_s3_uri,
            ),
            default_arguments={
                "--RAW_BUCKET": raw_bucket.bucket_name,
                "--PROCESSED_BUCKET": processed_bucket.bucket_name,
                "--SPLIT": "train",
                "--enable-metrics": "true",
                "--enable-spark-ui": "true",
                "--spark-event-logs-path": f"s3://{artifacts_bucket.bucket_name}/glue/spark-logs/",
                "--enable-job-insights": "true",
                "--enable-continuous-cloudwatch-log": "true",
                "--job-language": "python",
                "--TempDir": f"s3://{artifacts_bucket.bucket_name}/glue/tmp/",
            },
            execution_property=glue.CfnJob.ExecutionPropertyProperty(
                max_concurrent_runs=1,
            ),
            tags={"Project": "HomeCredit", "ManagedBy": "CDK"},
        )
        glue_job.node.add_dependency(script_deploy)

        # --- Grant Feature Store write to the shared Glue role ---
        # Glue job itself doesn't talk to Feature Store, but we reuse this
        # role for the local ingestion script below.
        glue_role.add_to_principal_policy(iam.PolicyStatement(
            actions=[
                "sagemaker:CreateFeatureGroup",
                "sagemaker:DescribeFeatureGroup",
                "sagemaker:DeleteFeatureGroup",
                "sagemaker:PutRecord",
                "sagemaker:BatchGetRecord",
                "sagemaker:GetRecord",
                "sagemaker:ListFeatureGroups",
                "sagemaker:UpdateFeatureGroup",
            ],
            resources=["*"],
        ))

        # --- Outputs ---
        CfnOutput(self, "GlueJobName", value=GLUE_JOB_NAME,
                  export_name="HomeCreditGlueJobName")
        CfnOutput(self, "GlueScriptS3Uri", value=script_s3_uri)
        CfnOutput(self, "FeatureGroupName",
                  value="homecredit-features",
                  description="Created at ingestion time, not by CDK",
                  export_name="HomeCreditFeatureGroupName")
