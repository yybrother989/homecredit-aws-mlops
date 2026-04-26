"""Phase 3 training stack: Model Package Group + Athena grants for HPO pipeline.

What it provisions:
  - `HomeCreditModels` SageMaker Model Package Group (registry for trained
    model versions produced by HomeCreditTrainingPipeline).
  - Extends the SageMaker execution role with Athena + Glue Data Catalog read
    so the PullFeatures ProcessingStep can query via CTAS.

What it intentionally does NOT do:
  - CDK does not create the SageMaker Pipeline definition itself; the
    pipeline is declared/upserted from `src/pipelines/training_pipeline.py`
    via the SageMaker Python SDK. CDK-managing pipelines couples infra to
    pipeline changes too tightly.
"""
from aws_cdk import (
    CfnOutput,
    Stack,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_sagemaker as sagemaker,
)
from constructs import Construct

MODEL_PACKAGE_GROUP = "HomeCreditModels"


class HomeCreditTrainingStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        sm_role: iam.IRole,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # --- Model Package Group ---
        mpg = sagemaker.CfnModelPackageGroup(
            self, "ModelPackageGroup",
            model_package_group_name=MODEL_PACKAGE_GROUP,
            model_package_group_description="Home Credit credit-risk models (LightGBM, produced by HomeCreditTrainingPipeline)",
            tags=[
                {"key": "Project", "value": "HomeCredit"},
                {"key": "ManagedBy", "value": "CDK"},
            ],
        )

        # --- Grant SageMaker exec role Athena + Glue Catalog read ---
        # PullFeatures step queries Athena via awswrangler; needs these APIs
        # that AmazonSageMakerFullAccess does NOT include.
        sm_role.add_to_principal_policy(iam.PolicyStatement(
            actions=[
                "athena:StartQueryExecution",
                "athena:GetQueryExecution",
                "athena:GetQueryResults",
                "athena:StopQueryExecution",
                "athena:GetWorkGroup",
                "athena:CreatePreparedStatement",
                "athena:DeletePreparedStatement",
                "athena:GetPreparedStatement",
            ],
            resources=["*"],
        ))
        sm_role.add_to_principal_policy(iam.PolicyStatement(
            actions=[
                "glue:GetDatabase",
                "glue:GetDatabases",
                "glue:GetTable",
                "glue:GetTables",
                "glue:GetPartition",
                "glue:GetPartitions",
                "glue:BatchGetPartition",
                "glue:CreateTable",     # CTAS writes an intermediate table
                "glue:DeleteTable",
                "glue:UpdateTable",
            ],
            resources=["*"],
        ))
        # HPO creates many training jobs; make sure role can describe them
        sm_role.add_to_principal_policy(iam.PolicyStatement(
            actions=[
                "sagemaker:CreateHyperParameterTuningJob",
                "sagemaker:DescribeHyperParameterTuningJob",
                "sagemaker:ListHyperParameterTuningJobs",
                "sagemaker:StopHyperParameterTuningJob",
            ],
            resources=["*"],
        ))

        CfnOutput(self, "ModelPackageGroupName",
                  value=MODEL_PACKAGE_GROUP,
                  export_name="HomeCreditModelPackageGroup")
        CfnOutput(self, "ModelPackageGroupArn", value=mpg.attr_model_package_group_arn)
