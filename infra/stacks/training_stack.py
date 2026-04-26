"""Phase 3 training stack: Model Package Group + ECR repo for BYO container
+ Athena grants for HPO pipeline.

What it provisions:
  - `HomeCreditModels` SageMaker Model Package Group (registry for trained
    model versions produced by HomeCreditTrainingPipeline).
  - `homecredit-images` ECR repository for the project's custom container
    (built by scripts/build_and_push_image.sh, consumed by the Pipeline).
  - Extends the SageMaker execution role with: Athena + Glue Data Catalog
    read so PullFeatures can CTAS; ECR pull on the new repo; HPO APIs.

What it intentionally does NOT do:
  - CDK does not create the SageMaker Pipeline definition itself; the
    pipeline is declared/upserted from `src/pipelines/training_pipeline.py`
    via the SageMaker Python SDK. CDK-managing pipelines couples infra to
    pipeline changes too tightly.
"""
from aws_cdk import (
    CfnOutput,
    RemovalPolicy,
    Stack,
)
from aws_cdk import (
    aws_ecr as ecr,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_sagemaker as sagemaker,
)
from constructs import Construct

MODEL_PACKAGE_GROUP = "HomeCreditModels"
ECR_REPO_NAME = "homecredit-images"


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

        # --- ECR repo for custom container (BYO image, Phase 7 slice) ---
        images_repo = ecr.Repository(
            self, "ImagesRepo",
            repository_name=ECR_REPO_NAME,
            image_scan_on_push=True,
            image_tag_mutability=ecr.TagMutability.MUTABLE,
            removal_policy=RemovalPolicy.DESTROY,
            empty_on_delete=True,
            lifecycle_rules=[
                ecr.LifecycleRule(
                    description="Keep last 10 image versions",
                    max_image_count=10,
                    rule_priority=1,
                ),
            ],
        )
        # Allow SageMaker exec role to pull this image at job start
        images_repo.grant_pull(sm_role)

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
        CfnOutput(self, "ImagesRepoUri",
                  value=images_repo.repository_uri,
                  export_name="HomeCreditImagesRepoUri")
        CfnOutput(self, "ImagesRepoName",
                  value=images_repo.repository_name,
                  export_name="HomeCreditImagesRepoName")
