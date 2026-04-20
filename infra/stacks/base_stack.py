"""Phase 1 base stack: data lake buckets, SageMaker execution role, billing alarm."""
from aws_cdk import (
    Stack,
    RemovalPolicy,
    Duration,
    CfnOutput,
    aws_s3 as s3,
    aws_iam as iam,
    aws_cloudwatch as cw,
    aws_budgets as budgets,
)
from constructs import Construct


class HomeCreditBaseStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        account = self.account
        suffix = f"{account}-usw2"

        # --- Data lake buckets (bronze / silver / gold) ---
        raw_bucket = s3.Bucket(
            self, "RawBucket",
            bucket_name=f"homecredit-raw-{suffix}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="transition-to-ia",
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=Duration.days(30),
                        ),
                    ],
                ),
            ],
        )

        processed_bucket = s3.Bucket(
            self, "ProcessedBucket",
            bucket_name=f"homecredit-processed-{suffix}",
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        artifacts_bucket = s3.Bucket(
            self, "ArtifactsBucket",
            bucket_name=f"homecredit-artifacts-{suffix}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # --- SageMaker execution role ---
        sm_role = iam.Role(
            self, "SageMakerExecutionRole",
            role_name="HomeCreditSageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
            ],
        )
        for b in (raw_bucket, processed_bucket, artifacts_bucket):
            b.grant_read_write(sm_role)

        # --- Glue role for feature engineering (Phase 2) ---
        glue_role = iam.Role(
            self, "GlueETLRole",
            role_name="HomeCreditGlueETLRole",
            assumed_by=iam.ServicePrincipal("glue.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSGlueServiceRole"),
            ],
        )
        for b in (raw_bucket, processed_bucket, artifacts_bucket):
            b.grant_read_write(glue_role)

        # --- Monthly budget guardrail ($50) ---
        budgets.CfnBudget(
            self, "MonthlyBudget",
            budget=budgets.CfnBudget.BudgetDataProperty(
                budget_type="COST",
                time_unit="MONTHLY",
                budget_limit=budgets.CfnBudget.SpendProperty(amount=50, unit="USD"),
                budget_name="HomeCreditMonthlyBudget",
            ),
        )

        # --- Outputs ---
        CfnOutput(self, "RawBucketName", value=raw_bucket.bucket_name)
        CfnOutput(self, "ProcessedBucketName", value=processed_bucket.bucket_name)
        CfnOutput(self, "ArtifactsBucketName", value=artifacts_bucket.bucket_name)
        CfnOutput(self, "SageMakerRoleArn", value=sm_role.role_arn)
        CfnOutput(self, "GlueRoleArn", value=glue_role.role_arn)
