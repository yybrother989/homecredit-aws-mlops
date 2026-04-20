"""SageMaker Pipeline: preprocess -> train -> evaluate -> conditional register.

Run:
    python -m src.pipelines.training_pipeline --role-arn <arn> --upsert
    python -m src.pipelines.training_pipeline --role-arn <arn> --start

Gate: model is registered only if stability metric > threshold.
"""
from __future__ import annotations

import argparse
import logging

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PIPELINE_NAME = "HomeCreditTrainingPipeline"
MODEL_PACKAGE_GROUP = "HomeCreditModels"


def build_pipeline(
    role_arn: str,
    region: str = "us-west-2",
    artifacts_bucket: str | None = None,
) -> Pipeline:
    session = PipelineSession()
    if artifacts_bucket is None:
        artifacts_bucket = session.default_bucket()

    raw_uri = ParameterString("RawDataUri", default_value=f"s3://homecredit-raw-{session.account_id()}-usw2/")
    processed_uri = ParameterString("ProcessedDataUri", default_value=f"s3://{artifacts_bucket}/processed/")
    stability_threshold = ParameterFloat("StabilityThreshold", default_value=0.40)
    approval_status = ParameterString("ModelApprovalStatus", default_value="PendingManualApproval")
    instance_count = ParameterInteger("TrainingInstanceCount", default_value=1)
    train_instance = ParameterString("TrainingInstanceType", default_value="ml.m5.xlarge")

    # Step 1: feature engineering
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role_arn,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name="homecredit-features",
        sagemaker_session=session,
    )
    step_process = ProcessingStep(
        name="FeatureEngineering",
        processor=sklearn_processor,
        inputs=[ProcessingInput(source=raw_uri, destination="/opt/ml/processing/input")],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/output/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/output/validation"),
        ],
        code="src/features/build_features.py",
        job_arguments=["--input-dir", "/opt/ml/processing/input", "--output-dir", "/opt/ml/processing/output", "--split", "train"],
    )

    # Step 2: training (script mode; container image placeholder — replace with prebuilt LightGBM image or BYO)
    estimator = Estimator(
        image_uri=sagemaker.image_uris.retrieve("sklearn", region, "1.2-1"),
        entry_point="train_lightgbm.py",
        source_dir="src/training",
        role=role_arn,
        instance_count=instance_count,
        instance_type=train_instance,
        output_path=f"s3://{artifacts_bucket}/models/",
        base_job_name="homecredit-train",
        sagemaker_session=session,
        hyperparameters={"num-leaves": 63, "learning-rate": 0.05},
    )
    step_train = TrainingStep(
        name="TrainLightGBM",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri
            ),
        },
    )

    # Step 3: evaluation
    eval_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")
    evaluator = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve("sklearn", region, "1.2-1"),
        command=["python3"],
        role=role_arn,
        instance_type="ml.m5.large",
        instance_count=1,
        base_job_name="homecredit-evaluate",
        sagemaker_session=session,
    )
    step_eval = ProcessingStep(
        name="EvaluateModel",
        processor=evaluator,
        inputs=[
            ProcessingInput(source=step_train.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/model"),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                destination="/opt/ml/processing/validation",
            ),
        ],
        outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
        code="src/pipelines/evaluate.py",
        property_files=[eval_report],
    )

    # Step 4: conditional register
    register = RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=MODEL_PACKAGE_GROUP,
        approval_status=approval_status,
        model_metrics=ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=step_eval.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri + "/evaluation.json",
                content_type="application/json",
            ),
        ),
    )
    step_cond = ConditionStep(
        name="StabilityGate",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(step_name=step_eval.name, property_file=eval_report, json_path="metrics.stability.value"),
                right=stability_threshold,
            ),
        ],
        if_steps=[register],
        else_steps=[],
    )

    return Pipeline(
        name=PIPELINE_NAME,
        parameters=[raw_uri, processed_uri, stability_threshold, approval_status, instance_count, train_instance],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=session,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role-arn", required=True)
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--artifacts-bucket", default=None)
    ap.add_argument("--upsert", action="store_true", help="Create or update the pipeline definition")
    ap.add_argument("--start", action="store_true", help="Start a pipeline execution")
    args = ap.parse_args()

    pipeline = build_pipeline(args.role_arn, args.region, args.artifacts_bucket)
    if args.upsert:
        log.info("Upserting pipeline %s", PIPELINE_NAME)
        pipeline.upsert(role_arn=args.role_arn)
    if args.start:
        log.info("Starting pipeline execution")
        pipeline.start()


if __name__ == "__main__":
    main()
