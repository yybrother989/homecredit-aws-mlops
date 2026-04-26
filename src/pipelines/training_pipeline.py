"""SageMaker Pipeline: pull features → HPO → evaluate best → conditional register.

Steps:
  1. ProcessingStep  — pull_features.py queries Athena via CTAS, writes
     train/validation parquet to the pipeline's S3 workspace.
  2. TuningStep      — Automatic Model Tuning, Bayesian, `max_jobs` trials
     over num_leaves / learning_rate / feature_fraction / bagging_fraction.
     Objective: maximize validation:stability regex-parsed from training logs.
  3. ProcessingStep  — evaluate_best runs evaluate.py against the best trial's
     model artifact, writes evaluation.json (AUC + stability).
  4. ConditionStep   — if stability ≥ threshold → RegisterModel.
  5. RegisterModel   — adds a new version to the HomeCreditModels package group,
     PendingManualApproval.

Run:
    python -m src.pipelines.training_pipeline --upsert
    python -m src.pipelines.training_pipeline --start
"""
from __future__ import annotations

import argparse
import logging

import boto3
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.tuner import ContinuousParameter, HyperparameterTuner, IntegerParameter
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TuningStep

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PIPELINE_NAME = "HomeCreditTrainingPipeline"
MODEL_PACKAGE_GROUP = "HomeCreditModels"
REGION = "us-west-2"


def cfn_output(stack_name: str, key: str) -> str:
    cf = boto3.client("cloudformation", region_name=REGION)
    outs = cf.describe_stacks(StackName=stack_name)["Stacks"][0]["Outputs"]
    for o in outs:
        if o["OutputKey"] == key:
            return o["OutputValue"]
    raise KeyError(f"{key} not in {stack_name} outputs")


def build_pipeline(role_arn: str, max_jobs: int = 20, max_parallel_jobs: int = 4) -> Pipeline:
    session = PipelineSession()
    artifacts_bucket = cfn_output("HomeCreditBaseStack", "ArtifactsBucketName")
    images_repo = cfn_output("HomeCreditTrainingStack", "ImagesRepoUri")
    athena_staging = f"s3://{artifacts_bucket}/athena/"
    custom_image = f"{images_repo}:latest"   # built by scripts/build_and_push_image.sh

    # --- Parameters (overridable at StartPipelineExecution) ---
    p_stability = ParameterFloat("StabilityThreshold", default_value=0.55)
    p_approval = ParameterString("ModelApprovalStatus", default_value="PendingManualApproval")
    p_max_jobs = ParameterInteger("MaxJobs", default_value=max_jobs)
    p_max_parallel = ParameterInteger("MaxParallelJobs", default_value=max_parallel_jobs)

    # All Processing + Training steps use our custom python:3.11-slim image
    # (numpy 1.26 + pandas 2.2 + pyarrow 15 + awswrangler 3.6 + lightgbm 4.3
    # + scikit-learn 1.4) — built once, ABI-consistent, no requirements.txt
    # mid-job pip installs.

    # ========================================================================
    # Step 1: pull features from Athena
    # ========================================================================
    pull_processor = ScriptProcessor(
        image_uri=custom_image,
        command=["python3"],
        role=role_arn,
        instance_type="ml.t3.xlarge",
        instance_count=1,
        base_job_name="hc-pull-features",
        sagemaker_session=session,
    )
    step_pull = ProcessingStep(
        name="PullFeatures",
        processor=pull_processor,
        outputs=[
            ProcessingOutput(output_name="train",
                             source="/opt/ml/processing/output/train"),
            ProcessingOutput(output_name="validation",
                             source="/opt/ml/processing/output/validation"),
        ],
        code="src/pipelines/pull_features.py",
        job_arguments=[
            "--athena-database", "homecredit_ml",
            "--athena-table", "features",
            "--athena-staging-uri", athena_staging,
            "--output-dir", "/opt/ml/processing/output",
        ],
    )

    # ========================================================================
    # Step 2: HPO via SageMaker Automatic Model Tuning
    # Custom container's `/usr/local/bin/train` shim parses
    # /opt/ml/input/config/hyperparameters.json and execs the entrypoint
    # script with --flag value pairs. The SDK uploads source_dir → /opt/ml/code,
    # sets SAGEMAKER_PROGRAM, and the shim handles the rest.
    # ========================================================================
    lightgbm_estimator = Estimator(
        image_uri=custom_image,
        entry_point="train_lightgbm.py",
        source_dir="src/training",
        role=role_arn,
        instance_type="ml.t3.xlarge",
        instance_count=1,
        output_path=f"s3://{artifacts_bucket}/models/hpo/",
        base_job_name="hc-train",
        sagemaker_session=session,
        hyperparameters={
            "n-estimators": 500,
            "early-stopping-rounds": 30,
        },
        metric_definitions=[
            {"Name": "validation:auc",       "Regex": r"Validation AUC=([0-9\.]+)"},
            {"Name": "validation:stability", "Regex": r"Stability=([0-9\.]+)"},
        ],
    )

    tuner = HyperparameterTuner(
        estimator=lightgbm_estimator,
        objective_metric_name="validation:stability",
        objective_type="Maximize",
        metric_definitions=[
            {"Name": "validation:auc",       "Regex": r"Validation AUC=([0-9\.]+)"},
            {"Name": "validation:stability", "Regex": r"Stability=([0-9\.]+)"},
        ],
        hyperparameter_ranges={
            "num-leaves":        IntegerParameter(16, 127),
            "learning-rate":     ContinuousParameter(0.01, 0.1),
            "feature-fraction":  ContinuousParameter(0.5, 0.95),
            "bagging-fraction":  ContinuousParameter(0.5, 0.95),
        },
        max_jobs=max_jobs,
        max_parallel_jobs=max_parallel_jobs,
        strategy="Bayesian",
    )
    step_tune = TuningStep(
        name="HPOTuning",
        tuner=tuner,
        inputs={
            "train": TrainingInput(
                s3_data=step_pull.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            ),
            "validation": TrainingInput(
                s3_data=step_pull.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
            ),
        },
    )

    # ========================================================================
    # Step 3: evaluate the best trial
    # ========================================================================
    eval_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    evaluator = ScriptProcessor(
        image_uri=custom_image,
        command=["python3"],
        role=role_arn,
        instance_type="ml.t3.large",
        instance_count=1,
        base_job_name="hc-evaluate",
        sagemaker_session=session,
    )

    # The TuningStep surfaces the best model's S3 URI via get_top_model_s3_uri
    best_model_uri = step_tune.get_top_model_s3_uri(
        top_k=0,
        s3_bucket=artifacts_bucket,
        prefix="models/hpo",
    )

    step_eval = ProcessingStep(
        name="EvaluateBestModel",
        processor=evaluator,
        inputs=[
            ProcessingInput(source=best_model_uri, destination="/opt/ml/processing/model"),
            ProcessingInput(
                source=step_pull.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                destination="/opt/ml/processing/validation",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation",
                             source="/opt/ml/processing/evaluation"),
        ],
        code="src/pipelines/evaluate.py",
        property_files=[eval_report],
    )

    # ========================================================================
    # Step 4: conditional registration
    # ========================================================================
    eval_s3 = Join(
        on="/",
        values=[
            step_eval.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
            "evaluation.json",
        ],
    )
    register = RegisterModel(
        name="RegisterBestModel",
        estimator=lightgbm_estimator,
        model_data=best_model_uri,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=MODEL_PACKAGE_GROUP,
        approval_status=p_approval,
        model_metrics=ModelMetrics(
            model_statistics=MetricsSource(s3_uri=eval_s3, content_type="application/json"),
        ),
    )
    step_cond = ConditionStep(
        name="StabilityGate",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(step_name=step_eval.name, property_file=eval_report,
                             json_path="metrics.stability.value"),
                right=p_stability,
            ),
        ],
        if_steps=[register],
        else_steps=[],
    )

    return Pipeline(
        name=PIPELINE_NAME,
        parameters=[p_stability, p_approval, p_max_jobs, p_max_parallel],
        steps=[step_pull, step_tune, step_eval, step_cond],
        sagemaker_session=session,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role-arn", default=None, help="Default: read from CFN output")
    ap.add_argument("--max-jobs", type=int, default=20)
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument("--upsert", action="store_true")
    ap.add_argument("--start", action="store_true")
    args = ap.parse_args()

    role_arn = args.role_arn or cfn_output("HomeCreditBaseStack", "SageMakerRoleArn")
    log.info("Building pipeline with role=%s  max_jobs=%d  parallel=%d",
             role_arn, args.max_jobs, args.max_parallel)

    pipeline = build_pipeline(role_arn, args.max_jobs, args.max_parallel)

    if args.upsert:
        log.info("Upserting pipeline %s", PIPELINE_NAME)
        pipeline.upsert(role_arn=role_arn)
    if args.start:
        execution = pipeline.start()
        log.info("Started: %s", execution.arn)


if __name__ == "__main__":
    main()
