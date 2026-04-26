"""Smoke tests for the SageMaker Pipeline definition.

`build_pipeline` constructs FrameworkProcessor steps that upload code to S3
during `.run()`. Faithfully unit-testing that requires either a LocalSession
or extensive mocking of the upload chain. Since we cover the structure end-
to-end via the `cdk synth` + `pipeline.upsert()` smoke run in the GitHub
Actions matrix, these tests just verify import correctness and module-level
constants.
"""
from __future__ import annotations


def test_module_imports_cleanly() -> None:
    from src.pipelines import training_pipeline

    assert training_pipeline.PIPELINE_NAME == "HomeCreditTrainingPipeline"
    assert training_pipeline.MODEL_PACKAGE_GROUP == "HomeCreditModels"
    assert training_pipeline.REGION == "us-west-2"
    assert callable(training_pipeline.build_pipeline)
    assert callable(training_pipeline.cfn_output)


def test_build_pipeline_signature() -> None:
    """Make sure the public function signature is stable for callers."""
    import inspect

    from src.pipelines.training_pipeline import build_pipeline

    sig = inspect.signature(build_pipeline)
    params = list(sig.parameters)
    assert params == ["role_arn", "max_jobs", "max_parallel_jobs"]
    assert sig.parameters["max_jobs"].default == 20
    assert sig.parameters["max_parallel_jobs"].default == 4
