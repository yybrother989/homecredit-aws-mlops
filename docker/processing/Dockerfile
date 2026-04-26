# Custom container for SageMaker Processing + Training jobs.
#
# Why: SageMaker's stock sklearn 1.2-1 image pins Python 3.9 + numpy 1.24 +
# pandas 1.1 + scipy 1.8 with C extensions tied to those numpy ABI bytes.
# Layering modern awswrangler / pyarrow on top via requirements.txt produces
# a hybrid env where pyarrow.Table → pandas conversion crashes on 975-col
# tables ("numpy.core.multiarray failed to import"). Only a clean image with
# all deps installed against ONE numpy version fixes this.
#
# The image is consumed by the SageMaker Pipeline as `image_uri` for both
# ScriptProcessor (PullFeatures, EvaluateBestModel) and the Estimator
# (TrainingStep / TuningStep). sagemaker-training-toolkit's `train` entrypoint
# is what makes the same image work for training jobs.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AWS_DEFAULT_REGION=us-west-2

# OpenMP for LightGBM, build tools for any source-only deps, ca-certs for boto.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
        curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /tmp/requirements.txt

# Hand-rolled SageMaker training entrypoint (replaces sagemaker-training-
# toolkit's `train` console script, which doesn't build on Py 3.11).
COPY train /usr/local/bin/train
RUN chmod +x /usr/local/bin/train

# Default SAGEMAKER_PROGRAM; the SDK overrides this for each training job.
ENV SAGEMAKER_PROGRAM=train_lightgbm.py
ENTRYPOINT ["train"]
