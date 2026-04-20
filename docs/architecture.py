"""Hero architecture diagram for the Home Credit MLOps project.

Run:
    uv run python docs/architecture.py

Writes docs/architecture.png (committed to the repo for the README).
Uses the `diagrams` library with official AWS icons — Lucidchart-style output.
"""
from diagrams import Cluster, Diagram, Edge
from diagrams.aws.analytics import Glue
from diagrams.aws.compute import Lambda
from diagrams.aws.devtools import Codebuild, Codepipeline
from diagrams.aws.general import User
from diagrams.aws.integration import Eventbridge
from diagrams.aws.management import (
    Cloudformation,
    Cloudwatch,
    CloudwatchAlarm,
)
from diagrams.aws.ml import (
    Sagemaker,
    SagemakerModel,
    SagemakerTrainingJob,
)
from diagrams.aws.network import APIGateway
from diagrams.aws.storage import S3
from diagrams.onprem.vcs import Github

GRAPH_ATTR = {
    "fontsize": "18",
    "bgcolor": "white",
    "pad": "0.8",
    "splines": "ortho",
    "nodesep": "0.6",
    "ranksep": "1.0",
}
NODE_ATTR = {"fontsize": "12"}
EDGE_ATTR = {"fontsize": "11"}

with Diagram(
    "Home Credit MLOps on AWS",
    filename="docs/architecture",
    show=False,
    direction="LR",
    outformat=["png", "svg"],
    graph_attr=GRAPH_ATTR,
    node_attr=NODE_ATTR,
    edge_attr=EDGE_ATTR,
):
    developer = User("developer")
    github = Github("GitHub")

    with Cluster("1. CI / CD"):
        codepipeline = Codepipeline("CodePipeline")
        codebuild = Codebuild("CodeBuild")
        cdk = Cloudformation("CDK\nIaC")
        codepipeline >> codebuild >> cdk

    with Cluster("2. Data Lake (S3)"):
        raw = S3("raw")
        processed = S3("processed")
        artifacts = S3("artifacts")

    with Cluster("3. Feature Pipeline"):
        glue = Glue("Glue ETL")
        feature_store = Sagemaker("Feature Store")
        glue >> feature_store

    with Cluster("4. Training"):
        sm_pipeline = Sagemaker("SageMaker\nPipeline")
        training_job = SagemakerTrainingJob("Train")
        gate = Lambda("Stability\nGate")
        registry = SagemakerModel("Model\nRegistry")
        sm_pipeline >> training_job >> gate
        gate >> Edge(label="pass") >> registry

    with Cluster("5. Serving"):
        apigw = APIGateway("API GW")
        endpoint = Sagemaker("Endpoint")
        batch = Sagemaker("Batch")
        apigw >> endpoint

    with Cluster("6. Monitoring → Retrain"):
        monitor = Sagemaker("Model\nMonitor")
        metrics = Cloudwatch("CloudWatch")
        alarm = CloudwatchAlarm("Drift\nAlarm")
        events = Eventbridge("EventBridge")
        monitor >> metrics >> alarm >> events

    developer >> github >> codepipeline
    cdk >> Edge(style="dashed", color="gray", label="deploys") >> sm_pipeline

    raw >> glue
    feature_store >> processed >> sm_pipeline
    registry >> artifacts
    registry >> [endpoint, batch]
    processed >> batch

    endpoint >> Edge(label="capture") >> monitor
    events >> Edge(color="firebrick", label="retrain") >> sm_pipeline
