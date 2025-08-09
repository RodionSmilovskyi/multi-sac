import os
import argparse
import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.debugger import TensorBoardOutputConfig

WORKDIR = os.path.dirname(os.path.abspath(__file__))
ROLE = "arn:aws:iam::905418352696:role/SageMakerFullAccess"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--job-name", type=str, default=None)
    parser.add_argument("--upload-training-data", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--samples-per-image", type=int, default=1)
    parser.add_argument("--trainable-backbone-layers", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument(
        "--scheduler-type",
        choices=["cosine_warm_restarts", "one_cycle", "reduce_on_plateau", "multi_step"],
        default="multi_step"
    )

    args = parser.parse_args()

    boto_session = boto3.session.Session(
        profile_name="905418352696_AdministratorAccess", region_name="us-east-1"
    )
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    output_path = f"s3://{sagemaker_session.default_bucket()}/{args.prefix}"
    checkpoint_path = (
        f"s3://{sagemaker_session.default_bucket()}/{args.prefix}/checkpoint"
    )

    estimator = Estimator(
        sagemaker_session=sagemaker_session,
        base_job_name=args.job_name,
        image_uri=f"763104351884.dkr.ecr.{boto_session.region_name}.amazonaws.com/pytorch-training:2.6.0-gpu-py312-cu126-ubuntu22.04-sagemaker",
        role=ROLE,
        max_run=24 * 60 * 60,
        instance_count=1,
        instance_type="ml.g4dn.2xlarge",
        source_dir="src",
        entry_point="train.py",
        output_path=output_path,
        checkpoint_s3_uri=checkpoint_path,
        environment={"PYTHONPATH": "/opt/ml/code/detection"},
        tensorboard_output_config=TensorBoardOutputConfig(
            s3_output_path=f"s3://{sagemaker_session.default_bucket()}/{args.prefix}/tensorboard"
        ),
        hyperparameters={
            "epochs": args.epochs,
            "rounds": args.rounds,
            "seed": args.seed,
            "batch-size": args.batch_size,
            "lr": args.lr,
            "samples-per-image": args.samples_per_image,
            "trainable-backbone-layers": args.trainable_backbone_layers,
            "scheduler-type": args.scheduler_type,
            "weight-decay": args.weight_decay,
            "comment": args.comment
        },
    )

    if args.upload_training_data:
        data_location = sagemaker_session.upload_data(
            os.path.join(WORKDIR, "input"), key_prefix=args.prefix
        )

    estimator.fit(
        {
            "train": f"s3://{sagemaker_session.default_bucket()}/{args.prefix}/train",
            "validation": f"s3://{sagemaker_session.default_bucket()}/{args.prefix}/validation",
            "config": f"s3://{sagemaker_session.default_bucket()}/{args.prefix}/config",
        },
        wait=False,
    )

    print("Job submitted")
