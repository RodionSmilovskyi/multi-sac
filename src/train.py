import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(os.environ["SM_OUTPUT_DIR"]):
        os.makedirs(os.environ["SM_OUTPUT_DIR"])

    checkpoint_dir = (
        os.environ["CHECKPOINT_DIR"]
        if "CHECKPOINT_DIR" in os.environ
        else "/opt/ml/checkpoints"
    )
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    tensorboard_dir = (
        os.environ["TENSORBOARD_DIR"]
        if "TENSORBOARD_DIR" in os.environ
        else "/opt/ml/output/tensorboard"
    )
