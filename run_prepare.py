import argparse
from pathlib import Path

from cellxpredict.config import Models, config_from_args
from cellxpredict.prepare import prepare_temporal

# check whether we're running in a container
current_path = Path(__file__).parent.resolve()
container_path = current_path / "container"
if container_path.exists():
    DEFAULT_PATH = container_path
else:
    DEFAULT_PATH = current_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data")

    parser.add_argument(
        "--model",
        choices=[m.name.lower() for m in Models],
        type=str,
        required=True,
        help="name of the model to train",
    )

    parser.add_argument(
        "--src_dir",
        type=Path,
        default=DEFAULT_PATH / "data",
        help="path to the data directory",
    )

    parser.add_argument(
        "--model_dir",
        type=Path,
        default=DEFAULT_PATH / "models",
        help="path to the model output directory",
    )

    parser.add_argument(
        "--use_probabilistic_encoder",
        action="store_true",
        help="use a probabilistic encoder while training model",
    )

    parser.add_argument(
        "--use_rotations",
        action="store_true",
        help="use rotations in XY plane during data preparation",
    )

    args = parser.parse_args()
    print(args)
    config = config_from_args(args)
    prepare_temporal(config)
