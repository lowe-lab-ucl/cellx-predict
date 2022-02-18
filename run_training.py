import argparse
import ast
from pathlib import Path

from cellxpredict.config import Models, config_from_args
from cellxpredict.train import train

# check whether we're running in a container
current_path = Path(__file__).parent.resolve()
container_path = current_path / "container"
if container_path.exists():
    DEFAULT_PATH = container_path
else:
    DEFAULT_PATH = current_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TauVAE models")

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
        "--log_dir",
        type=Path,
        default=DEFAULT_PATH / "logs",
        help="path to the TensorBoard log directory",
    )

    parser.add_argument(
        "--input_shape",
        type=ast.literal_eval,
        default=(64, 64, 2),
        help="input shape of the image data (W, H, C)",
    )

    parser.add_argument(
        "--layers",
        type=ast.literal_eval,
        default=[8, 16, 32, 64],
        help="encoder layers list",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="training mini-batch size",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="training epochs",
    )

    parser.add_argument(
        "--max_iterations_fraction",
        type=float,
        default=0.9,
        help="percentage of steps before capacity reaches max value",
    )

    parser.add_argument(
        "--capacity",
        type=int,
        default=50,
        help="network capacity",
    )

    parser.add_argument(
        "--num_outputs",
        type=int,
        default=3,
        help="number of outputs",
    )

    parser.add_argument(
        "--use_probabilistic_encoder",
        action="store_true",
        help="use a probabilistic encoder while training model",
    )

    parser.add_argument(
        "--noise",
        type=float,
        default=1.0,
        help="amplitude of noise when using a probabilistic encoder",
    )

    args = parser.parse_args()
    config = config_from_args(args)
    train(config)
