import argparse
import ast

import pytest

from cellxpredict import config

MODELS = [m.name.lower() for m in config.Models]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("input_shape", [(8, 8, 1), (64, 64, 2), (128, 128, 8)])
def test_arg_parse_to_config(model, input_shape):
    """Test the cmdline arguments to model config."""
    parser = argparse.ArgumentParser(description="test")

    parser.add_argument(
        "--model",
        choices=MODELS,
        type=str,
        required=True,
        help="name of the model to train",
    )

    parser.add_argument(
        "--input_shape",
        type=ast.literal_eval,
        default=(64, 64, 2),
        help="input shape of the image data (W, H, C)",
    )

    args = parser.parse_args(["--model", model, "--input_shape", str(input_shape)])
    cfg = config.config_from_args(args)

    assert isinstance(cfg, config.ConfigBase)
    assert args.model == model
    assert cfg.model == model
    assert args.input_shape == input_shape
    assert cfg.input_shape == input_shape
