import json

from pathlib import Path
from typing import Union

from .config import ConfigBase


def write_config_json_file(config: ConfigBase) -> None:
    """Record params of the training run."""
    # extract the params into dict:
    json_data = {prm : str(getattr(config, prm)) for prm in config.__dict__}

    # write the data into json file:
    # model = str(config.model).capitalize()
    # json_fn = config.model_dir / f'ConfigHyperParams-{model}.json'
    json_fn = config.model_dir / 'ConfigHyperParams.json'
    with open(json_fn, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)


def load_config_attributes(
    default_config: ConfigBase,
    json_params_file: Union[str, Path]
) -> ConfigBase:
    """Overwrite default config params from JSON file.

    Parameters
    ----------
    default_config : ConfigBase
        The @dataclass EncoderConfig / TemporalConfig / FullConfig
        inheriting from ConfigBase.

    json_params_file : Union[str, Path]
        The path to the JSON file storing the hyperparameters (attributes)
        of how the model was trained.

    Returns
    -------
    default_config : ConfigBase:
        The respective @dataclass with updated attributes.
    """

    # Check validity of the JSON params file:
    if isinstance(json_params_file, str):
        json_params_file = Path(json_params_file)
    assert json_params_file.is_file()
    assert str(json_params_file).endswith('.json')

    # Read the JSON file into dict:
    with open(json_params_file) as json_data:
        json_data_dict = json.load(json_data)

    # Change the attributes to the model's values:
    # default_config = EncoderConfig()  # inherits from ConfigBase()

    for attr in default_config.__dict__:

        default_value = getattr(default_config, attr)
        jsonstr_value = json_data_dict[attr]

        if attr.endswith("_dir"):
            value = Path(jsonstr_value)
        elif default_value is None:
            value = int(jsonstr_value)
        elif isinstance(default_value, (list, tuple)):
            value = eval(jsonstr_value)
        else:
            value = type(default_value)(jsonstr_value)

        setattr(default_config, attr, value)

    return default_config
