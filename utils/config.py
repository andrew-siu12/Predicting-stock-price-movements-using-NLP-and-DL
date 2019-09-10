import json
from dotmap import DotMap
import os
import time


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = DotMap(config_dict)

    return config, config_dict


def process_config(json_file):
    """
    Add callbacks argument to config
    :param json_file:
    :return:  config(namespace)
    """
    config, _ = get_config_from_json(json_file)
    config.callbacks.tensorboard_log_dir = os.path.join("experiments", config.exp_name, "logs/")
    config.callbacks.checkpoint_dir = os.path.join("experiments", config.exp_name, "checkpoints/")
    return config
