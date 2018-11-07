import os
import sys

import yaml

GLOBAL_CONF = None
TRAIN_CONF = None
SPLITS = None
META = None
DEV_PATH = None
WEIGHTS_PATH = None


def load_config(root: str, file_path: str) -> dict:
    with open(os.path.join(root, file_path), 'r') as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError as exc:
            print("error parsing file")
            sys.exit(1)


def initialize() -> dict:
    root_dir = os.path.dirname(os.path.abspath(__file__))

    global_conf = load_config(root_dir, 'config_files/global_configs.yaml')
    train_conf = load_config(root_dir, 'config_files/train_config.yaml')

    splits = os.path.join(root_dir, global_conf['files']['vox_celeb_splits'])
    meta = os.path.join(root_dir, global_conf['files']['vox_celeb_meta'])
    dev_path = os.path.join(root_dir, global_conf['files']['vox_dev_wav'])
    weights = os.path.join(root_dir, global_conf['files']['model_weights'])

    GLOBAL_CONF, TRAIN_CONF, SPLITS, META, DEV_PATH, WEIGHTS_PATH = initialize()

initialize()