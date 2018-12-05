import os
import sys
import yaml


def load_config(root: str, file_path: str) -> dict:
    with open(os.path.join(root, file_path), 'r') as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError as exc:
            print("error parsing file")
            sys.exit(1)


def initialize() -> tuple:
    root_dir = os.path.dirname(os.path.abspath(__file__))

    global_conf = load_config(root_dir, 'config_files/global_configs.yaml')
    train_conf = load_config(root_dir, 'config_files/train_config.yaml')

    splits = os.path.join(root_dir, global_conf['files']['vox_celeb_splits'])
    meta = os.path.join(root_dir, global_conf['files']['vox_celeb_meta'])
    vox_dev_wav = os.path.join(root_dir, global_conf['files']['vox_dev_wav'])
    vox_test_wav = os.path.join(root_dir, global_conf['files']['vox_test_wav'])
    weights_path = os.path.join(root_dir, global_conf['files']['weights_path'])
    log_dir = os.path.join(root_dir, global_conf['files']['log_dir'])

    return global_conf, train_conf, splits, meta, vox_dev_wav, vox_test_wav, weights_path, log_dir


GLOBAL_CONF, TRAIN_CONF, SPLITS, META, VOX_DEV_WAV, VOX_TEST_WAV, WEIGHTS_PATH, LOG_DIR = initialize()
