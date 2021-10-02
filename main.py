import os

import yaml
from sacred import Experiment
from sacred.observers import TelegramObserver

from models.model import train_val_models
from utils import set_seed

ex = Experiment('kaggle_ex.efficient', interactive=True)
TELEGRAM_KEY = 'telegram.json'
if os.path.isfile(TELEGRAM_KEY):
    telegram_obs = TelegramObserver.from_config(TELEGRAM_KEY)
    ex.observers.append(telegram_obs)
os.environ["WANDB_START_METHOD"] = "thread"


def save_yaml(params: dict):
    with open('config/config.yaml', 'w') as stream:
        try:
            yaml.safe_dump(params, stream, default_flow_style=False)
        except yaml.YAMLError as e:
            print(e)


def load_yaml():
    with open("config/config.yaml", "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)
    return params


@ex.automain
def main():
    params = load_yaml()
    set_seed(params['seed'])
    lair_model, t1w_model, t1wce_model, t2w_model = train_val_models(params)
