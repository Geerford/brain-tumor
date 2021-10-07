import os

from sacred import Experiment
from sacred.observers import TelegramObserver

from models.model import train_val_models, predict
from utils import set_seed, load_yaml

ex = Experiment('kaggle_ex.efficient', interactive=True)
TELEGRAM_KEY = 'telegram.json'
if os.path.isfile(TELEGRAM_KEY):
    telegram_obs = TelegramObserver.from_config(TELEGRAM_KEY)
    ex.observers.append(telegram_obs)
os.environ["WANDB_START_METHOD"] = "thread"


@ex.automain
def main():
    params = load_yaml()
    set_seed(params['seed'])
    train_val_models(params)
    pred = predict(params)
    pred.to_csv('submission.csv', index=False)
