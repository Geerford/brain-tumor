import wandb
import os

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


@ex.automain
def main():
    wandb.init(
        project="RSNA-MICCAI",
        tags=["effnetb0", "model4seq"],
        config={
            'epochs': 20,
            'train_batch_size': 8,
            'learning_rate': 1e-5,
            'optimizer_type': 'Adam',
    })
    config = wandb.config

    # CONFIGS
    params = {
        'data_directory': 'dataset',
        'train_directory': 'dataset\\train\\',
        'test_directory': 'dataset\\test\\',
        'save_directory': 'models\\experiments\\',
        'last_checkpoint': None,
        'csv': 'train_labels.csv',
        'seed': 21,
        # Data configs
        'train_val_size': 0.2,  # Train: 465, Val: 117
        'mri_types': ['FLAIR', 'T1w', 'T1wCE', 'T2w'],
        'type': 'train',
        'sample_format': '.png',
        'image_read_mode': 'grayscale',
        'input_size': 256,
        'num_images_seq': 64,
        'transform': False,
        # Loader configs
        'train_batch_size': 8,
        'test_batch_size': 1,
        'num_workers': 0,
        'pin_memory': True,
        'train_shuffle': True,
        'test_shuffle': False,
        'train_drop_last': True,
        'test_drop_last': False,
        # Net configs
        'model_type': 'efficientnet-b0',
        'num_classes': 2,
        'input_channels': 4,

        'learning_rate': 0.0001,
        'optimizer_type': 'Adam',
        'optimizer_weight_decay': 0.0005,
        'meta_features': ['pct'],
        'freeze_cnn': True,
        'schedule_type': 'StepLR',
        'lower_lr_after': 25,
        'lr_step': 5,
        'lr_max': 0.1,

        'epochs': 20,

        'learning_rate_meta': 0.0001,

        # TEST

    }
    set_seed(params['seed'])

    params['epochs'] = config.epochs
    params['train_batch_size'] = config.train_batch_size
    params['learning_rate'] = config.learning_rate
    params['optimizer_type'] = config.optimizer_type

    lair_model, t1w_model, t1wce_model, t2w_model = train_val_models(params)

    wandb.finish()
