import os

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils.data import ConcatDataset, SubsetRandomSampler
from tqdm import tqdm

from dataset.dataset import BrainTumorDataset
from models import model_list
from utils import save_yaml


def get_optimizer(init_params: dict, params: dict):
    if params['optimizer_type'] == 'SGD':
        return optim.SGD(init_params,
                         lr=init_params["lr"],
                         momentum=0.9,
                         weight_decay=params["optimizer_weight_decay"],
                         nesterov=True)
    elif params['optimizer_type'] == 'Adam':
        return optim.Adam(init_params['init_params'],
                          lr=init_params['lr'],
                          weight_decay=params["optimizer_weight_decay"])
    elif params['optimizer_type'] == "AdamW":
        return optim.AdamW(init_params,
                           lr=init_params["lr"],
                           weight_decay=params["optimizer_weight_decay"])
    elif params['optimizer_type'] == "RmsProp":
        return optim.RMSprop(init_params,
                             lr=init_params["lr"],
                             weight_decay=params["optimizer_weight_decay"])


def create_optimizer(model, params: dict):
    if params.get('meta_features', None) is not None:
        if params['freeze_cnn']:
            init_optimizer_params = {
                'init_params': filter(lambda p: p.requires_grad, model.parameters()),
                'lr': params['learning_rate_meta']
            }
            # [print(param.name, param.shape) for param in filter(lambda p: p.requires_grad, model.parameters())]
            return get_optimizer(init_optimizer_params, params)

        else:
            init_optimizer_params = {
                'init_params': [
                    {
                        'params': filter(lambda p: not p.is_cnn_param, model.parameters()),
                        'lr': params['learning_rate_meta']
                    },
                    {
                        'params': filter(lambda p: p.is_cnn_param, model.parameters()),
                        'lr': params['learning_rate']
                    }
                ],
                'lr': params['learning_rate_meta']
            }
            return get_optimizer(init_optimizer_params, params)
    else:
        init_optimizer_params = {
            'init_params': model.parameters(),
            'lr': model['learning_rate']
        }
        return get_optimizer(init_optimizer_params, params)


def create_scheduler(optimizer, params: dict):
    gamma = 1 / np.float32(params['lr_step'])
    if params['schedule_type'] == 'StepLR':
        return lr_scheduler.StepLR(optimizer, step_size=params['step_size'], gamma=gamma)
    elif params['schedule_type'] == 'MultiStepLR':
        return lr_scheduler.MultiStepLR(optimizer, gamma=gamma)
    elif params['schedule_type'] == 'CyclicLR':
        return lr_scheduler.CyclicLR(optimizer, base_lr=params['learning_rate'], max_lr=params['lr_max'], gamma=gamma)
    else:
        return lr_scheduler.ExponentialLR(optimizer, gamma=gamma)


def train_epoch(model, loader, criterion, optimizer):
    model.train()

    train_loss, train_correct = 0.0, 0
    train_labels, train_outputs = [], []

    for images, labels in loader:
        images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        scores, predictions = torch.max(output.data, 1)
        batch_loss = loss.item() * images.size(0)
        train_loss += batch_loss
        batch_correct = sum(predictions == labels).item()
        train_correct += batch_correct

        train_labels.extend(labels.tolist())
        train_outputs.extend(output.tolist())
        wandb.log({"train batch loss": batch_loss})
        wandb.log({"train batch correct": batch_correct})

    train_roc = roc_auc_score([[1, 0] if a_i == 0 else [0, 1] for a_i in train_labels], train_outputs)

    return train_loss, train_correct, train_roc


def valid_epoch(model, loader, criterion, optimizer):
    model.eval()

    valid_loss, valid_correct = 0.0, 0
    valid_labels, valid_outputs = [], []

    for images, labels in loader:
        images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)

        scores, predictions = torch.max(output.data, 1)
        batch_loss = loss.item() * images.size(0)
        valid_loss += batch_loss
        batch_correct = sum(predictions == labels).item()
        valid_correct += batch_correct

        valid_labels.extend(labels.tolist())
        valid_outputs.extend(output.tolist())
        wandb.log({"valid batch loss": batch_loss})
        wandb.log({"valid batch correct": batch_correct})

    valid_roc = roc_auc_score([[1, 0] if a_i == 0 else [0, 1] for a_i in valid_labels], valid_outputs)
    return valid_loss, valid_correct, valid_roc


def cross_validation(params: dict):
    df_labels = pd.read_csv(os.path.join(params['data_directory'], params['csv']))

    # Issues id
    for ids in [109, 123, 709]:
        df_labels.drop(df_labels[df_labels["BraTS21ID"] == ids].index, inplace=True)

    df_train, df_val = train_test_split(
        df_labels,
        test_size=params['train_val_size'],
        random_state=params['seed'],
        stratify=df_labels["MGMT_value"],
    )

    train_dataset = BrainTumorDataset(
        labels=df_train,
        root_dir=params['train_directory'],
        params=params
    )
    valid_dataset = BrainTumorDataset(
        labels=df_val,
        root_dir=params['train_directory'],
        params=params
    )
    dataset = ConcatDataset([train_dataset, valid_dataset])

    model = model_list.efficientnet(params).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, params)
    scheduler = create_scheduler(optimizer, params)

    folds_history = {}
    if params['fold_type'] == 'StratifiedKFold':
        splitter = StratifiedKFold(n_splits=params['k_fold'], shuffle=True,
                                   random_state=params['seed']).split(df_labels['BraTS21ID'], df_labels['MGMT_value'])
    else:
        splitter = KFold(n_splits=params['k_fold'], shuffle=True,
                         random_state=params['seed']).split(np.arange(len(dataset)))
    for fold, (train_idx, valid_idx) in enumerate(splitter):
        wandb.init(
            project="RSNA-MICCAI",
            tags=["effnetb0"],
            group=params['seq_type'],
            notes=f"cross-validation each model",
            job_type=f"fold{fold + 1}",
            config=params)
        print(f'\033[4mFold {fold + 1}\033[0m')
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=params['train_batch_size'],
                                                   sampler=train_sampler,
                                                   num_workers=params['num_workers'],
                                                   pin_memory=params['pin_memory'])
        valid_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=params['test_batch_size'],
                                                   sampler=valid_sampler,
                                                   num_workers=params['num_workers'],
                                                   pin_memory=params['pin_memory'])

        history = {
            'train_epoch_loss': [],
            'valid_epoch_loss': [],
            'train_epoch_acc': [],
            'valid_epoch_acc': [],
            'train_epoch_roc': [],
            'valid_epoch_roc': []
        }

        best_roc = 0.0
        print(f'\n\033[4m{"Epoch": <50}{"Loss": >42}{"Accuracy": >10}{"ROC": >5}{"": <1}\033[0m')
        for epoch in range(params['epochs']):
            train_loss, train_correct, train_roc = train_epoch(model, train_loader, criterion, optimizer)
            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            print(f'train_{epoch + 1:<50}{train_loss:>36.3f},{train_acc:>8.3f},{train_roc:>6.3f}')
            valid_loss, valid_correct, valid_roc = valid_epoch(model, valid_loader, criterion, optimizer)
            valid_loss = valid_loss / len(valid_loader.sampler)
            valid_acc = valid_correct / len(valid_loader.sampler) * 100
            print(f'valid_{epoch + 1:<50}{valid_loss:>36.3f},{valid_acc:>8.3f},{valid_roc:>6.3f}')
            scheduler.step()

            history['train_epoch_loss'].append(train_loss)
            history['valid_epoch_loss'].append(valid_loss)
            history['train_epoch_acc'].append(train_acc)
            history['valid_epoch_acc'].append(valid_acc)
            history['train_epoch_roc'].append(train_roc)
            history['valid_epoch_roc'].append(valid_roc)
            wandb.log({f"train_epoch_loss": train_loss})
            wandb.log({f"train_epoch_acc": train_acc})
            wandb.log({f"train_epoch_roc": train_roc})
            wandb.log({f"valid_epoch_loss": valid_loss})
            wandb.log({f"valid_epoch_acc": valid_acc})
            wandb.log({f"valid_epoch_roc": valid_roc})
            folds_history[f'fold_{fold}'] = history

            # Save each epoch
            params.update({
                'last_checkpoint': f"{fold}_{epoch}"
            })
            save_yaml(params)

            save_path = os.path.join(params['save_directory'], params['seq_type'])
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(save_path, f"{params['model_type']}_{params['last_checkpoint']}.pt"))

            # Save best model
            if valid_roc > best_roc:
                best_roc = valid_roc
                torch.save(model.state_dict(),
                           os.path.join(save_path, f"{params['model_type']}_best.pt"))
        wandb.log({f"train_fold_loss": np.mean(history['train_epoch_loss'])})
        wandb.log({f"train_fold_acc": np.mean(history['train_epoch_acc'])})
        wandb.log({f"train_fold_roc": np.mean(history['train_epoch_roc'])})
        wandb.log({f"valid_fold_loss": np.mean(history['valid_epoch_loss'])})
        wandb.log({f"valid_fold_acc": np.mean(history['valid_epoch_acc'])})
        wandb.log({f"valid_fold_roc": np.mean(history['valid_epoch_roc'])})

        wandb.finish()
    return model


def train_val_models(params: dict):
    params['seq_type'] = 'FLAIR'
    flair_model = cross_validation(params)
    params['seq_type'] = 'T1w'
    t1w_model = cross_validation(params)
    params['seq_type'] = 'T1wCE'
    t1wce_model = cross_validation(params)
    params['seq_type'] = 'T2w'
    t2w_model = cross_validation(params)
    return flair_model, t1w_model, t1wce_model, t2w_model


def predict(params: dict):
    params['seq_type'] = 'FLAIR'
    flair_pred = load_model(params)
    params['seq_type'] = 'T1w'
    t1w_pred = load_model(params)
    params['seq_type'] = 'T1wCE'
    t1wce_pred = load_model(params)
    params['seq_type'] = 'T2w'
    t2w_pred = load_model(params)
    return (flair_pred + t1w_pred + t1wce_pred + t2w_pred) / 4


def load_model(params: dict):
    params['type'] = 'test'
    test_dataset = BrainTumorDataset(
        labels=pd.DataFrame(),
        root_dir=params['test_directory'],
        params=params)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=params['test_batch_size'],
        num_workers=params['num_workers'],
        pin_memory=params['pin_memory'],
        shuffle=params['test_shuffle'],
        drop_last=params['test_drop_last'])

    model = model_list.efficientnet(params).cuda()
    load_path = os.path.join(params['save_directory'], params['seq_type'], f"{params['model_type']}_best.pt")
    # load_path = os.path.join(params['save_directory'], params['seq_type'], f"{params['model_type']}_4_19.pt")

    model.load_state_dict(torch.load(load_path))
    model.cuda()
    model.eval()

    pred = pd.DataFrame(columns=['BraTS21ID', 'MGMT_value'])
    with torch.no_grad():
        for inputs, idx in tqdm(test_loader, desc=f"Prediction {params['seq_type']}"):
            inputs = inputs.cuda()
            output = model(inputs)
            pred = pred.append(pd.Series([idx.numpy(), torch.sigmoid(output).squeeze().cpu().numpy()[1]], index=pred.columns),
                               ignore_index=True)
        pred['BraTS21ID'] = pred['BraTS21ID'].astype(int)
        pred = pred.set_index('BraTS21ID')
    return pred
