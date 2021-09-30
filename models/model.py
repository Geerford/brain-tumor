import os

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils import data
from tqdm import tqdm

from dataset.dataset import BrainTumorDataset
from models import model_list


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
        return lr_scheduler.StepLR(optimizer, step_size=params['lower_lr_after'], gamma=gamma)
    elif params['schedule_type'] == 'MultiStepLR':
        return lr_scheduler.MultiStepLR(optimizer, gamma=gamma)
    elif params['schedule_type'] == 'CyclicLR':
        return lr_scheduler.CyclicLR(optimizer, base_lr=params['learning_rate'], max_lr=params['lr_max'], gamma=gamma)
    else:
        return lr_scheduler.ExponentialLR(optimizer, gamma=gamma)


def train_val_model(params: dict):
    df_labels = pd.read_csv(f"{params['data_directory']}/{params['csv']}")
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
        params=params)
    val_dataset = BrainTumorDataset(
        labels=df_val,
        root_dir=params['train_directory'],
        params=params)
    print(f"Train dataset length: {len(train_dataset)}; Validation dataset length: {len(val_dataset)}")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params['train_batch_size'],
        num_workers=params['num_workers'],
        pin_memory=params['pin_memory'],
        shuffle=params['train_shuffle'],
        drop_last=params['train_drop_last'])
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=params['test_batch_size'],
        num_workers=params['num_workers'],
        pin_memory=params['pin_memory'],
        shuffle=params['test_shuffle'],
        drop_last=params['test_drop_last'])

    params['schedule_max_iter'] = len(train_loader)

    model = model_list.efficientnet_3d(params).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, params)
    scheduler = create_scheduler(optimizer, params)

    best_roc = 0.0
    for epoch in tqdm(range(params['epochs']), desc='Epoch'):
        ################################################################################################################
        # Train
        ################################################################################################################
        model.train()

        train_roc = 0.0
        train_labels = []
        train_outputs = []
        train_loss = []
        for inputs, labels in train_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_labels.extend(labels.tolist())
            train_outputs.extend(outputs.tolist())
            train_loss.append(loss.item())

            wandb.log({"train batch loss": loss.item()})
        train_roc += roc_auc_score([[1, 0] if a_i == 0 else [0, 1] for a_i in train_labels], train_outputs)
        wandb.log({"train epoch loss": sum(train_loss) / params['train_batch_size']})
        print(f"Epoch: {epoch}. Train loss: {sum(train_loss) / params['train_batch_size']}. Train ROC: {train_roc}")
        ################################################################################################################
        # Validation
        ################################################################################################################
        model.eval()

        val_roc = 0.0
        val_labels = []
        val_outputs = []
        val_loss = []
        for inputs, labels in val_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_labels.extend(labels.tolist())
            val_outputs.extend(outputs.tolist())
            val_loss.append(loss.item())

            wandb.log({"val batch loss": loss.item()})
        val_roc += roc_auc_score([[1, 0] if a_i == 0 else [0, 1] for a_i in val_labels], val_outputs)
        wandb.log({"val epoch loss": sum(val_loss)})
        print(f'Epoch: {epoch}. Val loss: {sum(val_loss)}. Val ROC: {val_roc}')

        scheduler.step()

        ################################################################################################################
        # Save
        ################################################################################################################
        params['last_checkpoint'] = epoch
        torch.save(model.state_dict(),
                   os.path.join(params['save_directory'], f"{params['model_type']}_checkpoint_{epoch}.pt"))

        # best model
        if val_roc > best_roc:
            best_roc = val_roc
            torch.save(model.state_dict(),
                       os.path.join(params['save_directory'], f"{params['model_type']}_checkpoint_{epoch}_best.pt"))
    return model


def train_val_models(params: dict):
    params['seq_type'] = 'FLAIR'
    flair_model = train_val_model(params)
    params['seq_type'] = 'T1w'
    t1w_model = train_val_model(params)
    params['seq_type'] = 'T1wCE'
    t1wce_model = train_val_model(params)
    params['seq_type'] = 'T2w'
    t2w_model = train_val_model(params)
    return flair_model, t1w_model, t1wce_model, t2w_model


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

    model = model_list.efficientnet_3d(params).cuda()
    model.load_state_dict(
        torch.load(f"{params['save_directory']}{params['model_type']}_checkpoint_{params['last_checkpoint']}.pt"))
    model.cuda()
    model.eval()

    idxs = []
    preds = []

    with torch.no_grad():
        for inputs, idx in tqdm(test_loader):
            inputs = inputs.cuda()
            output = model(inputs)
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
            pred = pred[1]
            # print(idx, pred)
            idx = int(idx[0])

            idxs.append(idx)
            preds.append(pred)
    preddf = pd.DataFrame({"BraTS21ID": idxs, "MGMT_value": preds})
    preddf = preddf.set_index("BraTS21ID")
