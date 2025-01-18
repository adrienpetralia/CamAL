import os
import pickle
import heapq
import time
import torch

import numpy as np
import torch.nn as nn

from src.helpers.data_processing import UnderSampler
from src.helpers.torch_dataset import NILMDataset, TSDataset
from src.helpers.torch_trainer import (
    SeqToSeqTrainer,
    ClassifTrainerSigmoid,
    BasedClassifTrainer
)
from src.helpers.other import NILMmetrics

from src.models.camal.classifiers.camal_resnet import CamALResNet

from src.models.nilm_models.bigru import BiGRU
from src.models.nilm_models.unet_nilm import UNetNiLM
from src.models.nilm_models.fcn_seqtoseq import Zhang_SeqtoSeq
from src.models.nilm_models.tpnilm import TPNILM
from src.models.nilm_models.transnilm import TransNILM
from src.models.nilm_models.crnn import SCRNN, CRNN

from torch.utils.data import DataLoader


def get_log_results_if_exist(path, name_file='LogResults.pkl'):
    # Check if LogResults already exists for this expes (n_houses, window_size, sampling_rate)
    if os.path.exists(path + name_file):
        with open(path + name_file, 'rb') as handle:
            log_results = pickle.load(handle)
        print(f'LogResults already exists at provided expes path, results loaded.')
    else:
        log_results = {}

    return log_results

def save_log_results(log_results, path, name_file='LogResults.pkl'):
    path_file = path + name_file
    with open(path_file, 'wb') as f:
        pickle.dump(log_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'LogResults save at path: {path_file}')
    return


def get_nilm_instance(model_name, window_size):
    if model_name=='TPNILM':
        inst = TPNILM(in_channels=1)
        dt_params = {'model_name': model_name, 'lr': 1e-4, 'wd': 0, 'batch_size': 32, 'epochs': 30, 'p_es': 10, 'p_rlr': 5, 'n_warmup_epochs': 5, 'apply_sigmoid': True, 'training_in_model': True}
    elif model_name=='TransNILM':
        inst = TransNILM(in_channels=1)
        dt_params = {'model_name': model_name, 'lr': 1e-4, 'wd': 0, 'batch_size': 32, 'epochs': 30, 'p_es': 10, 'p_rlr': 5, 'n_warmup_epochs': 5, 'apply_sigmoid': True, 'training_in_model': True}
    elif model_name=='BiGRU':
        inst = BiGRU(c_in=1, window_size=window_size, return_values='states')
        dt_params = {'model_name': model_name, 'lr': 1e-4, 'wd': 0, 'batch_size': 32, 'epochs': 30, 'p_es': 10, 'p_rlr': 5, 'n_warmup_epochs': 5, 'apply_sigmoid': True, 'training_in_model': False}
    elif model_name =='UNET_NILM':
        inst = UNetNiLM(c_in=1, window_size=window_size, return_values='states')
        dt_params = {'model_name': model_name, 'lr': 1e-3, 'wd': 0, 'batch_size': 32, 'epochs': 30, 'p_es': 10, 'p_rlr': 5, 'n_warmup_epochs': 5, 'apply_sigmoid': True, 'training_in_model': False}
    elif model_name =='FCN':
        inst = Zhang_SeqtoSeq(c_in=1, window_size=window_size)
        dt_params = {'model_name': model_name, 'lr': 1e-4, 'wd': 0, 'batch_size': 32, 'epochs': 30, 'p_es': 10, 'p_rlr': 5, 'n_warmup_epochs': 5, 'apply_sigmoid': True, 'training_in_model': False}
    elif model_name =='SCRNN':
        inst = SCRNN(c_in=1)
        dt_params = {'model_name': model_name, 'lr': 2e-3, 'wd': 0, 'batch_size': 32, 'epochs': 30, 'p_es': 10, 'p_rlr': 5, 'n_warmup_epochs': 5, 'apply_sigmoid': False, 'training_in_model': False}
    elif 'CRNN' in model_name:
        inst = CRNN(c_in=1, return_values='frame_level')
        dt_params = {'model_name': model_name, 'lr': 2e-3, 'wd': 0, 'batch_size': 32, 'epochs': 30, 'p_es': 10, 'p_rlr': 5, 'n_warmup_epochs': 5, 'apply_sigmoid': False, 'training_in_model': True}
    else:
        raise ValueError('Model name {} unknown'.format(model_name))

    return inst, dt_params

def get_pytorch_style_dataset(expes_config,
                              model_name,
                              data_train, st_date_train,
                              data_valid, st_date_valid,
                              data_test,  st_date_test,
                              data_train_weak=None, st_date_train_weak=None):

    if model_name=='CRNNStrong':
        train_dataset = NILMDataset(data_train, X_weak=data_train_weak, return_output='strong_weak')
        valid_dataset = NILMDataset(data_valid)
        test_dataset  = NILMDataset(data_test)
    elif model_name=='CRNNWeak':
        train_dataset = NILMDataset(data_train, return_output='weak')
        valid_dataset = NILMDataset(data_valid)
        test_dataset  = NILMDataset(data_test)
    elif model_name=='CRNNStrongWeak':
        train_dataset = NILMDataset(data_train, X_weak=data_train_weak, return_output='strong_weak')
        valid_dataset = NILMDataset(data_valid)
        test_dataset  = NILMDataset(data_test)
    elif model_name=='TPNILM' or model_name=='TransNILM':
        train_dataset = NILMDataset(data_train, X_weak=data_train_weak, padding_output_seq=15)
        valid_dataset = NILMDataset(data_valid, padding_output_seq=15)
        test_dataset  = NILMDataset(data_test,  padding_output_seq=15)
    else:
        train_dataset = NILMDataset(data_train, X_weak=data_train_weak)
        valid_dataset = NILMDataset(data_valid)
        test_dataset  = NILMDataset(data_test)

    return train_dataset, valid_dataset, test_dataset

def get_resnet_instance(resnet_name, kernel_size, **kwargs):
    if resnet_name == 'CamALResNet':
        inst = CamALResNet(kernel_size=kernel_size, **kwargs)
    else:
        raise ValueError('ResNet name {} unknown'.format(resnet_name))

    return inst

def get_resnet_layers(resnet_name, resnet_inst):
    if 'ResNet3' in resnet_name:
        last_conv_layer = resnet_inst._modules['layers'][2]
        fc_layer_name   = resnet_inst._modules['linear']
    elif 'ResNet5' in resnet_name:
        last_conv_layer = resnet_inst._modules['layers'][4]
        fc_layer_name   = resnet_inst._modules['linear']
    elif 'CamALResNet' in resnet_name:
        last_conv_layer = resnet_inst._modules['layers'][2]
        fc_layer_name   = resnet_inst._modules['linear']
    else:
        raise ValueError('ResNet name {} unknown'.format(resnet_name))

    return last_conv_layer, fc_layer_name


def nilm_model_inference(expes_config, 
                         path, 
                         model, dt_params,
                         test_dataset,
                         pos_weight=1,
                         compress_and_save_space=True):

    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=1, shuffle=False)

    model_trainer = SeqToSeqTrainer(model,
                                    train_loader=test_loader, valid_loader=test_loader,
                                    learning_rate=dt_params['lr'], weight_decay=dt_params['wd'],
                                    criterion=nn.BCELoss(),
                                    f_metrics=NILMmetrics(),
                                    training_in_model=dt_params['training_in_model'],
                                    consumption_pred=expes_config['consumption_pred'], apply_sigmoid=dt_params['apply_sigmoid'],
                                    patience_es=dt_params['p_es'], patience_rlr=dt_params['p_rlr'],
                                    n_warmup_epochs=dt_params['n_warmup_epochs'],
                                    verbose=True, plotloss=False, 
                                    save_fig=False, path_fig=None,
                                    device=expes_config['device'], all_gpu=expes_config['all_gpu'],
                                    save_checkpoint=True, path_checkpoint=path)

    model_trainer.evaluate(test_loader,  scaler=expes_config['scaler'], 
                           appliance_mean_on_power=expes_config['appliance_mean_on_power'])

    if compress_and_save_space:
        # Attention, model logs compressed! -> file end with ".xz" not ".pt"
        model_trainer.compress()

    if not expes_config['save_model_weights']:
        model_trainer.delete()
        print('Model log deleted.')
    elif 'FCN' in path:
        model_trainer.delete()
        print('Model FCN log deleted.')

    print(model_trainer.log['test_metrics_timestamp'])

    metrics = model_trainer.log['test_metrics_timestamp']

    return metrics


def nilm_model_training(expes_config, 
                        path, 
                        model, dt_params,
                        train_dataset, 
                        valid_dataset, 
                        test_dataset,
                        compress_and_save_space=True):
        

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=dt_params['batch_size'], shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=1, shuffle=False)

    if not expes_config['consumption_pred']:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    model_trainer = SeqToSeqTrainer(model,
                                    train_loader=train_loader, valid_loader=valid_loader,
                                    learning_rate=dt_params['lr'], weight_decay=dt_params['wd'],
                                    criterion=criterion,
                                    f_metrics=NILMmetrics(),
                                    training_in_model=dt_params['training_in_model'],
                                    consumption_pred=expes_config['consumption_pred'], apply_sigmoid=dt_params['apply_sigmoid'],
                                    patience_es=dt_params['p_es'], patience_rlr=dt_params['p_rlr'],
                                    n_warmup_epochs=dt_params['n_warmup_epochs'],
                                    verbose=True, plotloss=False, 
                                    save_fig=False, path_fig=None,
                                    device=expes_config['device'], all_gpu=expes_config['all_gpu'],
                                    save_checkpoint=True, path_checkpoint=path)


    model_trainer.train(dt_params['epochs'])
    model_trainer.restore_best_weights()
    model_trainer.evaluate(test_loader,  scaler=expes_config['scaler'], 
                           appliance_mean_on_power=expes_config['appliance_mean_on_power'])

    if compress_and_save_space:
        # Attention, model logs compressed! -> file end with ".xz" not ".pt"
        model_trainer.compress()

    if not expes_config['save_model_weights']:
        model_trainer.delete()
        print('Model log deleted.')
    elif 'FCN' in path:
        model_trainer.delete()
        print('Model FCN log deleted.')

    print(model_trainer.log['test_metrics_timestamp'])

    metrics = model_trainer.log['test_metrics_timestamp']
    metrics['TrainingTime'] = model_trainer.log['training_time']

    return metrics


def train_crnnweak_possession(expes_config,
                              path,
                              train_dataset,
                              valid_dataset,
                              test_dataset,
                              balance_class=True,
                              compress_and_save_space=False):
    """
    Inputs:
    - expes_config: [dict] with expes parameters
    - path: [string] path to save ResNets clf instances
    - train_dataset: [tuple] such as (X_train, y_train)
    - valid_dataset: [tuple] such as (X_valid, y_valid)
    - test_dataset:  [tuple] such as (X_test, y_test)

    Output:
    - dict_ens_results: [dict] with all infos (best models, loss, metrics)
    """

    X_train, y_train = train_dataset[0], train_dataset[1]
    X_valid, y_valid = valid_dataset[0], valid_dataset[1]
    X_test,  y_test  = test_dataset[0],  test_dataset[1]

    # Balance data class for training
    if balance_class:
        X_train, y_train = UnderSampler(X_train, y_train, sampling_strategy="auto", seed=0)
    
    # Create dataset
    train_dataset = TSDataset(X_train, y_train)
    valid_dataset = TSDataset(X_valid, y_valid)
    test_dataset  = TSDataset(X_test,  y_test)

    crnn_instance, dt_params = get_nilm_instance('CRNN', 1000)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1,  shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=1,  shuffle=False)

    dict_results = {}

    print('Train CRNN Weak on possession clf ensemble')
    tmp_time = time.time()

    # Init trainer
    crnn_instance.return_values = 'bag_level'

    clf_trainer = ClassifTrainerSigmoid(crnn_instance,
                                        train_loader=train_loader, valid_loader=valid_loader,
                                        learning_rate=dt_params['lr'], weight_decay=dt_params['wd'],
                                        patience_es=dt_params['p_es'], patience_rlr=dt_params['p_rlr'],
                                        n_warmup_epochs=2, # n_warmup_epochs=dt_params['n_warmup_epochs'],
                                        device="cuda", all_gpu=True,
                                        verbose=True, plotloss=False, 
                                        save_checkpoint=True, path_checkpoint=f'{path}CRNNWeakClf')

    # Train 
    #clf_trainer.train(dt_params['epochs'])
    clf_trainer.train(15)

    # Eval
    clf_trainer.restore_best_weights()
    loss_eval, metrics = clf_trainer.evaluate(test_loader)

    dict_results[f'CRNNClfTraining'] = {'LossTest': loss_eval, 'MetricTest': metrics}

    print('Loss test:', loss_eval)
    print('Score:', metrics)

    dict_results['TrainingTime'] = round((time.time() - tmp_time), 3)

    if compress_and_save_space:
        # Attention, model logs compressed! -> file end with ".xz" not ".pt"
        clf_trainer.compress()

    crnn_instance.return_values = 'frame_level'

    return dict_results, crnn_instance, dt_params


def train_resnet_ensemble(expes_config,
                          path,
                          train_dataset,
                          valid_dataset,
                          test_dataset,
                          kernel_size_list=[5, 7, 9],
                          n_try_clf=3,
                          balance_class=True,
                          compress_and_save_space=True):
    """
    Inputs:
    - expes_config: [dict] with expes parameters
    - path: [string] path to save ResNets clf instances
    - train_dataset: [tuple] such as (X_train, y_train)
    - valid_dataset: [tuple] such as (X_valid, y_valid)
    - test_dataset:  [tuple] such as (X_test, y_test)

    Output:
    - dict_ens_results: [dict] with all infos (best models, loss, metrics)
    """

    dt_params = {'lr': 1e-4 if 'LN' in expes_config['resnet_name'] else 1e-3, 'wd': 0, 'batch_size': 128, 'epochs': 50, 'p_es': 5, 'p_rlr': 3, 'n_warmup_epochs': 1}

    X_train, y_train = train_dataset[0], train_dataset[1]
    X_valid, y_valid = valid_dataset[0], valid_dataset[1]
    X_test,  y_test  = test_dataset[0],  test_dataset[1]

    # Balance data class for training
    if balance_class:
        X_train, y_train = UnderSampler(X_train, y_train, sampling_strategy="auto", seed=0)
    
    # Create dataset
    train_dataset = TSDataset(X_train, y_train)
    valid_dataset = TSDataset(X_valid, y_valid)
    test_dataset  = TSDataset(X_test,  y_test)

    # Init dicts result
    tmp_dict_results = {}
    dict_results     = {}

    print('Train ResNet clf ensemble')
    tmp_time = time.time()

    idx_clf = 0
    for kernel_size in kernel_size_list:
        for i in range(n_try_clf):

            resnet_inst = get_resnet_instance(expes_config['resnet_name'], kernel_size=kernel_size)

            # Dataloader
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=dt_params['batch_size'], shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1,  shuffle=False)
            test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=1,  shuffle=False)

            # Init trainer
            clf_trainer = BasedClassifTrainer(resnet_inst,
                                             train_loader=train_loader, valid_loader=valid_loader,
                                             learning_rate=dt_params['lr'], weight_decay=dt_params['wd'],
                                             patience_es=dt_params['p_es'], patience_rlr=dt_params['p_rlr'],
                                             n_warmup_epochs=dt_params['n_warmup_epochs'],
                                             device="cuda", all_gpu=True,
                                             verbose=False, plotloss=False, 
                                             save_checkpoint=True, path_checkpoint=f'{path}ResNet_{idx_clf}')

            # Train 
            clf_trainer.train(dt_params['epochs'])

            # Eval
            clf_trainer.restore_best_weights()
            loss_eval, metrics = clf_trainer.evaluate(test_loader)

            if compress_and_save_space:
                # Attention, model logs compressed! -> file end with ".xz" not ".pt"
                clf_trainer.compress()

            print(f'Trained ResNet_{idx_clf} with Kernel size: {kernel_size} (nth_try {i}).')
            print('Loss test:', loss_eval)
            print('Score:', metrics)

            # Save loss value
            tmp_dict_results[f'ResNet_{idx_clf}'] = loss_eval
            dict_results[f'ResNet_{idx_clf}']     = {'LossTest': loss_eval, 'MetricTest': metrics, 'kernel_size': kernel_size, 'nth_try': i}

            # Update counters
            idx_clf +=1

    dict_results['TrainingTime'] = round((time.time() - tmp_time), 3)
    
    # Sort clf by performance according to test loss score [TODO! add sorting by metric: Acc? F1?]
    heap = [(loss, name) for name, loss in tmp_dict_results.items()]
    heapq.heapify(heap)
    smallest = heapq.nsmallest(expes_config['n_best_clf'], heap)

    # Get the name of the best clf and select them for final ensemble
    list_best_clf = [name for _, name in smallest]
    print('List best clf:', list_best_clf)
    dict_results['ListBestResNets'] = list_best_clf

    # Delete the clfs for the ensemble (worst performance)
    print('Delete worst clf...')
    if compress_and_save_space:
        for i in range(idx_clf):
            if not (f'ResNet_{i}' in list_best_clf):
                os.remove(f'{path}ResNet_{i}.xz')
                print(f'ResNet_{i} removed.')
    print('Done.')

    # Save ensemble dict_results in pkl file at provided path
    name_file = f'{path}LogResNetsEnsemble.pkl'
    print(f'Save ResNets ensemble log at path: {path}LogResNetsEnsemble.pkl')
    with open(name_file, 'wb') as f:
        pickle.dump(dict_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    return dict_results




class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.samples = X
            
        if len(self.samples.shape)==2:
            self.samples = np.expand_dims(self.samples, axis=1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        return self.samples[idx]


def get_window_labels(model, data, batch_size, device='cuda'):
    model.to(device)
    model.eval()
    data   = SimpleDataset(data)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    y_hat = np.array([])
    with torch.no_grad():
        for ts in loader:
            logits       = model(torch.Tensor(ts.float()).to(device))
            _, predicted = torch.max(logits, 1)

            y_hat = np.concatenate((y_hat, predicted.detach().cpu().numpy().flatten())) if y_hat.size else predicted.detach().cpu().numpy().flatten()

    return y_hat.astype(np.int8)
