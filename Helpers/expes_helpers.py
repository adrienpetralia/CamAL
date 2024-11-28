import os, sys, lzma, io
import numpy as np
import pandas as pd
import pickle
import heapq
import time

from Helpers.class_activation_map import *
from Helpers.data_processing import *
from Helpers.torch_dataset import *
from Helpers.torch_trainer import *
from Helpers.other import *

from Models.Classifiers.CamALResNet import CamALResNet

from Models.Sota.BiGRU import BiGRU
from Models.Sota.UNET_NILM import UNetNiLM
from Models.Sota.Zhang_SeqtoSeq import Zhang_SeqtoSeq
from Models.Sota.TPNILM import TPNILM
from Models.Sota.TransNILM import TransNILM
from Models.Sota.CRNN import SCRNN, CRNN

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
        X_train, y_train = RandomUnderSampler_(X_train, y_train, sampling_strategy="auto", seed=0)
    
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
        X_train, y_train = RandomUnderSampler_(X_train, y_train, sampling_strategy="auto", seed=0)
    
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
    

def SigmoidAttention(data, soft_label, w_ma=5):
    # Apply moving average
    soft_label = np.apply_along_axis(lambda x: moving_average(x, w=w_ma), axis=1, arr=soft_label)

    # Apply sigmoid on the product of averaged soft labels
    soft_label = sigmoid(soft_label * data)

    # Thresholding to obtain binary labels
    return np.round(soft_label)


def create_soft_label(expes_config,
                      path_ensemble_clf,
                      data,
                      y=None,
                      return_prob=False):

    assert len(data.shape)<=2, f'dat input need to be 2D numpy array for create_soft_label, got a {data.shape}D array.'

    if len(data.shape)==1:
        data = np.expand_dims(data, axis=0)

    with open(f'{path_ensemble_clf}LogResNetsEnsemble.pkl', 'rb') as handle:
        dict_results = pickle.load(handle)

    list_best_resnets = dict_results['ListBestResNets']

    soft_label  = np.zeros_like(data)
    prob_detect = np.zeros((len(data), 1))

    # Loop on BestResNets 
    for resnet_name in list_best_resnets:
        resnet_inst = get_resnet_instance(expes_config['resnet_name'], dict_results[resnet_name]['kernel_size'])
        resnet_inst.to(expes_config['device'])

        if os.path.exists(f'{path_ensemble_clf}{resnet_name}.xz'):
            path_model = f'{path_ensemble_clf}{resnet_name}.xz'
            with lzma.open(path_model, 'rb') as file:
                decompressed_file = file.read()
            log = torch.load(io.BytesIO(decompressed_file), map_location=torch.device(expes_config['device']))
            del decompressed_file
        elif os.path.exists(f'{path_ensemble_clf}{resnet_name}.pt'):
            path_model = f'{path_ensemble_clf}{resnet_name}.pt'
            log = torch.load(path_model, map_location=torch.device(expes_config['device']))
        else:
            raise ValueError(f'Provide folders {path_ensemble_clf} does not contain {resnet_name} clf.')

        resnet_inst.load_state_dict(log['model_state_dict'])
        resnet_inst.eval()

        # Get first the per window prediction labels using large batch
        if (expes_config['device']!='cpu') and (y is None):
            batch_size = expes_config['batch_inference'] if 'batch_inference' in expes_config else 128
            y = get_window_labels(resnet_inst, data, batch_size=batch_size, device=expes_config['device'])

        last_conv_layer, fc_layer_name = get_resnet_layers(expes_config['resnet_name'], resnet_inst)

        for idx in range(len(data)): 
            # Use provide window label to speed up (by not computing the CAM)
            if y is not None:
                if y[idx]<1:
                    continue

            CAM_builder = CAM(model=resnet_inst, device=expes_config['device'], 
                              last_conv_layer=last_conv_layer, fc_layer_name=fc_layer_name)
            
            cam, y_pred, proba   = CAM_builder.run(instance=data[idx], returned_cam_for_label=1)
            prob_detect[idx][0] += proba[1]

            # If y is not None, ground true is used
            # if y is not None:
            #     # Clip CAM and MaxNormalization (between 0 and 1)
            #     clip_cam = np.clip(cam, a_min=0, a_max=None)
            #     clip_cam = clip_cam / clip_cam.max()
                
            # Or if app detected in this window
            if y_pred>0:
                # Clip CAM and MaxNormalization (between 0 and 1)
                clip_cam = np.clip(cam, a_min=0, a_max=None)
                clip_cam = np.nan_to_num(clip_cam.astype(np.float32), nan=0.0, neginf=0.0, posinf=0.0)

                if clip_cam.max()>0:
                    clip_cam = clip_cam / clip_cam.max()
                else:
                    clip_cam = np.zeros_like(clip_cam)

                soft_label[idx] += clip_cam.ravel()

        del resnet_inst
        
    # Majority voting: if the ensemble probability of detection < 0.5 appliance not detected in wins, soft label set to 0
    prob_detect = prob_detect / len(list_best_resnets)
    soft_label  = soft_label / len(list_best_resnets)
    
    # Set window soft label to 0 if the ensemble not detect an appliance in a window
    soft_label  = soft_label * np.round(prob_detect)

    # Sigmoid-Attention Module
    soft_label  = SigmoidAttention(data, soft_label, w_ma=5) # TODO: improve the hardcoding of Moving Average parameter

    if return_prob:
        return soft_label, prob_detect
    else:
        return soft_label


# def create_soft_label_old(expes_config,
#                           path_ensemble_clf,
#                           data,
#                           y=None):

#     assert len(data.shape)==2, f'Provide 2D numpy array for create_soft_label, got a {data.shape}D array.'

#     with open(f'{path_ensemble_clf}LogResNetsEnsemble.pkl', 'rb') as handle:
#         dict_results = pickle.load(handle)

#     list_best_resnets = dict_results['ListBestResNets']

#     soft_label = np.zeros_like(data)

#     for idx in range(len(data)):
#         current_win         = data[idx]
#         instance_soft_label = np.zeros_like(data[idx])

#         # Use true weak label if y provide
#         if y is not None:
#             if y[idx]<1:
#                 continue
#         else:
#             instance_prob_detect = 0

#             # Loop on BestResNets 
#             for resnet_name in list_best_resnets:
#                 resnet_inst = get_resnet_instance(expes_config['resnet_name'], dict_results[resnet_name]['kernel_size'])
#                 resnet_inst.to(expes_config['device'])

#                 if os.path.exists(f'{path_ensemble_clf}{resnet_name}.xz'):
#                     path_model = f'{path_ensemble_clf}{resnet_name}.xz'
#                     with lzma.open(path_model, 'rb') as file:
#                         decompressed_file = file.read()
#                     log = torch.load(io.BytesIO(decompressed_file), map_location=torch.device(expes_config['device']))
#                     del decompressed_file
#                 elif os.path.exists(f'{path_ensemble_clf}{resnet_name}.pt'):
#                     path_model = f'{path_ensemble_clf}{resnet_name}.pt'
#                     log = torch.load(path_model, map_location=torch.device(expes_config['device']))
#                 else:
#                     raise ValueError(f'Provide folders {path_ensemble_clf} does not contain {resnet_name} clf.')

#                 resnet_inst.load_state_dict(log['model_state_dict'])
#                 resnet_inst.eval()
#                 last_conv_layer, fc_layer_name = get_resnet_layers(expes_config['resnet_name'], resnet_inst)

#                 CAM_builder = CAM(model=resnet_inst, device=expes_config['device'], 
#                                   last_conv_layer=last_conv_layer, fc_layer_name=fc_layer_name)
#                 cam, y_pred, proba = CAM_builder.run(instance=current_win, returned_cam_for_label=1)
#                 instance_prob_detect += proba[1]

#                 # If y is not None, we use ground true 
#                 if y is not None:
#                     # Clip CAM and MaxNormalization (between 0 and 1)
#                     clip_cam = np.clip(cam, a_min=0, a_max=None)
#                     clip_cam = clip_cam / clip_cam.max()
                    
#                 # Or if app detected in this window
#                 elif y_pred>0:
#                     # Clip CAM and MaxNormalization (between 0 and 1)
#                     clip_cam = np.clip(cam, a_min=0, a_max=None)
#                     clip_cam = np.nan_to_num(clip_cam.astype(np.float32), nan=0.0, neginf=0.0, posinf=0.0)

#                     if clip_cam.max()>0:
#                         clip_cam = clip_cam / clip_cam.max()
#                     else:
#                         clip_cam = np.zeros_like(clip_cam)

#                     instance_soft_label = instance_soft_label + clip_cam.ravel()

#                 del resnet_inst
        
#         # Majority voting: if appliance not detected in current win, soft label set to 0
#         # if (y is not None) or ((instance_prob_detect / len(list_best_resnets)) >= 0.5):
#         if (instance_prob_detect / len(list_best_resnets)) >= 0.5:
#             instance_soft_label = instance_soft_label / len(list_best_resnets)
#         else: 
#             instance_soft_label = np.zeros_like(data[idx])

#         # # Thresholding if not using probabilities as labels
#         # if threshold is not None:
#         #   instance_soft_label = threshold_cam(instance_soft_label, threshold=0.5)

#         # If app detected, compute Sigmoid Attention with current win and thresshold the results
#         if (instance_soft_label > 0).any(): 
#             # Small moving average 
#             instance_soft_label = moving_average(instance_soft_label, w=5) # TODO: improve the hardcoding of Moving Average w parameter

#             # Sigmoid-Attention between input aggregate power and computed avg. CAM score
#             instance_soft_label = sigmoid(instance_soft_label * current_win)

#             # Thresholding to ensure to obtained binary labels
#             # instance_soft_label[instance_soft_label >= 0.5] = 1
#             # instance_soft_label[instance_soft_label < 0.5]  = 0 

#         # Thresholding to ensure to obtained binary labels
#         soft_label[idx, :] = np.round(instance_soft_label)

#     return soft_label


def evaluate_soft_label(expes_config,
                        path_ensemble_clf,
                        data_test,
                        f_metrics=NILMmetrics(),
                        clf_metrics=Classifmetrics()):
    
    # Create soft label
    soft_label, prob_detect = create_soft_label(expes_config,
                                                path_ensemble_clf,
                                                data_test[:, 0, 0, :],
                                                return_prob=True)

    # Get y_state and y_hat_state
    y_state     = data_test[:, 1, 1, :].ravel().astype(dtype=int)
    y_hat_state = soft_label.ravel().astype(dtype=int)

    if expes_config['appliance_mean_on_power'] is not None:
        # Get true appliance power
        data_test_rescale = expes_config['scaler'].inverse_transform(data_test)
        tmp_agg = data_test_rescale[:, 0, 0, :].ravel()
        y       = data_test_rescale[:, 1, 0, :].ravel()

        # Create soft label power value according to mean appliance power param
        y_hat = y_hat_state * expes_config['appliance_mean_on_power']
        # Ensure that app consumption doesn't exceed aggregate
        y_hat[y_hat>tmp_agg] = tmp_agg[y_hat>tmp_agg]
        del data_test_rescale
        del tmp_agg

        # Compute metric (NILM reg + classif)
        metric_softlabel = f_metrics(y, y_hat, y_state, y_hat_state)
    else: 
        # Compute metric (NILM classif)
        metric_softlabel = f_metrics(y=None, y_hat=None, y_state=y_state, y_hat_state=y_hat_state)

    _, y_clf_true = NILMdataset_to_Clf(data_test)
    metric_classif = clf_metrics(y=y_clf_true.ravel(), y_hat=prob_detect.ravel())

    return metric_softlabel, metric_classif