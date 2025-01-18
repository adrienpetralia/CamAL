import os
import sys
import pickle

import numpy as np

current_dir = os.getcwd()
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.helpers.data_processing import UKDALE_DataBuilder
from src.helpers.data_processing import (
    split_train_valid_test,
    split_train_test_pdl_nilmdataset,
    nilmdataset_to_clfdataset
)
from src.helpers.expes_helpers import (
    save_log_results,
    get_log_results_if_exist,
    get_pytorch_style_dataset,
    get_nilm_instance,
    nilm_model_training
)
from src.helpers.other import (
    create_dir,
    load_config
)
from src.helpers.torch_dataset import NILMscaler
from src.models.camal.core import CamAL


def get_houses_indices(expes_config, seed):
    np.random.seed(seed=seed)

    if expes_config['fixed_houses_test']:
        if expes_config['appliance']=='dishwasher':
            id_train = [1, 3, 4]
            id_valid = [5]
            id_test  = [2]
        elif expes_config['appliance']=='microwave':
            id_train = [1, 3]
            id_valid = [5]
            id_test  = [2]
        elif expes_config['appliance']=='kettle':
            id_train = [1, 3]
            id_valid = [5]
            id_test  = [2]
        elif expes_config['appliance']=='washing_machine':
            id_train = [1, 3]
            id_valid = [5]
            id_test  = [2]
    else:
        if expes_config['appliance']=='dishwasher':
            id_train = [1, 3, 4]
        elif expes_config['appliance']=='microwave':
            id_train = [1, 3]
        elif expes_config['appliance']=='kettle':
            id_train = [1, 3]
        elif expes_config['appliance']=='washing_machine':
            id_train = [1, 3]
        
        permute_ind_valid_test = list(np.random.permutation(np.array([2, 5])))
        id_valid, id_test = permute_ind_valid_test[:1], permute_ind_valid_test[1:]

    return id_train, id_valid, id_test


def camal_expes(expes_config, path, seed):

    np.random.seed(seed=seed)
    
    log_results = get_log_results_if_exist(path)

    id_train, id_valid, id_test = get_houses_indices(expes_config, seed)

    # id_train = id_train[:expes_config['n_houses_train']]

    log_results['id_train'] = id_train
    log_results['id_valid'] = id_valid
    log_results['id_test']  = id_test

    # ========== Get UKDALE data ============== #
    data_builder = UKDALE_DataBuilder(data_path=os.path.dirname(os.getcwd())+'/Data/UKDALE/',
                                      mask_app=expes_config['appliance'],
                                      sampling_rate=expes_config['sampling_rate'],
                                      window_size=expes_config['window_size'],
                                      window_stride=None,
                                      use_status_from_kelly_paper=expes_config['use_kelly_status'])

    data_train_all, st_date_train_all = data_builder.get_nilm_dataset(id_train)
    data_valid, st_date_valid = data_builder.get_nilm_dataset(id_valid)

    # Init NILM scaler
    scaler = NILMscaler(power_scaling_type=expes_config['power_scaling'], appliance_scaling_type=expes_config['appliance_scaling'])
    expes_config['scaler'] = scaler
    
    # Scale all data according to NILM scaler parameters
    data_train_all = scaler.fit_transform(data_train_all)
    data_valid = scaler.transform(data_valid)

    #for nb_data_for_training in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    nb_data_for_training = expes_config['nb_data_for_training']
    print(f'Perc data for training:{nb_data_for_training}')
    tmp_res = {}

    if nb_data_for_training!=1:
        _, _, data_train, st_date_train = split_train_test_pdl_nilmdataset(data_train_all, st_date_train_all, perc_house_test=nb_data_for_training, seed=seed)
    else:
        data_train, st_date_train = data_train_all, st_date_train_all

    tmp_res['n_labels_train'] = len(data_train)
    tmp_res['nb_houses']      = len(st_date_train.index.unique())

    # ========== Create train and test dataset for clf training ============== #
    train_clf_dataset = nilmdataset_to_clfdataset(data_train, st_date=st_date_train)
    test_clf_dataset  = nilmdataset_to_clfdataset(data_valid, st_date=st_date_valid)

    try:
        X_train, y_train, X_valid, y_valid = split_train_valid_test(train_clf_dataset, test_size=0.2)
    except:
        raise ValueError(f'Not enough sample to train with Perc data for training: {nb_data_for_training}')
        #continue

    tmp_res['n_instances_train'] = len(X_train)
    tmp_res['n_labels_train']    = len(X_train)
    tmp_res['n_instances_train_pos'] = np.sum(y_train.ravel())
    tmp_res['n_instances_train_neg'] = len(y_train.ravel()) - np.sum(y_train.ravel())

    if (np.sum(y_train.ravel())==0 or (len(y_train.ravel()) - np.sum(y_train.ravel()))==0):
        raise ValueError(f'Not enough sample to train with Perc data for training: {nb_data_for_training}')
        #continue

    train_dataset = (X_train, y_train)
    valid_dataset = (X_valid, y_valid)
    test_dataset  = (test_clf_dataset.iloc[:,:-1], test_clf_dataset.iloc[:,-1:])

    camal = CamAL(path, device=expes_config['device'])
    camal.train(train_dataset, valid_dataset, test_dataset)

    data_builder.window_size = expes_config['window_size_test']
    data_test,  _  = data_builder.get_nilm_dataset(id_test)
    data_test  = scaler.transform(data_test)

    nilm_perf, classif_perf = camal.test(data_test, 
                                         appliance_mean_on_power=expes_config['appliance_mean_power'], 
                                         scaler=expes_config['scaler'])
    
    print('CamAL localization score:', nilm_perf)
    print('CamAL classification score:', classif_perf)

    tmp_res['CamAL'] = {f'localization': nilm_perf, f'classif': classif_perf}
    log_results[f'{nb_data_for_training}DataForTrain'] = tmp_res

    save_log_results(log_results, path)

    return log_results


def strong_nilm_expes(expes_config, path, seed):

    np.random.seed(seed=seed)

    log_results = get_log_results_if_exist(path)

    id_train, id_valid, id_test = get_houses_indices(expes_config, seed)

    log_results['id_train'] = id_train
    log_results['id_valid'] = id_valid
    log_results['id_test']  = id_test

    # ========== Get UKDALE data ============== #
    data_builder = UKDALE_DataBuilder(data_path=os.path.dirname(os.getcwd())+'/Data/UKDALE/',
                                      mask_app=expes_config['appliance'],
                                      sampling_rate=expes_config['sampling_rate'],
                                      window_size=expes_config['window_size'],
                                      window_stride=None,
                                      use_status_from_kelly_paper=expes_config['use_kelly_status'])

    # Get data test
    data_train_all, st_date_train_all = data_builder.get_nilm_dataset(id_train)
    data_test,  st_date_test          = data_builder.get_nilm_dataset(id_test)
    data_valid, st_date_valid         = data_builder.get_nilm_dataset(id_valid)

    # Init NILM scaler and scale data
    scaler = NILMscaler(power_scaling_type=expes_config['power_scaling'], appliance_scaling_type=expes_config['appliance_scaling'])
    expes_config['scaler'] = scaler
    
    data_train_all = scaler.fit_transform(data_train_all)
    data_valid = scaler.transform(data_valid)
    data_test  = scaler.transform(data_test)

    #for nb_data_for_training in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    nb_data_for_training = expes_config['nb_data_for_training']
    print(f'Perc data for training:{nb_data_for_training}')
    tmp_res = {}

    if nb_data_for_training!=1:
        _, _, data_train, st_date_train = split_train_test_pdl_nilmdataset(data_train_all, st_date_train_all, perc_house_test=nb_data_for_training, seed=seed)
    else:
        data_train, st_date_train = data_train_all, st_date_train_all

    tmp_res['n_labels_train'] = len(data_train) * expes_config['window_size']
    tmp_res['nb_houses']      = len(st_date_train.index.unique())

    # ========== Training NILM methods ============== #
    for model_name in expes_config['List_Model_Sota_Expes']:
        print(f'{model_name}')
        
        if model_name=='TPNILM' or model_name=='TransNILM':
            if expes_config['window_size']!=510:
                print(f'{model_name} only compatible with window_size=510')
                continue
            else:
                tmp_data_test = np.copy(data_test)
                tmp_df_st_date_test = st_date_test.copy()
                tmp_window_stride = data_builder.window_stride
                data_builder.window_stride = 480
                data_test,  st_date_test   = data_builder.get_nilm_dataset(id_test)
                data_test  = scaler.transform(data_test)
                data_builder.window_stride = tmp_window_stride

        train_dataset, valid_dataset, test_dataset = get_pytorch_style_dataset(expes_config,
                                                                                model_name,
                                                                                data_train, st_date_train,
                                                                                data_valid, st_date_valid,
                                                                                data_test,  st_date_test)

        # Get NILM instance
        nilm_model, dt_params = get_nilm_instance(model_name, expes_config['window_size'])
        
        # Train NILM models
        nilm_metrics = nilm_model_training(expes_config, 
                                           path+model_name, 
                                           nilm_model, dt_params,
                                           train_dataset, 
                                           valid_dataset, 
                                           test_dataset)
        del nilm_model

        if model_name=='TPNILM' or model_name=='TransNILM':
            data_test    = tmp_data_test
            st_date_test = tmp_df_st_date_test

        tmp_res[model_name] = nilm_metrics
            
        log_results[f'{nb_data_for_training}DataForTrain'] = tmp_res
        print(tmp_res)


    save_log_results(log_results, path)

    return log_results



if __name__ == "__main__":
    
    expes                = str(sys.argv[1]) # CamALExpes or NILMExpes
    nb_data_for_training = float(sys.argv[2]) # 1 by default
    appliance            = str(sys.argv[3])
    seed                 = int(sys.argv[4])

    root  = os.path.dirname(os.getcwd())

    path_results = root + f'/Results/UKDALE/'
    _ = create_dir(path_results)

    print('Launch experiments using UKDALE dataset')
    print('Nb data for training:', nb_data_for_training)
    print('Appliance:', appliance)
    print('Seed:', seed)

    dataset_config = load_config(root+'/Configs/config_ukdale_data.yaml')
    expes_config   = load_config(root+'/Configs/config_expes.yaml')
    print(expes_config)
    
    expes_config['nb_data_for_training'] = nb_data_for_training if nb_data_for_training < 1 else int(nb_data_for_training)
    expes_config['appliance'] = dataset_config[appliance]['app_ukdale_dataset_name']
    expes_config['appliance_mean_on_power'] = dataset_config[appliance]['appliance_mean_on_power']

    if expes=='CamALExpes':
        print('CamAL Expes')
        path_results = create_dir(f'{path_results}CamAL/')
        path_results = create_dir(f'{path_results}{appliance}_Seed{seed}/')
        log_results_seed = camal_expes(expes_config, path=path_results, seed=seed)

    elif expes=='NILMExpes':
        print('SotA Expes')
        path_results = create_dir(f'{path_results}SotA/')
        path_results = create_dir(f'{path_results}{appliance}_Seed{seed}/')
        log_results_seed = strong_nilm_expes(expes_config, path=path_results, seed=seed)

    print('Results summary')
    print(log_results_seed)

    name_file = path_results + 'LogResults.pkl'
    with open(name_file, 'wb') as f:
        pickle.dump(log_results_seed, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print('Results save at:')
    print(name_file)