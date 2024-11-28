import os, sys
import numpy as np
import pickle

current_dir = os.getcwd()
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Helpers.expes_helpers import *
from Models.CamAL import CamAL


def get_houses_indices(expes_config, seed):
    np.random.seed(seed=seed)

    if expes_config['fixed_houses_test']:
        id_train = expes_config['houses_id']['train_houses_ind']
        id_test  = expes_config['houses_id']['test_houses_ind']
        id_train = list(np.random.permutation(np.array(id_train))[:10])
        id_train, id_valid = id_train[:8], id_train[8:]
    else:
        id_train  = expes_config['houses_id']['train_houses_ind'] + expes_config['houses_id']['test_houses_ind']
        id_train = list(np.random.permutation(np.array(id_train))[:12])
        id_train, id_valid, id_test = id_train[:8], id_train[8:10], id_train[10:]

    return id_train, id_valid, id_test


def camal_expes(expes_config, path, seed):

    np.random.seed(seed=seed)
    
    log_results = get_log_results_if_exist(path)

    id_train, id_valid, id_test = get_houses_indices(expes_config, seed)


    log_results['id_train'] = id_train
    log_results['id_valid'] = id_valid
    log_results['id_test']  = id_test

    # ========== Get REFIT data ============== #

    # Get data train and valid
    data_builder = REFIT_DataBuilder(data_path=os.path.dirname(os.getcwd())+'/Data/REFIT/RAW_DATA_CLEAN/',
                                     mask_app=expes_config['appliance'],
                                     sampling_rate=expes_config['sampling_rate'],
                                     window_size=expes_config['window_size'],
                                     window_stride=None,
                                     use_status_from_kelly_paper=expes_config['use_kelly_status'])

    data_train_all, st_date_train_all = data_builder.get_nilm_dataset(id_train)
    data_valid, st_date_valid         = data_builder.get_nilm_dataset(id_valid)

    # Init NILM scaler
    scaler = NILMscaler(power_scaling_type=expes_config['power_scaling'], appliance_scaling_type=expes_config['appliance_scaling'])
    expes_config['scaler'] = scaler
    
    # Scale all data according to NILM scaler parameters
    data_train_all = scaler.fit_transform(data_train_all)
    data_valid = scaler.transform(data_valid)

    #for nb_data_for_training in [0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5, 6, 7, 8]:
    nb_data_for_training = expes_config['nb_data_for_training']
    
    tmp_res = {}

    if nb_data_for_training!=8:
        if nb_data_for_training<=1:
            _, _, data_train, st_date_train = Split_train_test_pdl_NILMDataset(data_train_all, st_date_train_all, nb_house_test=1, seed=seed)

            if nb_data_for_training<1:
                _, _, data_train, st_date_train = Split_train_test_NILMDataset(data_train, st_date_train, perc_house_test=nb_data_for_training, seed=seed)
                
        else:
            _, _, data_train, st_date_train   = Split_train_test_pdl_NILMDataset(data_train_all, st_date_train_all, nb_house_test=nb_data_for_training, seed=seed)
    else:
        data_train    = data_train_all
        st_date_train = st_date_train_all

    tmp_res['n_labels_train'] = len(data_train)
    tmp_res['nb_houses']      = len(st_date_train.index.unique())

    # ========== Create train and test dataset for clf training ============== #
    train_clf_dataset = NILMdataset_to_Clf(data_train, st_date=st_date_train)
    test_clf_dataset  = NILMdataset_to_Clf(data_valid, st_date=st_date_valid)

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
        print(f'Not enough sample to train with Perc data for training with perc {nb_data_for_training}')
        #print("Let's continue...")

    train_dataset = (X_train, y_train)
    valid_dataset = (X_valid, y_valid)
    test_dataset  = (test_clf_dataset.iloc[:,:-1], test_clf_dataset.iloc[:,-1:])

    camal = CamAL(path)
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

    # ========== Data Builder ============== #
    data_builder = REFIT_DataBuilder(data_path=os.path.dirname(os.getcwd())+'/Data/REFIT/RAW_DATA_CLEAN/',
                                     mask_app=expes_config['appliance'],
                                     sampling_rate=expes_config['sampling_rate'],
                                     window_size=expes_config['window_size'],
                                     window_stride=None,
                                     use_status_from_kelly_paper=expes_config['use_kelly_status'])

    # Get data test
    data_train_all, st_date_train_all = data_builder.get_nilm_dataset(id_train)
    data_test,  st_date_test  = data_builder.get_nilm_dataset(id_test)
    data_valid, st_date_valid = data_builder.get_nilm_dataset(id_valid)

    # Init NILM scaler and scale data
    scaler = NILMscaler(power_scaling_type=expes_config['power_scaling'], appliance_scaling_type=expes_config['appliance_scaling'])
    expes_config['scaler'] = scaler
    
    data_train_all = scaler.fit_transform(data_train_all)
    data_valid = scaler.transform(data_valid)
    data_test  = scaler.transform(data_test)

    #for nb_data_for_training in [0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5, 6, 7, 8]:
    nb_data_for_training = expes_config['nb_data_for_training']

    tmp_res = {}

    if nb_data_for_training!=8:
        if nb_data_for_training<=1:
            _, _, data_train, st_date_train = Split_train_test_pdl_NILMDataset(data_train_all, st_date_train_all, nb_house_test=1, seed=seed)

            if nb_data_for_training<1:
                _, _, data_train, st_date_train = Split_train_test_NILMDataset(data_train, st_date_train, perc_house_test=nb_data_for_training, seed=seed)
                
        else:
            _, _, data_train, st_date_train   = Split_train_test_pdl_NILMDataset(data_train_all, st_date_train_all, nb_house_test=nb_data_for_training, seed=seed)
    else:
        data_train    = data_train_all
        st_date_train = st_date_train_all

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
    nb_data_for_training = float(sys.argv[2]) # 8 by default
    appliance            = str(sys.argv[3])
    seed                 = int(sys.argv[4])

    root  = os.path.dirname(os.getcwd())

    path_results = root + f'/Results/REFIT/'
    _ = create_dir(path_results)

    print('Launch experiments using REFIT dataset')
    print('Nb data for training:', nb_data_for_training)
    print('Appliance:', appliance)
    print('Seed:', seed)

    refit_data   = load_config(root+'/Configs/config_refit_data.yaml')
    expes_config = load_config(root+'/Configs/config_expes.yaml')
    print(expes_config)
    
    expes_config['nb_data_for_training'] = nb_data_for_training if nb_data_for_training < 1 else int(nb_data_for_training)
    expes_config['appliance'] = appliance
    expes_config['appliance_mean_on_power'] = refit_data[appliance]['appliance_mean_on_power']

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