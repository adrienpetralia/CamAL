import os, sys
import numpy as np
import pickle

current_dir = os.getcwd()
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Helpers.expes_helpers import *
from Models.CamAL import CamAL


def ideal_camal_expes_possession(expes_config, path, win_weak,seed):

    np.random.seed(seed=seed)
    
    log_results = get_log_results_if_exist(path)

    # ========== Get IDEAL data ============== #
    data_builder = IDEAL_DataBuilder(data_path=os.path.dirname(os.getcwd())+'/Data/IDEAL/',
                                     mask_app=expes_config['appliance'],
                                     sampling_rate=expes_config['sampling_rate'],
                                     window_size=expes_config['window_size'],
                                     threshold_app_activation=expes_config['threshold_app_activation'],
                                     limit_ffill=60*6,
                                     window_stride=expes_config['window_size'])

    data, y, st_date  = data_builder.get_classif_dataset(w=win_weak)

    log_results['nb_instance']     = len(data)
    log_results['nb_houses']       = len(st_date.index.unique())
    log_results['n_instances_pos'] = np.sum(y)
    log_results['n_instances_neg'] = len(y) - np.sum(y)

    print(log_results)

    # Scale all data 
    data = data / 1000 # TODO: improve according of scaling params
    data = np.concatenate((data, np.expand_dims(y, axis=-1)), axis=1)

    #X_train, y_train, X_valid, y_valid, X_test, y_test = split_train_valid_test(data, test_size=0.3, valid_size=0.2, seed=seed)
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_train_valid_test_pdl(pd.DataFrame(data=data, index=st_date.index), test_size=0.3, valid_size=0.2, seed=seed)


    assert np.sum(y_train.ravel())>0, 'No positive labeled instance in training set'

    log_results['n_instances_train'] = len(X_train)
    log_results['n_labels_train']    = len(X_train)
    log_results['n_instances_train_pos'] = np.sum(y_train.ravel())
    log_results['n_instances_train_neg'] = len(y_train.ravel()) - np.sum(y_train.ravel())


    train_dataset = (X_train, y_train)
    valid_dataset = (X_valid, y_valid)
    test_dataset  = (X_test,  y_test)

    # ========= Train on Possession information ========= ""
    camal = CamAL(path, device=expes_config['device'])
    camal.train(train_dataset, valid_dataset, test_dataset)

    data_builder.limit_ffill = 60*2
    data, st_date = data_builder.get_nilm_dataset()

    assert len(st_date.index.unique())>1, 'No houses found with strong label for this appliance.'

    _, _, data_test, st_date_test = Split_train_test_pdl_NILMDataset(data, st_date, nb_house_test=8, seed=seed)

    print('Number of household for this appliance', len(st_date_test.index.unique()))
    print(list(st_date_test.index.unique()))

    list_pdl_test  = list(st_date_test.index.unique())
    log_results['id_test']  = list_pdl_test
        
    # Init NILM scaler
    scaler = NILMscaler(power_scaling_type=expes_config['power_scaling'], appliance_scaling_type=expes_config['appliance_scaling'])
    expes_config['scaler'] = scaler
    
    # Scale all data according to NILM scaler parameters
    data_test = scaler.fit_transform(data_test)

    nilm_perf, classif_perf = camal.test(data_test, 
                                         appliance_mean_on_power=expes_config['appliance_mean_power'], 
                                         scaler=expes_config['scaler'])
    
    print('CamALP localization score:', nilm_perf)
    print('CamALP classification score:', classif_perf)

    log_results.setdefault('CamALP', {})
    log_results['CamALP'][f'localization']   = nilm_perf
    log_results['CamALP'][f'classification'] = classif_perf

    save_log_results(log_results, path)

    return log_results


def ideal_crnn_expes_possession(expes_config, path, win_weak, seed):

    np.random.seed(seed=seed)
    
    log_results = get_log_results_if_exist(path)

    # ========== Get IDEAL data ============== #
    data_builder = IDEAL_DataBuilder(data_path=os.path.dirname(os.getcwd())+'/Data/IDEAL/',
                                     mask_app=expes_config['appliance'],
                                     sampling_rate=expes_config['sampling_rate'],
                                     window_size=expes_config['window_size'],
                                     threshold_app_activation=expes_config['threshold_app_activation'],
                                     limit_ffill=60*6,
                                     window_stride=expes_config['window_size'])

    data, y, st_date  = data_builder.get_classif_dataset(w=win_weak)

    log_results['nb_instance']     = len(data)
    log_results['nb_houses']       = len(st_date.index.unique())
    log_results['n_instances_pos'] = np.sum(y)
    log_results['n_instances_neg'] = len(y) - np.sum(y)

    print(log_results)

    # Scale all data 
    data = data / 1000 # TODO: improve according of scaling params
    data = np.concatenate((data, np.expand_dims(y, axis=-1)), axis=1)

    #X_train, y_train, X_valid, y_valid, X_test, y_test = split_train_valid_test(data, test_size=0.3, valid_size=0.2, seed=seed)
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_train_valid_test_pdl(pd.DataFrame(data=data, index=st_date.index), test_size=0.3, valid_size=0.2, seed=1)


    assert np.sum(y_train.ravel())>0, 'No positive labeled instance in training set'

    log_results['n_instances_train'] = len(X_train)
    log_results['n_labels_train']    = len(X_train)
    log_results['n_instances_train_pos'] = np.sum(y_train.ravel())
    log_results['n_instances_train_neg'] = len(y_train.ravel()) - np.sum(y_train.ravel())

    train_dataset = (X_train, y_train)
    valid_dataset = (X_valid, y_valid)
    test_dataset  = (X_test,  y_test)

    res_clf_crnn_possession, crnn_instance, dt_params = train_crnnweak_possession(expes_config,
                                                                                  path,
                                                                                  train_dataset,
                                                                                  valid_dataset,
                                                                                  test_dataset,
                                                                                  balance_class=True,
                                                                                  compress_and_save_space=False)


    # ========== Evaluation on NILM data ========== #
    data_builder.limit_ffill = 60*2
    data, st_date = data_builder.get_nilm_dataset()

    assert len(st_date.index.unique())>1, 'No houses found with strong label for this appliance.'

    _, _, data_test, st_date_test = Split_train_test_pdl_NILMDataset(data, st_date, nb_house_test=8, seed=seed)

    print('Number of household for this appliance', len(st_date_test.index.unique()))
    print(list(st_date_test.index.unique()))

    list_pdl_test  = list(st_date_test.index.unique())
    log_results['id_test']  = list_pdl_test
        
    # Init NILM scaler
    scaler = NILMscaler(power_scaling_type=expes_config['power_scaling'], appliance_scaling_type=expes_config['appliance_scaling'])
    expes_config['scaler'] = scaler
    
    # Scale all data according to NILM scaler parameters
    data_test = scaler.fit_transform(data_test)

    data_test,  st_date_test  = data_builder.get_nilm_dataset(list_pdl_test)
    data_test  = scaler.transform(data_test)

    _, _, test_dataset = get_pytorch_style_dataset(expes_config, 'CRNNWeak',
                                                    data_test, st_date_test,
                                                    data_test, st_date_test,
                                                    data_test, st_date_test)

    # ========== Evaluate soft label ============== #
    nilm_metrics = nilm_model_inference(expes_config, 
                                        path, 
                                        crnn_instance, dt_params,
                                        test_dataset,
                                        compress_and_save_space=True)

    log_results.setdefault('CRNNWeak', {})
    log_results['CRNNWeak'][f'localization']   = nilm_metrics
    log_results['CRNNWeak'][f'classification'] = res_clf_crnn_possession

    print('Score localization CRNN with possession', nilm_metrics)

    save_log_results(log_results, path)

    return log_results





def camal_expes(expes_config, path, seed):
    np.random.seed(seed=seed)
    log_results = get_log_results_if_exist(path)

    #for nb_data_for_training in [0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 6, 8, 10, 12, 'AllPossible']:
    nb_data_for_training = expes_config['nb_data_for_training']
    print(f'Perc data for training:{nb_data_for_training}')
    tmp_res = {}

    # ========== Get IDEAL data ============== #
    data_builder = IDEAL_DataBuilder(data_path=os.path.dirname(os.getcwd())+'/Data/IDEAL/',
                                    mask_app=expes_config['appliance'],
                                    sampling_rate=expes_config['sampling_rate'],
                                    window_size=expes_config['window_size'],
                                    threshold_app_activation=expes_config['threshold_app_activation'],
                                    limit_ffill=60*2,
                                    window_stride=expes_config['window_size'])

    data, st_date = data_builder.get_nilm_dataset()

    assert len(st_date.index.unique())>1, f'not enough houses for strong expes with this appliances, found {len(st_date.index.unique())}, 2 minimum are needed (train and test)'

    print('Houses found:', list(st_date.index.unique()))

    data_train, st_date_train, data_test, st_date_test   = Split_train_test_pdl_NILMDataset(data, st_date, nb_house_test=6, seed=seed)
    data_train, st_date_train, data_valid, st_date_valid = Split_train_test_pdl_NILMDataset(data_train, st_date_train, nb_house_test=2, seed=seed)
    
    print(f'NB data for training:{nb_data_for_training}')
    tmp_res = {}

    if nb_data_for_training!='AllPossible':
        if nb_data_for_training<=1:
            _, _, data_train, st_date_train   = Split_train_test_pdl_NILMDataset(data_train, st_date_train, nb_house_test=1, seed=seed)

            if nb_data_for_training<1:
                _, _, data_train, st_date_train = Split_train_test_NILMDataset(data_train, st_date_train, perc_house_test=nb_data_for_training, seed=seed)
                
        else:
            if nb_data_for_training>len(st_date_train.index.unique()):
                raise ValueError(f'nb_data_for_training>len(st_date_train.index.unique()): {nb_data_for_training}>{len(st_date_train.index.unique())}')
                #log_results[f'{nb_data_for_training}DataForTrain'] = -1
                #continue

            elif nb_data_for_training<len(st_date_train.index.unique()):
                _, _, data_train, st_date_train   = Split_train_test_pdl_NILMDataset(data_train, st_date_train, nb_house_test=nb_data_for_training, seed=seed)

            else:
                print('Same as AllPossible case')
                #continue


    list_pdl_train = st_date_train.index.unique()
    list_pdl_valid = st_date_valid.index.unique()
    list_pdl_test  = st_date_test.index.unique()

    tmp_res['id_train'] = list(list_pdl_train)
    tmp_res['id_valid'] = list(list_pdl_valid)
    tmp_res['id_test']  = list(list_pdl_test)

    tmp_res['n_labels_train'] = len(data_train)
    tmp_res['nb_houses']      = len(list_pdl_train)

    # Init NILM scaler and scale data
    scaler = NILMscaler(power_scaling_type=expes_config['power_scaling'], appliance_scaling_type=expes_config['appliance_scaling'])
    expes_config['scaler'] = scaler
    
    data_train = scaler.fit_transform(data_train)
    data_valid = scaler.transform(data_valid)
    data_test  = scaler.transform(data_test)

    tmp_res['n_labels_train'] = len(data_train)
    tmp_res['nb_houses']      = len(st_date_train.index.unique())

    # ========== Create train and test dataset for clf training ============== #
    train_clf_dataset = NILMdataset_to_Clf(data_train, st_date=st_date_train)
    test_clf_dataset  = NILMdataset_to_Clf(data_valid, st_date=st_date_valid)

    try:
        X_train, y_train, X_valid, y_valid = split_train_valid_test(train_clf_dataset, test_size=0.2)
    except:
        raise ValueError(f'Not enough sample to train with Perc data for training:{nb_data_for_training}')
        #continue

    tmp_res['n_instances_train'] = len(X_train)
    tmp_res['n_labels_train']    = len(X_train)
    tmp_res['n_instances_train_pos'] = np.sum(y_train.ravel())
    tmp_res['n_instances_train_neg'] = len(y_train.ravel()) - np.sum(y_train.ravel())

    if (np.sum(y_train.ravel())==0 or (len(y_train.ravel()) - np.sum(y_train.ravel()))==0):
        raise ValueError(f'Not enough sample to train with Perc data for training with perc {nb_data_for_training}')
        #continue

    train_dataset = (X_train, y_train)
    valid_dataset = (X_valid, y_valid)
    test_dataset  = (test_clf_dataset.iloc[:,:-1], test_clf_dataset.iloc[:,-1:])

    camal = CamAL(path, device=expes_config['device'])
    camal.train(train_dataset, valid_dataset, test_dataset)

    data_builder.window_size = expes_config['window_size_test']
    data_test,  _  = data_builder.get_nilm_dataset(tmp_res['id_test'])
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

    #for nb_data_for_training in [0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 6, 8, 10, 12, 'AllPossible']:
    nb_data_for_training = expes_config['nb_data_for_training']
    print(f'Perc data for training:{nb_data_for_training}')
    tmp_res = {}

    # ========== Get IDEAL data ============== #
    data_builder = IDEAL_DataBuilder(data_path=os.path.dirname(os.getcwd())+'/Data/IDEAL/',
                                    mask_app=expes_config['app'],
                                    sampling_rate=expes_config['sampling_rate'],
                                    window_size=expes_config['window_size'],
                                    threshold_app_activation=expes_config['threshold_app_activation'],
                                    limit_ffill=60*2,
                                    window_stride=expes_config['window_size'])

    data, st_date = data_builder.get_nilm_dataset()

    assert len(st_date.index.unique())>1, f'not enough houses for strong expes with this appliances, found {len(st_date.index.unique())}, 2 minimum are needed (train and test)'

    print('Houses found:', list(st_date.index.unique()))

    data_train, st_date_train, data_test, st_date_test   = Split_train_test_pdl_NILMDataset(data, st_date, nb_house_test=6, seed=seed)
    data_train, st_date_train, data_valid, st_date_valid = Split_train_test_pdl_NILMDataset(data_train, st_date_train, nb_house_test=2, seed=seed)

    if nb_data_for_training!='AllPossible':
        if nb_data_for_training<=1:
            _, _, data_train, st_date_train   = Split_train_test_pdl_NILMDataset(data_train, st_date_train, nb_house_test=1, seed=seed)

            if nb_data_for_training<1:
                _, _, data_train, st_date_train = Split_train_test_NILMDataset(data_train, st_date_train, perc_house_test=nb_data_for_training, seed=seed)
                
        else:
            if nb_data_for_training>len(st_date_train.index.unique()):
                raise ValueError(f'nb_data_for_training>len(st_date_train.index.unique()): {nb_data_for_training}>{len(st_date_train.index.unique())}')
                #log_results[f'{nb_data_for_training}DataForTrain'] = -1
                #continue

            elif nb_data_for_training<len(st_date_train.index.unique()):
                _, _, data_train, st_date_train   = Split_train_test_pdl_NILMDataset(data_train, st_date_train, nb_house_test=nb_data_for_training, seed=seed)

            else:
                warnings.warn('Same as AllPossible case')


    list_pdl_train = st_date_train.index.unique()
    list_pdl_valid = st_date_valid.index.unique()
    list_pdl_test  = st_date_test.index.unique()

    tmp_res['id_train'] = list(list_pdl_train)
    tmp_res['id_valid'] = list(list_pdl_valid)
    tmp_res['id_test']  = list(list_pdl_test)

    tmp_res['n_labels_train'] = len(data_train) * expes_config['window_size']
    tmp_res['nb_houses']      = len(list_pdl_train)

    # Init NILM scaler and scale data
    scaler = NILMscaler(power_scaling_type=expes_config['power_scaling'], appliance_scaling_type=expes_config['appliance_scaling'])
    expes_config['scaler'] = scaler
    
    data_train = scaler.fit_transform(data_train)
    data_valid = scaler.transform(data_valid)
    data_test  = scaler.transform(data_test)

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
                data_test,  st_date_test   = data_builder.get_nilm_dataset(list(list_pdl_test))
                data_builder.window_stride = tmp_window_stride
                data_test  = scaler.transform(data_test)

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

    expes                = str(sys.argv[1]) # CamALExpes or NILMExpes or PossessionExpes
    nb_data_for_training = str(sys.argv[2]) # AllPossible by default
    appliance            = str(sys.argv[3])
    seed                 = int(sys.argv[4])

    if nb_data_for_training!='AllPossible':
        try:
            nb_data_for_training = float(nb_data_for_training)
        except:
            raise ValueError(f"Please provide a percentage, number of household or 'AllPossible' as nb_data_for_training, received:{nb_data_for_training}")

    root  = os.path.dirname(os.getcwd())

    path_results = root + f'/Results/IDEAL/'
    _ = create_dir(path_results)

    print('Launch experiments using IDEAL dataset')
    print('Nb data for training:', nb_data_for_training)
    print('Appliance:', appliance)
    print('Seed:', seed)

    dataset_config = load_config(root+'/Configs/config_ideal_data.yaml')
    expes_config   = load_config(root+'/Configs/config_expes.yaml')
    print(expes_config)
    
    expes_config['nb_data_for_training'] = nb_data_for_training if nb_data_for_training < 1 else int(nb_data_for_training)
    expes_config['appliance'] = appliance
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

    elif expes=='PossessionExpes':
        print('Possession Expes')
        path_results = create_dir(f'{path_results}Possession/')
        path_results = create_dir(f'{path_results}{appliance}_Seed{seed}/')
        log_results_seed = {}
        for window_size_weak_possession in [1440, 2880, 5760, 10080, 20160, 30240, 40320, 50400]:
            path_results_win = create_dir(f'{window_size_weak_possession}/')
            log_results_seed[f'Possession_winweak{window_size_weak_possession}'] = strong_nilm_expes(expes_config, path=path_results_win, win_weak=window_size_weak_possession, seed=seed)
            log_results_seed[f'Possession_winweak{window_size_weak_possession}'] = ideal_crnn_expes_possession(expes_config, path=path_results_win, win_weak=window_size_weak_possession, seed=seed)

    print('Results summary')
    print(log_results_seed)

    name_file = path_results + 'LogResults.pkl'
    with open(name_file, 'wb') as f:
        pickle.dump(log_results_seed, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print('Results save at:')
    print(name_file)