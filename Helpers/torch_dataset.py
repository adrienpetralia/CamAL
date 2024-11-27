import numpy as np
import pandas as pd
import torch
    
from sklearn.preprocessing import StandardScaler
    
class NILMscaler():
    """
    Scale NILM dataset
    
    Nilm data need to be 4D Numpy array following the convention: 
    [N_sequences, Card[Agg_Power, 1_appliance,.., M_appliance], 2-dim:0:Power/1:States, Window Length]
    
    Follow sklearn convention (fit/transform/fit_transform) and is callable
    """
    def __init__(self, power_scaling_type='StandardScaling', appliance_scaling_type=None, 
                 scale_temperature=False, temp_min=0, temp_max=35, newRange_temp=(-1, 1)):
        
        if not isinstance(power_scaling_type, int):
            assert power_scaling_type in ['StandardScaling', 'MinMax', 'MeanScaling', 'MeanMaxScaling', 'MaxScaling']
        
        if not isinstance(appliance_scaling_type, int):
            assert appliance_scaling_type in ['StandardScaling', 'MinMax', 'MeanScaling', 'MeanMaxScaling', 'MaxScaling', 'SameAsPower']
        
        self.power_scaling_type = power_scaling_type
        self.appliance_scaling_type = appliance_scaling_type

        self.scale_temperature = scale_temperature
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.newRange_temp = newRange_temp

        self.n_appliance = 0
        self.is_fitted = False
        
        self.power_stat1 = 0
        self.power_stat2  = 0
        self.appliance_stat1 = []
        self.appliance_stat2 = []
        
    def __call__(self, data_train, data_test):
        scale_data_train = self.fit_transform(data_train.copy())
        scale_data_test  = self.transform(data_test.copy())
        
        return scale_data_train, scale_data_test
        
    def fit(self, data):
        if isinstance(self.power_scaling_type, int):
            self.power_stat1 = 0
            self.power_stat2 = self.power_scaling_type
        elif self.power_scaling_type=='StandardScaling':
            self.power_stat1 = data[:, 0, 0, :].mean()
            self.power_stat2 = data[:, 0, 0, :].std()
        elif self.power_scaling_type=='MinMaxScaling':
            self.power_stat1 = data[:, 0, 0, :].min()
            self.power_stat2 = data[:, 0, 0, :].max()
        elif self.power_scaling_type=='MeanScaling':
            self.power_stat1 = 0
            self.power_stat2 = data[:, 0, 0, :].mean()
        elif self.power_scaling_type=='MeanMaxScaling' or self.power_scaling_type=='MaxScaling':
            self.power_stat1 = data[:, 0, 0, :].mean() if self.power_scaling_type=='MeanMaxScaling' else 0
            self.power_stat2 = data[:, 0, 0, :].max()
            
        if self.appliance_scaling_type is not None:
            self.n_appliance = data.shape[1] - 1

            if isinstance(self.appliance_scaling_type, int):
                self.appliance_stat1.append(0)
                self.appliance_stat2.append(self.appliance_scaling_type)
            elif self.appliance_scaling_type=='StandardScaling':
                self.appliance_stat1.append(data[:, 1, 0, :].mean())
                self.appliance_stat2.append(data[:, 1, 0, :].std())
            elif self.appliance_scaling_type=='MinMax':
                self.appliance_stat1.append(data[:, 1, 0, :].min())
                self.appliance_stat2.append(data[:, 1, 0, :].max())
            elif self.appliance_scaling_type=='MeanMaxScaling' or self.appliance_scaling_type=='MaxScaling':
                if self.appliance_scaling_type=='MeanMaxScaling':
                    self.appliance_stat1.append(data[:, 1, 0, :].mean())  
                else:
                    self.appliance_stat1.append(0)
                self.appliance_stat2.append(data[:, 1, 0, :].max())
            elif self.appliance_scaling_type=='SameAsPower':
                self.appliance_stat1.append(self.power_stat1)
                self.appliance_stat2.append(self.power_stat2)
                
        self.is_fitted = True

        return
    
    def transform(self, data):
        assert self.is_fitted, 'Not fitted yet.'
        
        if self.power_scaling_type=='MinMax':
            data[:, 0, 0, :] = (data[:, 0, 0, :] - self.power_stat1) / (self.power_stat2 - self.power_stat1)
        else:
            data[:, 0, 0, :] = (data[:, 0, 0, :] - self.power_stat1) / self.power_stat2
            
        if self.appliance_scaling_type is not None:
            for n_app in range(1, self.n_appliance+1):
                if self.appliance_scaling_type=='MinMax' or (self.appliance_scaling_type=='SameAsPower' and self.power_scaling_type=='MinMax'):
                    data[:, n_app, 0, :] = (data[:, n_app, 0, :] - self.appliance_stat1[n_app-1]) / (self.appliance_stat2[n_app-1] - self.appliance_stat1[n_app-1])
                else:
                    data[:, n_app, 0, :] = (data[:, n_app, 0, :] - self.appliance_stat1[n_app-1]) / self.appliance_stat2[n_app-1]

        if self.scale_temperature:
            data[:, 0, 1, :] = (self.newRange_temp[1] - self.newRange_temp[0]) * ((np.clip(data[:, 0, 1, :], a_min=self.temp_min, a_max=self.temp_max) - self.temp_min) / (self.temp_max - self.temp_min))  + self.newRange_temp[0]
                    
        return data
        
    def fit_transform(self, data):
        self.fit(data)
        data = self.transform(data)

        return data
    
    def inverse_transform(self, data):
        rescale_data = data.copy()
        assert len(rescale_data.shape)<5, 'Data containing too many dimensions (>5).'
        if len(rescale_data.shape)<4:
            flag=True
            assert len(data.shape)==3, 'At lest non batched data (3D array) to inverse transform.'
            rescale_data = np.expand_dims(rescale_data, axis=0)
        else:
            flag=False
            
        if self.power_scaling_type=='MinMax':
            rescale_data[:, 0, 0, :] = rescale_data[:, 0, 0, :] * (self.power_stat2 - self.power_stat1) + self.power_stat1
        else:
            rescale_data[:, 0, 0, :] = rescale_data[:, 0, 0, :] * self.power_stat2 + self.power_stat1
            
        if self.appliance_scaling_type is not None:
            for n_app in range(1, self.n_appliance+1):
                if self.appliance_scaling_type=='MinMax' or (self.appliance_scaling_type=='SameAsPower' and self.power_scaling_type=='MinMax'):
                    rescale_data[:, n_app, 0, :] = rescale_data[:, n_app, 0, :]  * (self.appliance_stat2[n_app-1] - self.appliance_stat1[n_app-1]) + self.appliance_stat1[n_app-1]
                else:
                    rescale_data[:, n_app, 0, :] = rescale_data[:, n_app, 0, :]  * self.appliance_stat2[n_app-1] + self.appliance_stat1[n_app-1]
        
        if flag:
            rescale_data = rescale_data[0]
            
        return rescale_data
    
    def inverse_transform_appliance(self, data):
        if torch.is_tensor(data):
            rescale_data = data.clone()
            if len(rescale_data.shape)==2:
                rescale_data = rescale_data.unsqueeze(0)
        else:
            rescale_data = data.copy()
            if len(rescale_data.shape)==2:
                rescale_data = np.expand_dims(rescale_data, axis=0)

        assert self.appliance_scaling_type is not None, 'Original data : no scaling apply on appliance channels!'
            
        for n_app in range(self.n_appliance):
            if self.appliance_scaling_type=='MinMax' or (self.appliance_scaling_type=='SameAsPower' and self.power_scaling_type=='MinMax'):
                rescale_data[:, n_app, :] = rescale_data[:, n_app, :]  * (self.appliance_stat2[n_app] - self.appliance_stat1[n_app]) + self.appliance_stat1[n_app]
            else:
                rescale_data[:, n_app, :] = rescale_data[:, n_app, :]  * self.appliance_stat2[n_app] + self.appliance_stat1[n_app]
            
        return rescale_data

    def inverse_transform_agg_power(self, data):
        if torch.is_tensor(data):
            rescale_data = data.clone()
            if len(rescale_data.shape)==2:
                rescale_data = rescale_data.unsqueeze(0)
        else:
            rescale_data = data.copy()
            if len(rescale_data.shape)==2:
                rescale_data = np.expand_dims(rescale_data, axis=0)
            
        if self.appliance_scaling_type=='MinMax' or (self.appliance_scaling_type=='SameAsPower' and self.power_scaling_type=='MinMax'):
            rescale_data[:, 0, :] = rescale_data[:, 0, :]  * (self.power_stat2 - self.power_stat1) + self.power_stat1
        else:
            rescale_data[:, 0, :] = rescale_data[:, 0, :]  * self.power_stat2 + self.power_stat1
            
        return rescale_data



class TSDataset(torch.utils.data.Dataset):
    """
    MAP-Style PyTorch Time series Dataset with possibility of scaling
    
    - X matrix of TS input, can be 2D or 3D, Dataframe instance or Numpy array instance.
    - Labels : y labels associated to time series for classification. Possible to be None.
    - scaler : provided type of scaler (sklearn StandardScaler, MinMaxScaler instance for example).
    - scale_dim : list of dimensions to be scaled in case of multivariate TS.
    """
    def __init__(self, X, labels=None, scaler=False, scale_dim=None):
        
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values
        if isinstance(labels, pd.core.frame.DataFrame):
            labels = labels.values
        
        if scaler:
            # ==== Multivariate case ==== #
            if len(X.shape)==3:
                self.scaler_list = []
                self.samples = X
                if scale_dim is None:                    
                    for i in range(X.shape[1]):
                        self.scaler_list.append(StandardScaler())
                        self.samples[:,i,:] = self.scaler_list[i].fit_transform(X[:,i,:].T).T.astype(np.float32)
                else:
                    for idsc, i in enumerate(scale_dim):
                        self.scaler_list.append(StandardScaler())
                        self.samples[:,i,:] = self.scaler_list[idsc].fit_transform(X[:,i,:].T).T.astype(np.float32)
                        
            # ==== Univariate case ==== #
            else:
                self.scaler_list = [StandardScaler()]
                self.samples = self.scaler_list[0].fit_transform(X.T).T.astype(np.float32)
        else:
            self.samples = X
            
        if len(self.samples.shape)==2:
            self.samples = np.expand_dims(self.samples, axis=1)
        
        if labels is not None:
            self.labels = labels.ravel()
            assert len(self.samples)==len(self.labels), f"Number of X sample {len(self.samples)} doesn't match number of y sample {len(self.labels)}."
        else:
            self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        if self.labels is None:
            return self.samples[idx]
        else:
            return self.samples[idx], self.labels[idx]


class TSDatasetScaling(torch.utils.data.Dataset):
    """
    MAP-Style PyTorch Time series Dataset
    
    Scaling computed on the fly
    """
    def __init__(self, X, labels=None, 
                 scale_data=False, inst_scaling=False,
                 st_date=None, mask_date='start_date', list_exo_variables=[], freq='30T', 
                 cosinbase=True, newRange=(-1, 1)):

        self.scale_data = scale_data
        self.inst_scaling = inst_scaling

        self.freq = freq
        self.list_exo_variables = list_exo_variables
        self.cosinbase = cosinbase
        self.newRange = newRange
        self.L = X.shape[-1]

        if st_date is not None:
            assert list_exo_variables is not None and len(list_exo_variables) > 0, "Please provide list of exo variable if st_date not None."
            assert self.freq is not None, "Variable freq not defined but st_date provided."

            self.st_date = st_date[mask_date].values.flatten()

            if self.cosinbase:
                self.n_var = 2*len(self.list_exo_variables)
            else:
                self.n_var = len(self.list_exo_variables)
        else:
            self.n_var = None
        
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values
        if isinstance(labels, pd.core.frame.DataFrame):
            labels = labels.values
        
        self.samples = X
            
        if len(self.samples.shape)==2:
            self.samples = np.expand_dims(self.samples, axis=1)

        if not inst_scaling:
            self.mean = np.squeeze(np.mean(self.samples, axis=(0, 2), keepdims=True), axis=0)
            self.std  = np.squeeze(np.std(self.samples, axis=(0, 2), keepdims=True) + 1e-9, axis=0)
        
        if labels is not None:
            self.labels = labels.ravel()
            assert len(self.samples)==len(self.labels), f"Number of X sample {len(self.samples)} doesn't match number of y sample {len(self.labels)}."
        else:
            self.labels = labels

    def _create_exogene(self, idx):
        
        np_extra = np.zeros((self.n_var, self.L)).astype(np.float32)
        tmp = pd.date_range(start=self.st_date[idx], periods=self.L, freq=self.freq)
        
        k = 0
        for exo_var in self.list_exo_variables:
            if exo_var=='month':
                if self.cosinbase:
                    np_extra[k, :]   = np.sin(2 * np.pi * tmp.month.values/12.0)
                    np_extra[k+1, :] = np.cos(2 * np.pi * tmp.month.values/12.0)
                    k+=2
                else:
                    np_extra[k, :]   = self._normalize(tmp.month.values, xmin=1, xmax=12, newRange=self.newRange)
                    k+=1
            elif exo_var=='dom':
                if self.cosinbase:
                    np_extra[k, :]   = np.sin(2 * np.pi * tmp.day.values/31.0)
                    np_extra[k+1, :] = np.cos(2 * np.pi * tmp.day.values/31.0)
                    k+=2
                else:
                    np_extra[k, :]   = self._normalize(tmp.month.values, xmin=1, xmax=12, newRange=self.newRange)
                    k+=1
            elif exo_var=='dow':
                if self.cosinbase:
                    np_extra[k, :]   = np.sin(2 * np.pi * tmp.dayofweek.values/7.0)
                    np_extra[k+1, :] = np.cos(2 * np.pi * tmp.dayofweek.values/7.0)
                    k+=2
                else:
                    np_extra[k, :]   = self._normalize(tmp.month.values, xmin=1, xmax=7, newRange=self.newRange)
                    k+=1
            elif exo_var=='hour':
                if self.cosinbase:
                    np_extra[k, :]   = np.sin(2 * np.pi * tmp.hour.values/24.0)
                    np_extra[k+1, :] = np.cos(2 * np.pi * tmp.hour.values/24.0)
                    k+=2
                else:
                    np_extra[k, :]   = self._normalize(tmp.month.values, xmin=0, xmax=24, newRange=self.newRange)
                    k+=1
            elif exo_var=='minute':
                if self.cosinbase:
                    np_extra[k, :]   = np.sin(2 * np.pi * tmp.minute.values/60.0)
                    np_extra[k+1, :] = np.cos(2 * np.pi * tmp.minute.values/60.0)
                    k+=2
                else:
                    np_extra[k, :]   = self.normalize(tmp.minute.values, xmin=0, xmax=60, newRange=self.newRange)
                    k+=1
            else:
                raise ValueError("Embedding unknown for these Data. Only 'month', 'dow', 'dom', 'hour', 'minute' supported, received {}"
                                    .format(exo_var))  

        return np_extra

    def _normalize(self, x, xmin, xmax, newRange): 
        if xmin is None:
            xmin = np.min(x)
        if xmax is None:
            xmax = np.max(x) 
            
        norm = (x - xmin)/(xmax - xmin) 
        if newRange == (0, 1):
            return norm 
        elif newRange != (0, 1):
            return norm * (newRange[1] - newRange[0]) + newRange[0] 

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        tmp_sample = self.samples[idx].copy()

        if self.scale_data:
            if self.inst_scaling:
                tmp_sample = (tmp_sample - np.mean(tmp_sample, axis=1, keepdims=True)) / (np.std(tmp_sample, axis=1, keepdims=True) + 1e-9)
            else:
                tmp_sample = (tmp_sample - self.mean) / self.std

        if self.n_var is not None:
            exo = self._create_exogene(idx)
            tmp_sample = np.concatenate((tmp_sample, exo), axis=0)

        if self.labels is None:
            return tmp_sample
        else:
            return tmp_sample.astype(np.float32), self.labels[idx]
               

# class NILMDataset(torch.utils.data.Dataset):
#     """
#     Pytorch dataset
    
#     - X, 4D Numpy array : N subsequences, M appliances, Power/Activation, Values
#     - st_date, pd.dataframe : Starting date of each subsequence
#     - scaler, Boolean : True if apply scaling
#     """
#     def __init__(self, X, 
#                  list_exo_variables=[], cam=None, use_temperature=False,
#                  pretraining=False,
#                  st_date=None, mask_date='start_date', freq=None, 
#                  cosinbase=True, newRange=(-1, 1),
#                  inst_scaling=False):
        
#         self.samples = X

#         self.pretraining = pretraining
#         self.use_temperature = use_temperature
#         self.inst_scaling = inst_scaling

#         self.mask_date = mask_date
#         self.freq = freq
#         self.cosinbase = cosinbase
#         self.newRange = newRange
#         self.L = X.shape[-1]
#         self.list_exo_variables = list_exo_variables

#         if len(list_exo_variables) > 0:
#             assert st_date is not None,   "list_exo_variables provided but st_date is None: please provide st_date information to compute exogene variable."
#             assert freq is not None, "Variable freq is None but list_exo_variables provided: please set freq according to data sampling rate."
#             assert mask_date is not None, "Variable mask_date is None but list_exo_variables provided: please choose freq according to data sampling rate."

#             self.st_date = st_date[mask_date].values.flatten()

#             if self.cosinbase:
#                 self.n_var = 2*len(self.list_exo_variables)
#             else:
#                 self.n_var = len(self.list_exo_variables)
#         else:
#             self.n_var = None
                
#         if cam is not None:
#             self.cam = cam
#             if len(self.cam.shape)==2:
#                 self.cam = np.expand_dims(self.cam, axis=1)
#         else:
#             self.cam = None  

#     def _create_exogene(self, idx):
        
#         np_extra = np.zeros((self.n_var, self.L)).astype(np.float32)
#         tmp = pd.date_range(start=self.st_date[idx], periods=self.L, freq=self.freq)
        
#         k = 0
#         for exo_var in self.list_exo_variables:
#             if exo_var=='month':
#                 if self.cosinbase:
#                     np_extra[k, :]   = np.sin(2 * np.pi * tmp.month.values/12.0)
#                     np_extra[k+1, :] = np.cos(2 * np.pi * tmp.month.values/12.0)
#                     k+=2
#                 else:
#                     np_extra[k, :]   = self._normalize(tmp.month.values, xmin=1, xmax=12, newRange=self.newRange)
#                     k+=1
#             elif exo_var=='dom':
#                 if self.cosinbase:
#                     np_extra[k, :]   = np.sin(2 * np.pi * tmp.day.values/31.0)
#                     np_extra[k+1, :] = np.cos(2 * np.pi * tmp.day.values/31.0)
#                     k+=2
#                 else:
#                     np_extra[k, :]   = self._normalize(tmp.day.values, xmin=1, xmax=31, newRange=self.newRange)
#                     k+=1
#             elif exo_var=='dow':
#                 if self.cosinbase:
#                     np_extra[k, :]   = np.sin(2 * np.pi * tmp.dayofweek.values/7.0)
#                     np_extra[k+1, :] = np.cos(2 * np.pi * tmp.dayofweek.values/7.0)
#                     k+=2
#                 else:
#                     np_extra[k, :]   = self._normalize(tmp.dayofweek.values, xmin=1, xmax=7, newRange=self.newRange)
#                     k+=1
#             elif exo_var=='hour':
#                 if self.cosinbase:
#                     np_extra[k, :]   = np.sin(2 * np.pi * tmp.hour.values/24.0)
#                     np_extra[k+1, :] = np.cos(2 * np.pi * tmp.hour.values/24.0)
#                     k+=2
#                 else:
#                     np_extra[k, :]   = self._normalize(tmp.hour.values, xmin=0, xmax=24, newRange=self.newRange)
#                     k+=1
#             elif exo_var=='minute':
#                 if self.cosinbase:
#                     np_extra[k, :]   = np.sin(2 * np.pi * tmp.minute.values/60.0)
#                     np_extra[k+1, :] = np.cos(2 * np.pi * tmp.minute.values/60.0)
#                     k+=2
#                 else:
#                     np_extra[k, :]   = self.normalize(tmp.minute.values, xmin=0, xmax=60, newRange=self.newRange)
#                     k+=1
#             else:
#                 raise ValueError("Embedding unknown for these Data. Only 'month', 'dow', 'dom', 'hour', 'minute' supported, received {}"
#                                     .format(exo_var))   

#         return np_extra
    
#     def normalize(self, x, xmin=None, xmax=None, newRange=(-1, 1)): 
#         if xmin is None:
#             xmin = np.min(x)
#         if xmax is None:
#             xmax = np.max(x) 
            
#         norm = (x - xmin)/(xmax - xmin) 
#         if newRange == (0, 1):
#             return norm 
#         elif newRange != (0, 1):
#             return norm * (newRange[1] - newRange[0]) + newRange[0] 
        
#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         """
#         Return Tuple as follow:

#             if self.pretraining:
#                 Aggregate/Temp/Encoding
#             else:
#                 Aggregate/Temp/Encoding, App. Power, App. Activation States
#         """
#         if self.use_temperature:
#             tmp_sample = self.samples[idx, 0, :2, :].copy()
#         else:
#             tmp_sample = self.samples[idx, 0, :1, :].copy()

#         if self.inst_scaling:
#             tmp_sample = (tmp_sample - np.mean(tmp_sample, axis=1, keepdims=True)) / (np.std(tmp_sample, axis=1, keepdims=True) + 1e-9)

#         if self.n_var is not None:
#             exo = self._create_exogene(idx)
#             tmp_sample = np.concatenate((tmp_sample, exo), axis=0)

#         if self.cam is not None:
#             tmp_sample = np.concatenate((tmp_sample, self.cam[idx, :, :]), axis=0)

#         if self.pretraining:
#             return tmp_sample
#         else:
#             return tmp_sample, self.samples[idx, 1:2, 0, :], self.samples[idx, 1:2, 1, :]

class NILMDataset(torch.utils.data.Dataset):
    """
    Pytorch dataset
    
    - X, 4D Numpy array : N subsequences, M appliances, Power/Activation, Values
    - st_date, pd.dataframe : Starting date of each subsequence
    - scaler, Boolean : True if apply scaling
    """
    def __init__(self, X, X_weak=None, 
                 list_exo_variables=[], 
                 return_output='strong', 
                 padding_output_seq=0, 
                 pretraining=False,
                 st_date=None, st_date_weak=None,
                 mask_date='start_date', freq=None, 
                 cosinbase=True, newRange=(-1, 1)):

        assert return_output in ['strong', 'weak', 'strong_weak']
        assert 2*padding_output_seq < X.shape[-1]

        self.n_instances_strong = len(X)
        self.n_instances_weak   = len(X_weak) if X_weak is not None else 0

        self.samples = np.concatenate((X, X_weak), axis=0) if X_weak is not None else X

        self.pretraining  = pretraining

        self.return_output      = return_output # Default strong, change for CRNN training
        self.padding_output_seq = int(padding_output_seq) # For TPNILM as input != output for this model

        self.mask_date = mask_date
        self.freq = freq
        self.cosinbase = cosinbase
        self.newRange = newRange
        self.L = X.shape[-1]
        self.list_exo_variables = list_exo_variables

        if len(list_exo_variables) > 0:
            assert st_date is not None,   "list_exo_variables provided but st_date is None: please provide st_date information to compute exogene variable."
            assert freq is not None, "Variable freq is None but list_exo_variables provided: please set freq according to data sampling rate."
            assert mask_date is not None, "Variable mask_date is None but list_exo_variables provided: please choose freq according to data sampling rate."

            if X_weak is not None:
                assert st_date_weak is not None, "Please provide st_date for weak labeled data"
                self.st_date = np.concatenate((st_date[mask_date].values.flatten(), st_date_weak[mask_date].values.flatten()), axis=0)
            else:
                self.st_date = st_date[mask_date].values.flatten()

            if self.cosinbase:
                self.n_var = 2*len(self.list_exo_variables)
            else:
                self.n_var = len(self.list_exo_variables)
        else:
            self.n_var = None

    def _create_exogene(self, idx):
        
        np_extra = np.zeros((self.n_var, self.L)).astype(np.float32)
        tmp = pd.date_range(start=self.st_date[idx], periods=self.L, freq=self.freq)
        
        k = 0
        for exo_var in self.list_exo_variables:
            if exo_var=='month':
                if self.cosinbase:
                    np_extra[k, :]   = np.sin(2 * np.pi * tmp.month.values/12.0)
                    np_extra[k+1, :] = np.cos(2 * np.pi * tmp.month.values/12.0)
                    k+=2
                else:
                    np_extra[k, :]   = self._normalize(tmp.month.values, xmin=1, xmax=12, newRange=self.newRange)
                    k+=1
            elif exo_var=='dom':
                if self.cosinbase:
                    np_extra[k, :]   = np.sin(2 * np.pi * tmp.day.values/31.0)
                    np_extra[k+1, :] = np.cos(2 * np.pi * tmp.day.values/31.0)
                    k+=2
                else:
                    np_extra[k, :]   = self._normalize(tmp.day.values, xmin=1, xmax=31, newRange=self.newRange)
                    k+=1
            elif exo_var=='dow':
                if self.cosinbase:
                    np_extra[k, :]   = np.sin(2 * np.pi * tmp.dayofweek.values/7.0)
                    np_extra[k+1, :] = np.cos(2 * np.pi * tmp.dayofweek.values/7.0)
                    k+=2
                else:
                    np_extra[k, :]   = self._normalize(tmp.dayofweek.values, xmin=1, xmax=7, newRange=self.newRange)
                    k+=1
            elif exo_var=='hour':
                if self.cosinbase:
                    np_extra[k, :]   = np.sin(2 * np.pi * tmp.hour.values/24.0)
                    np_extra[k+1, :] = np.cos(2 * np.pi * tmp.hour.values/24.0)
                    k+=2
                else:
                    np_extra[k, :]   = self._normalize(tmp.hour.values, xmin=0, xmax=24, newRange=self.newRange)
                    k+=1
            elif exo_var=='minute':
                if self.cosinbase:
                    np_extra[k, :]   = np.sin(2 * np.pi * tmp.minute.values/60.0)
                    np_extra[k+1, :] = np.cos(2 * np.pi * tmp.minute.values/60.0)
                    k+=2
                else:
                    np_extra[k, :]   = self.normalize(tmp.minute.values, xmin=0, xmax=60, newRange=self.newRange)
                    k+=1
            else:
                raise ValueError("Embedding unknown for these Data. Only 'month', 'dow', 'dom', 'hour', 'minute' supported, received {}".format(exo_var))   

        return np_extra
    
    def normalize(self, x, xmin=None, xmax=None, newRange=(-1, 1)): 
        if xmin is None:
            xmin = np.min(x)
        if xmax is None:
            xmax = np.max(x) 
            
        norm = (x - xmin)/(xmax - xmin) 
        if newRange == (0, 1):
            return norm 
        elif newRange != (0, 1):
            return norm * (newRange[1] - newRange[0]) + newRange[0] 
        
    def __len__(self):
        return self.n_instances_strong + self.n_instances_weak

    def __getitem__(self, idx):
        """
        Return Tuple as follow:

            if self.pretraining:
                Aggregate/Temp/Encoding
            else:
                Aggregate/Temp/Encoding, App. Power, App. Activation States
        """
        tmp_sample = self.samples[idx, 0, :1, :].copy()

        if self.n_var is not None:
            exo = self._create_exogene(idx)
            tmp_sample = np.concatenate((tmp_sample, exo), axis=0)

        if self.pretraining:
            return tmp_sample
        
        if self.return_output=='weak':
            label = int(self.samples[idx, 1, 1, :].any())

            return tmp_sample, label
                
        elif self.return_output=='strong_weak':
            label = int(self.samples[idx, 1, 1, :].any())

            if self.n_instances_weak > 0:
                if idx >= self.n_instances_strong:
                    flag_strong = 0
                    return tmp_sample, self.samples[idx, 1:2, 0, :], self.samples[idx, 1:2, 1, :], label, flag_strong
                else:
                    flag_strong = 1
                    return tmp_sample, self.samples[idx, 1:2, 0, :], self.samples[idx, 1:2, 1, :], label, flag_strong
            else:
                return tmp_sample, self.samples[idx, 1:2, 0, :], self.samples[idx, 1:2, 1, :], label    

        else:
            if self.padding_output_seq>0:
                return tmp_sample, self.samples[idx, 1:2, 0, self.padding_output_seq:-self.padding_output_seq], self.samples[idx, 1:2, 1, self.padding_output_seq:-self.padding_output_seq]
            else:
                return tmp_sample, self.samples[idx, 1:2, 0, :], self.samples[idx, 1:2, 1, :]
                