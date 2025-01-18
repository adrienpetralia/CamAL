import os
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler


# ======================================================================================================================================================================= #
# Func for data Processing
# ======================================================================================================================================================================= #


# ========================================= Convert NILM dataset to Classif ========================================= #
def nilmdataset_to_clfdataset(data, st_date=None, return_df=True):
    """
    Input:
        data: Convention of 4D array obtained with every NILM Databuilder available in data_processing

    Output:
        X: 2D array of input time series
        y: 2D array as (len(X), 1) -> sum of energy consumed in each window
    """
    list_label = []
    for i in range(len(data)):
        list_label.append(1) if data[i, 1, 1, :].any() > 0 else list_label.append(0)

    if st_date is not None and return_df:
        return pd.concat([pd.DataFrame(data=data[:, 0, 0, :], index=st_date.index).reset_index(), pd.DataFrame(data=list_label, columns=["label"])], axis=1).set_index('index')
    else:
        return data[:, 0, 0, :], np.array(list_label)


# ========================================= Random Under sampler ========================================= #
def UnderSampler(X, y=None, sampling_strategy='auto', seed=0, nb_label=1):
    np.random.seed(seed)
    
    if isinstance(X, pd.core.frame.DataFrame):
        col = X.columns
        y = X.values[:, -nb_label].astype(int)
        X = X.values[:, :-nb_label]
        X_, y_ = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=seed).fit_resample(X, y)
        Mat = np.concatenate((X_, np.reshape(y_, (y_.shape[0],  1))), axis=1)
        Mat = pd.DataFrame(data=Mat, columns=col)
        Mat = Mat.sample(frac=1, random_state=seed)
        
        return Mat
    else:
        assert y is not None, f"For np.array, please provide an y vector."
        X_, y_ = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=seed).fit_resample(X, y)
        Mat = np.concatenate((X_, np.reshape(y_, (y_.shape[0],  1))), axis=1)
        np.random.shuffle(Mat)
        Mat = Mat.astype(np.float32)
        
        return Mat[:, :-1], Mat[:, -1]

    
def split_train_valid_test(data, test_size=0.2, valid_size=0, nb_label_col=1, seed=0):
    
    if isinstance(data, pd.core.frame.DataFrame):
        X = data.iloc[:,:-nb_label_col].to_numpy().astype(np.float32)
        y = data.iloc[:,-nb_label_col:].to_numpy().astype(np.float32)
    elif isinstance(data, np.ndarray):
        X = data[:,:-nb_label_col]
        y = data[:,-nb_label_col:]
    else:
        raise Exception('Please provide pandas Dataframe or numpy array object.')

    if valid_size != 0:
        X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=valid_size, random_state=seed)
                
        return X_train, y_train, X_valid, y_valid, X_test, y_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        
        return X_train, y_train, X_test, y_test
        
        
def split_train_valid_test_pdl(df_data, test_size=0.2, valid_size=0, nb_label_col=1, seed=0, return_df=False):
    """
    Split DataFrame based on index ID (ID PDL for example)
    
    - Input : df_data -> DataFrame
              test_size -> Percentage data for test
              valid_size -> Percentage data for valid
              nb_label_col -> Number of columns of label
              seed -> Set seed
              return_df -> Return DataFrame instances, or Numpy Instances
    - Output:
            np.arrays or DataFrame Instances
    """

    np.random.seed(seed)
    list_pdl = np.array(df_data.index.unique())
    np.random.shuffle(list_pdl)
    pdl_train_valid = list_pdl[:int(len(list_pdl) * (1-test_size))]
    pdl_test = list_pdl[int(len(list_pdl) * (1-test_size)):]
    np.random.shuffle(pdl_train_valid)
    pdl_train = pdl_train_valid[:int(len(pdl_train_valid) * (1-valid_size))]
    
    df_train = df_data.loc[pdl_train, :].copy()
    df_test = df_data.loc[pdl_test, :].copy()
    

    df_train = df_train.sample(frac=1, random_state=seed)
    df_test = df_test.sample(frac=1, random_state=seed)
    
    if valid_size != 0:
        pdl_valid = pdl_train_valid[int(len(pdl_train_valid) * (1-valid_size)):]
        df_valid = df_data.loc[pdl_valid, :].copy()
        df_valid = df_valid.sample(frac=1, random_state=seed)
            
    if return_df:
        if valid_size != 0:
            return df_train, df_valid, df_test
        else:
            return df_train, df_test
    else:
        X_train = df_train.iloc[:,:-nb_label_col].to_numpy().astype(np.float32)
        y_train = df_train.iloc[:,-nb_label_col:].to_numpy().astype(np.float32)
        X_test  = df_test.iloc[:,:-nb_label_col].to_numpy().astype(np.float32)
        y_test  = df_test.iloc[:,-nb_label_col:].to_numpy().astype(np.float32)

        if valid_size != 0:
            X_valid = df_valid.iloc[:,:-nb_label_col].to_numpy().astype(np.float32)
            y_valid = df_valid.iloc[:,-nb_label_col:].to_numpy().astype(np.float32)

            return X_train, y_train, X_valid, y_valid, X_test, y_test
        else:
            return X_train, y_train, X_test, y_test


def split_train_test_pdl_nilmdataset(data, st_date, seed=0,
                                     nb_house_test=None,  perc_house_test=None,  
                                     nb_house_valid=None, perc_house_valid=None):
    
    assert nb_house_test is not None or perc_house_test is not None
    assert len(data)==len(st_date)
    assert isinstance(st_date, pd.DataFrame)
    
    np.random.seed(seed)
    
    if nb_house_valid is not None or perc_house_valid is not None:
        assert (nb_house_test is not None and nb_house_valid is not None) or (perc_house_test is not None and perc_house_valid is not None)
    
    if len(data.shape) > 2:
        tmp_shape = data.shape
        data = data.reshape(data.shape[0], -1)
        
    data = pd.concat([st_date.reset_index(), pd.DataFrame(data)], axis=1).set_index('index')
    list_pdl = np.array(data.index.unique())
    np.random.shuffle(list_pdl)
    
    if nb_house_test is None:
        nb_house_test = max(1, int(len(list_pdl) * perc_house_test))
        if perc_house_valid is not None and nb_house_valid is None:
            nb_house_valid = max(1, int(len(list_pdl) * perc_house_valid))
        
    if nb_house_valid is not None:
        assert len(list_pdl) > nb_house_test + nb_house_valid
    else:
        assert len(list_pdl) > nb_house_test
    
    pdl_test = list_pdl[:nb_house_test]
    
    if nb_house_valid is not None:
        pdl_valid = list_pdl[nb_house_test:nb_house_test+nb_house_valid]
        pdl_train = list_pdl[nb_house_test+nb_house_valid:]
    else:
        pdl_train  = list_pdl[nb_house_test:]
    
    df_train = data.loc[pdl_train, :].copy()
    df_test  = data.loc[pdl_test, :].copy()
    
    st_date_train = df_train.iloc[:, :1]
    data_train    = df_train.iloc[:, 1:].values.reshape((len(df_train), tmp_shape[1], tmp_shape[2], tmp_shape[3]))
    st_date_test  = df_test.iloc[:, :1]
    data_test     = df_test.iloc[:, 1:].values.reshape((len(df_test), tmp_shape[1], tmp_shape[2], tmp_shape[3]))
    
    if nb_house_valid is not None:
        df_valid      = data.loc[pdl_valid, :].copy()
        st_date_valid = df_valid.iloc[:, :1]
        data_valid    = df_valid.iloc[:, 1:].values.reshape((len(df_valid), tmp_shape[1], tmp_shape[2], tmp_shape[3]))
        
        return data_train, st_date_train, data_valid, st_date_valid, data_test, st_date_test
    else:
        return data_train, st_date_train, data_test, st_date_test


def Split_train_test_NILMDataset(data, st_date, perc_house_test=0.2, seed=0):
    np.random.seed(seed)

    data_len = np.arange(len(data))
    np.random.shuffle(data_len)

    split_index = int(len(data_len) * (1-perc_house_test))
    train_idx, test_idx = data_len[:split_index], data_len[split_index:]

    data_train, st_date_train = data[train_idx], st_date.iloc[train_idx]
    data_test,  st_date_test  = data[test_idx],  st_date.iloc[test_idx]

    return data_train, st_date_train, data_test, st_date_test
        

def reshape_data(X, y, win, win_max=1024):
    factor = int(win_max // win)
    
    X = X.reshape(X.shape[0], X.shape[1], factor, win).transpose(0, 2, 1, 3).reshape(-1, X.shape[1], win)
    tmp = np.empty((len(y), factor), dtype=y.dtype)
    
    for k in range(factor):
        tmp[:, k] = y.ravel()
        
    tmp = tmp.ravel()
    tmp = tmp.reshape(len(tmp), 1)
        
    return X, tmp


# ======================================================================================================================================================================= #
# Datasets Builder: PanelVE, PanelUSage, REFIT, UKDALE
# ======================================================================================================================================================================= #



# ===================== UKDALE DataBuilder =====================#
class UKDALE_DataBuilder(object):
    def __init__(self, 
                 data_path,
                 mask_app,
                 sampling_rate,
                 window_size,
                 window_stride=None,
                 soft_label=False,
                 use_status_from_kelly_paper=False,
                 flag_week=False, flag_day=False):
        
        # =============== Class variables =============== #
        self.data_path = data_path
        self.mask_app = mask_app
        self.sampling_rate = sampling_rate 
        self.window_size = window_size
        self.soft_label = soft_label

        if isinstance(self.mask_app, str):
            self.mask_app = [self.mask_app]

        if isinstance(window_size, str):
            if window_size=='week':
                self.flag_week = True
                self.flag_day  = False
                if (self.sampling_rate=='1min') or (self.sampling_rate=='1T'):
                    self.window_size = 10080
                elif (self.sampling_rate=='10min') or (self.sampling_rate=='10T'):
                    self.window_size = 1008
                else:
                    raise ValueError(f"Only sampling rate 1min and 10min supported for window size='week', got: {sampling_rate}")
            elif window_size=='day':
                self.flag_week = False
                self.flag_day  = True
                if self.sampling_rate=='30s':
                    self.window_size = 2880
                elif (self.sampling_rate=='1min') or (self.sampling_rate=='1T'):
                    self.window_size = 1440
                elif (self.sampling_rate=='10min') or (self.sampling_rate=='10T'):
                    self.window_size = 144
                else:
                    raise ValueError(f"Only sampling rate 30s, 1min and 10min supported for window size='day', got: {sampling_rate}")
            else:
                raise ValueError(f'Only window size = "day" or "week" for window period related (i.e. str type), got: {window_size}')
        else:
            self.flag_week = flag_week
            self.flag_day  = flag_day
            self.window_size = window_size
        
        if window_stride is not None:
            self.window_stride = window_stride
        else:
            self.window_stride = self.window_size

        # ======= Add aggregate to appliance(s) list ======= #
        self._check_appliance_names()
        self.mask_app = ['aggregate'] + self.mask_app

        # ======= Dataset Parameters ======= #
        self.cutoff = 6000

        self.use_status_from_kelly_paper = use_status_from_kelly_paper

        # All parameters are in Watts and taken from Kelly et all. NeuralNILM paper
        if self.use_status_from_kelly_paper:
            # Threshold parameters are in Watts and time parameter in 10sec (minimum resampling)
            self.appliance_param = {'kettle': {'min_threshold': 2000, 'max_threshold': 3100, 'min_on_duration': 1, 'min_off_duration': 0, 'min_activation_time': 1},
                                    'fridge': {'min_threshold': 50, 'max_threshold': 300, 'min_on_duration': 6, 'min_off_duration': 1, 'min_activation_time': 1},
                                    'washing_machine': {'min_threshold': 20, 'max_threshold': 2500, 'min_on_duration': 180, 'min_off_duration': 16, 'min_activation_time': 12},
                                    'microwave': {'min_threshold': 200, 'max_threshold': 3000, 'min_on_duration': 1, 'min_off_duration': 3, 'min_activation_time': 1},
                                    'dishwasher': {'min_threshold': 10, 'max_threshold': 2500, 'min_on_duration': 180, 'min_off_duration': 180, 'min_activation_time': 12},
                               }
        else:
            # Threshold parameters are in Watts
            self.appliance_param = {'kettle': {'min_threshold': 500, 'max_threshold': 6000},
                                    'washing_machine': {'min_threshold': 300, 'max_threshold': 3000},
                                    'dishwasher': {'min_threshold': 300, 'max_threshold': 3000},
                                    'microwave': {'min_threshold': 200, 'max_threshold': 6000},
                                    'fridge': {'min_threshold': 50, 'max_threshold': 300}
                                }

    def get_house_data(self, house_indicies):

        assert len(house_indicies)==1, f'get_house_data() implemented to get data from 1 house only at a time.'

        return self._get_dataframe(house_indicies[0])

    def get_classif_dataset(self, house_indicies):
        """
        Process data to build classif dataset
        
        Return : 
            -Time Series: np.ndarray of size [N_ts, Win_Size]
            -Associated label for each subsequences: np.ndarray of size [N_ts, 1] in 0 or 1 for each TS
            -st_date: pandas.Dataframe as :
                - index: id of each house
                - column 'start_date': Starting timestamp of each TS
        """

        nilm_dataset, st_date = self.get_nilm_dataset(house_indicies)
        y = np.zeros(len(nilm_dataset))

        for idx in range(len(nilm_dataset)):    
            if (nilm_dataset[idx, 1, 1, :] > 0).any():
                y[idx] = 1

        return nilm_dataset[:, 0, 0, :], y, st_date        
        
    def get_nilm_dataset(self, house_indicies, padding=None):
        """
        Process data to build NILM usecase
        
        Return : 
            - np.ndarray of size [N_ts, M_appliances, 2, Win_Size] as :
        
                -1st dimension : nb ts obtained after slicing the total load curve of chosen Houses
                -2nd dimension : nb chosen appliances
                                -> indice 0 for aggregate load curve
                                -> Other appliance in same order as given "appliance_names" list
                -3rd dimension : access to load curve (values of consumption in Wh) or states of activation 
                                of the appliance (0 or 1 for each time step)
                                -> indice 0 : access to load curve
                                -> indice 1 : access to states of activation (0 or 1 for each time step) or Probability (i.e. value in [0, 1]) if soft_label
                -4th dimension : values

            - pandas.Dataframe as :
                index: id of each house
                column 'start_date': Starting timestamp of each TS
        """
        
        output_data = np.array([])
        st_date = pd.DataFrame()
        
        for indice in house_indicies:
            tmp_list_st_date = []
            
            data = self._get_dataframe(indice)
            stems, st_date_stems = self._get_stems(data)

            if padding is not None:
                pad_with = [(0, 0), (padding, padding)]
                stems = np.pad(stems, pad_with)
                st_date_stems = list(pd.date_range(end=st_date_stems[0], periods=padding+1, freq=self.sampling_rate))[:-1] + st_date_stems
                st_date_stems = st_date_stems + list(pd.date_range(start=st_date_stems[-1], periods=padding+1, freq=self.sampling_rate))[1:]

                assert len(stems[0])==len(st_date_stems)
                
            if self.window_size==self.window_stride:
                n_wins = len(data) // self.window_stride
            else:
                n_wins = 1 + ((len(data) - self.window_size) // self.window_stride)
            
            X = np.empty((len(house_indicies) * n_wins, len(self.mask_app), 2, self.window_size))
            
            cpt = 0
            for i in range(n_wins):
                tmp = stems[:, i*self.window_stride:i*self.window_stride+self.window_size]

                if not self._check_anynan(tmp): # Check if nan -> skip the subsequences if it's the case
                    tmp_list_st_date.append(st_date_stems[i*self.window_stride])

                    X[cpt, 0, 0, :] = tmp[0, :]
                    X[cpt, 0, 1, :] = (tmp[0, :] > 0).astype(dtype=int)

                    key = 1
                    for j in range(1, len(self.mask_app)):
                        X[cpt, j, 0, :] = tmp[key, :]
                        X[cpt, j, 1, :] = tmp[key+1, :]
                        key += 2

                    cpt += 1 # Add one subsequence
                    
            tmp_st_date = pd.DataFrame(data=tmp_list_st_date, index=[indice for j in range(cpt)], columns=['start_date'])
            output_data = np.concatenate((output_data, X[:cpt, :, :, :]), axis=0) if output_data.size else X[:cpt, :, :, :]
            st_date = pd.concat([st_date, tmp_st_date], axis=0) if st_date.size else tmp_st_date
                        
        return output_data, st_date
    
    def _compute_status(self, initial_status, min_on, min_off, min_activation_time):
        tmp_status = np.zeros_like(initial_status)
        status_diff = np.diff(initial_status)
        events_idx  = status_diff.nonzero()

        events_idx  = np.array(events_idx).squeeze()
        events_idx += 1

        if initial_status[0]:
            events_idx = np.insert(events_idx, 0, 0)

        if initial_status[-1]:
            events_idx = np.insert(events_idx, events_idx.size, initial_status.size)

        events_idx     = events_idx.reshape((-1, 2))
        on_events      = events_idx[:, 0].copy()
        off_events     = events_idx[:, 1].copy()
        assert len(on_events) == len(off_events)

        if len(on_events) > 0:
            off_duration = on_events[1:] - off_events[:-1]
            off_duration = np.insert(off_duration, 0, 1000)
            on_events    = on_events[off_duration > min_off]
            off_events   = off_events[np.roll(off_duration, -1) > min_off]

            on_duration  = off_events - on_events
            on_events    = on_events[on_duration  >= min_on]
            off_events   = off_events[on_duration >= min_on]
            assert len(on_events) == len(off_events)

        # Filter activations based on minimum continuous points after applying min_on and min_off
        activation_durations = off_events - on_events
        valid_activations = activation_durations >= min_activation_time
        on_events = on_events[valid_activations]
        off_events = off_events[valid_activations]

        for on, off in zip(on_events, off_events):
            tmp_status[on:off] = 1

        return tmp_status
    
    def _get_stems(self, dataframe):
        """
        Extract load curve for each chosen appliances.
        
        Return : np.ndarray instance
        """
        stems = np.empty((1 + (len(self.mask_app)-1)*2, dataframe.shape[0]))
        stems[0, :] = dataframe['aggregate'].values

        key = 1
        for appliance in self.mask_app[1:]:
            stems[key, :] = dataframe[appliance].values
            stems[key+1, :] = dataframe[appliance+'_status'].values
            key+=2

        return stems, list(dataframe.index)
    
    def _get_dataframe(self, indice):
        """
        Load houses data and return one dataframe with aggregate and appliance resampled at chosen time step.
        
        Return : pd.core.frame.DataFrame instance
        """
        path_house = self.data_path+'House'+str(indice)+os.sep
        self._check_if_file_exist(path_house+'labels.dat') # Check if labels exist at provided path
        
        # House labels
        house_label = pd.read_csv(path_house+'labels.dat',    sep=' ', header=None)
        house_label.columns = ['id', 'appliance_name']
        
        # Load aggregate load curve and resample to lowest sampling rate
        house_data = pd.read_csv(path_house+'channel_1.dat', sep=' ', header=None)
        house_data.columns = ['time','aggregate']
        house_data['time'] = pd.to_datetime(house_data['time'], unit = 's')
        house_data = house_data.set_index('time') # Set index to time
        house_data = house_data.resample('10s').mean().fillna(method='ffill', limit=6) # Resample to minimum of 10s and ffill for 1min30
        house_data[house_data < 5] = 0 # Remove small value

        if self.flag_week:
            tmp_min = house_data[(house_data.index.weekday == 0) & (house_data.index.hour == 0) & (house_data.index.minute == 0) & (house_data.index.second == 0)]
            house_data = house_data[house_data.index >= tmp_min.index[0]]
        elif self.flag_day:
            tmp_min = house_data[(house_data.index.hour == 0) & (house_data.index.minute == 0) & (house_data.index.second == 0)]
            house_data = house_data[house_data.index >= tmp_min.index[0]]

        for appliance in self.mask_app[1:]:
            # Check if appliance is in this house
            if len(house_label.loc[house_label['appliance_name']==appliance]['id'].values) != 0:
                i = house_label.loc[house_label['appliance_name']==appliance]['id'].values[0]

                 # Load aggregate load curve and resample to lowest sampling rate
                appl_data = pd.read_csv(path_house+'channel_'+str(i)+'.dat', sep=' ', header=None)
                appl_data.columns = ['time', appliance]
                appl_data['time'] = pd.to_datetime(appl_data['time'],unit = 's')
                appl_data = appl_data.set_index('time')
                appl_data = appl_data.resample('10s').mean().fillna(method='ffill', limit=6) 
                appl_data[appl_data < 5] = 0 # Remove small value
                
                # Merge aggregate load curve with appliance load curve
                house_data = pd.merge(house_data, appl_data, how='inner', on='time')
                del appl_data
                house_data = house_data.clip(lower=0, upper=self.cutoff) # Apply general cutoff
                house_data = house_data.sort_index()

                # Replace nan values by -1 during appliance activation status filtering
                house_data[appliance] = house_data[appliance].replace(np.nan, -1)

                # Creating status
                initial_status = ((house_data[appliance] >= self.appliance_param[appliance]['min_threshold']) & (house_data[appliance] <= self.appliance_param[appliance]['max_threshold'])).astype(int).values
                
                if self.use_status_from_kelly_paper:
                    house_data[appliance+'_status'] = self._compute_status(initial_status, 
                                                                           self.appliance_param[appliance]['min_on_duration'], 
                                                                           self.appliance_param[appliance]['min_off_duration'],
                                                                           self.appliance_param[appliance]['min_activation_time'])
                else:
                    house_data[appliance+'_status'] = initial_status

                # Finally replacing nan values put to -1 by nan
                house_data[appliance] = house_data[appliance].replace(-1, np.nan)

        if self.sampling_rate!='10s':
            house_data = house_data.resample(self.sampling_rate).mean()

        for appliance in self.mask_app[1:]:
            if appliance in house_data:
                if not self.soft_label:
                    house_data[appliance+'_status'] = (house_data[appliance+'_status'] > 0).astype(int)
                else:
                    continue
            else:
                house_data[appliance] = 0
                house_data[appliance+'_status'] = 0

        return house_data
    
    def _check_appliance_names(self):
        """
        Check appliances names for UKDALE case.
        """
        for appliance in self.mask_app:
            assert appliance in ['washing_machine', 'cooker', 'dishwasher', 'kettle', 'fridge', 'microwave', 'electric_heater'], f"Selected applicance unknow for UKDALE Dataset, got: {appliance}"
        return
            
    def _check_if_file_exist(self, file):
        """
        Check if file exist at provided path.
        """
        if os.path.isfile(file):
            pass
        else:
            raise FileNotFoundError
        return
    
    def _check_anynan(self, a):
        """
        Fast check of NaN in a numpy array.
        """
        return np.isnan(np.sum(a))
    

# ===================== REFIT DataBuilder =====================#
class REFIT_DataBuilder(object):
    def __init__(self, 
                 data_path,
                 mask_app,
                 sampling_rate,
                 window_size,
                 window_stride=None,
                 soft_label=False,
                 use_status_from_kelly_paper=False,
                 flag_week=False, flag_day=False):
        
        # =============== Class variables =============== #
        self.data_path = data_path
        self.mask_app = mask_app
        self.sampling_rate = sampling_rate 
        self.soft_label = soft_label

        if isinstance(self.mask_app, str):
            self.mask_app = [self.mask_app]

        if isinstance(window_size, str):
            if window_size=='week':
                self.flag_week = True
                self.flag_day  = False
                if (self.sampling_rate=='1min') or (self.sampling_rate=='1T'):
                    self.window_size = 10080
                elif (self.sampling_rate=='10min') or (self.sampling_rate=='10T'):
                    self.window_size = 1008
                else:
                    raise ValueError(f"Only sampling rate 1min and 10min supported for window size='week', got: {sampling_rate}")
            elif window_size=='day':
                self.flag_week = False
                self.flag_day  = True
                if self.sampling_rate=='30s':
                    self.window_size = 2880
                elif (self.sampling_rate=='1min') or (self.sampling_rate=='1T'):
                    self.window_size = 1440
                elif (self.sampling_rate=='10min') or (self.sampling_rate=='10T'):
                    self.window_size = 144
                else:
                    raise ValueError(f"Only sampling rate 30s, 1min and 10min supported for window size='day', got: {sampling_rate}")
            else:
                raise ValueError(f'Only window size = "day" or "week" for window period related (i.e. str type), got: {window_size}')
        else:
            self.flag_week = flag_week
            self.flag_day  = flag_day
            self.window_size = window_size
        
        if window_stride is not None:
            self.window_stride = window_stride
        else:
            self.window_stride = self.window_size

        # ======= Add aggregate to appliance(s) list ======= #
        self._check_appliance_names()
        self.mask_app = ['Aggregate'] + self.mask_app

        # ======= Dataset Parameters ======= #
        self.cutoff = 10000

        self.use_status_from_kelly_paper = use_status_from_kelly_paper

        # All parameters are in Watts and adapted from Kelly et all. NeuralNILM paper
        if self.use_status_from_kelly_paper:
            # Threshold parameters are in Watts and time parameter in 10sec
            self.appliance_param = {'Kettle': {'min_threshold': 1000, 'max_threshold': 6000, 'min_on_duration': 1, 'min_off_duration': 0, 'min_activation_time': 0},
                                    'WashingMachine': {'min_threshold': 20, 'max_threshold': 3500, 'min_on_duration': 6, 'min_off_duration': 16, 'min_activation_time': 12},
                                    'Dishwasher': {'min_threshold': 50, 'max_threshold': 3000, 'min_on_duration': 2, 'min_off_duration': 180, 'min_activation_time': 12},
                                    'Microwave': {'min_threshold': 200, 'max_threshold': 6000, 'min_on_duration': 1, 'min_off_duration': 3, 'min_activation_time': 0}
                                }
        else:
            # Threshold parameters are in Watts
            self.appliance_param = {'Kettle': {'min_threshold': 500, 'max_threshold': 6000},
                                    'WashingMachine': {'min_threshold': 300, 'max_threshold': 4000},
                                    'Dishwasher': {'min_threshold': 300, 'max_threshold': 4000},
                                    'Microwave': {'min_threshold': 200, 'max_threshold': 6000}
                                }


    def get_house_data(self, house_indicies):

        assert len(house_indicies)==1, f'get_house_data() implemented to get data from 1 house only at a time.'

        return self._get_dataframe(house_indicies[0])

    def get_classif_dataset(self, house_indicies):
        """
        Process data to build classif dataset
        
        Return : 
            -Time Series: np.ndarray of size [N_ts, Win_Size]
            -Associated label for each subsequences: np.ndarray of size [N_ts, 1] in 0 or 1 for each TS
            -st_date: pandas.Dataframe as :
                - index: id of each house
                - column 'start_date': Starting timestamp of each TS
        """

        nilm_dataset, st_date = self.get_nilm_dataset(house_indicies)
        y = np.zeros(len(nilm_dataset))

        for idx in range(len(nilm_dataset)):    
            if (nilm_dataset[idx, 1, 1, :] > 0).any():
                y[idx] = 1

        return nilm_dataset[:, 0, 0, :], y, st_date        
        
    def get_nilm_dataset(self, house_indicies, padding=None):
        """
        Process data to build NILM usecase
        
        Return : 
            - np.ndarray of size [N_ts, M_appliances, 2, Win_Size] as :
        
                -1st dimension : nb ts obtained after slicing the total load curve of chosen Houses
                -2nd dimension : nb chosen appliances
                                -> indice 0 for aggregate load curve
                                -> Other appliance in same order as given "appliance_names" list
                -3rd dimension : access to load curve (values of consumption in Wh) or states of activation 
                                of the appliance (0 or 1 for each time step)
                                -> indice 0 : access to load curve
                                -> indice 1 : access to states of activation (0 or 1 for each time step) or Probability (i.e. value in [0, 1]) if soft_label
                -4th dimension : values

            - pandas.Dataframe as :
                index: id of each house
                column 'start_date': Starting timestamp of each TS
        """
        
        output_data = np.array([])
        st_date = pd.DataFrame()
        
        for indice in house_indicies:
            tmp_list_st_date = []
            
            data = self._get_dataframe(indice)
            stems, st_date_stems = self._get_stems(data)

            if padding is not None:
                pad_with = [(0, 0), (padding, padding)]
                stems = np.pad(stems, pad_with)
                st_date_stems = list(pd.date_range(end=st_date_stems[0], periods=padding+1, freq=self.sampling_rate))[:-1] + st_date_stems
                st_date_stems = st_date_stems + list(pd.date_range(start=st_date_stems[-1], periods=padding+1, freq=self.sampling_rate))[1:]

                assert len(stems[0])==len(st_date_stems)
            
            if self.window_size==self.window_stride:
                n_wins = len(data) // self.window_stride
            else:
                n_wins = 1 + ((len(data) - self.window_size) // self.window_stride)
            
            X = np.empty((len(house_indicies) * n_wins, len(self.mask_app), 2, self.window_size))
            
            cpt = 0
            for i in range(n_wins):
                tmp = stems[:, i*self.window_stride:i*self.window_stride+self.window_size]

                if not self._check_anynan(tmp): # Check if nan -> skip the subsequences if it's the case
                    tmp_list_st_date.append(st_date_stems[i*self.window_stride])

                    X[cpt, 0, 0, :] = tmp[0, :]
                    X[cpt, 0, 1, :] = (tmp[0, :] > 0).astype(dtype=int)

                    key = 1
                    for j in range(1, len(self.mask_app)):
                        X[cpt, j, 0, :] = tmp[key, :]
                        X[cpt, j, 1, :] = tmp[key+1, :]
                        key += 2

                    cpt += 1 # Add one subsequence
                    
            tmp_st_date = pd.DataFrame(data=tmp_list_st_date, index=[indice for j in range(cpt)], columns=['start_date'])
            output_data = np.concatenate((output_data, X[:cpt, :, :, :]), axis=0) if output_data.size else X[:cpt, :, :, :]
            st_date = pd.concat([st_date, tmp_st_date], axis=0) if st_date.size else tmp_st_date
                        
        return output_data, st_date
    
    def _get_stems(self, dataframe):
        """
        Extract load curve for each chosen appliances.
        
        Return : np.ndarray instance
        """
        stems = np.empty((1 + (len(self.mask_app)-1)*2, dataframe.shape[0]))
        stems[0, :] = dataframe['Aggregate'].values

        key = 1
        for appliance in self.mask_app[1:]:
            stems[key, :] = dataframe[appliance].values
            stems[key+1, :] = dataframe[appliance+'_status'].values
            key+=2

        return stems, list(dataframe.index)
    
    def _compute_status(self, initial_status, min_on, min_off, min_activation_time):
        tmp_status = np.zeros_like(initial_status)
        status_diff = np.diff(initial_status)
        events_idx  = status_diff.nonzero()

        events_idx  = np.array(events_idx).squeeze()
        events_idx += 1

        if initial_status[0]:
            events_idx = np.insert(events_idx, 0, 0)

        if initial_status[-1]:
            events_idx = np.insert(events_idx, events_idx.size, initial_status.size)

        events_idx     = events_idx.reshape((-1, 2))
        on_events      = events_idx[:, 0].copy()
        off_events     = events_idx[:, 1].copy()
        assert len(on_events) == len(off_events)

        if len(on_events) > 0:
            off_duration = on_events[1:] - off_events[:-1]
            off_duration = np.insert(off_duration, 0, 1000)
            on_events    = on_events[off_duration > min_off]
            off_events   = off_events[np.roll(off_duration, -1) > min_off]

            on_duration  = off_events - on_events
            on_events    = on_events[on_duration  >= min_on]
            off_events   = off_events[on_duration >= min_on]
            assert len(on_events) == len(off_events)

        # Filter activations based on minimum continuous points after applying min_on and min_off
        activation_durations = off_events - on_events
        valid_activations = activation_durations >= min_activation_time
        on_events = on_events[valid_activations]
        off_events = off_events[valid_activations]

        for on, off in zip(on_events, off_events):
            tmp_status[on:off] = 1

        return tmp_status
    
    def _get_dataframe(self, indice):
        """
        Load houses data and return one dataframe with aggregate and appliance resampled at chosen time step.
        
        Return : pd.core.frame.DataFrame instance
        """
        file = self.data_path+'CLEAN_House'+str(indice)+'.csv'
        self._check_if_file_exist(file)
        labels_houses = pd.read_csv(self.data_path+'HOUSES_Labels').set_index('House_id')

        house_data = pd.read_csv(file)
        house_data.columns = list(labels_houses.loc[int(indice)].values)
        house_data = house_data.set_index('Time').sort_index()
        house_data.index = pd.to_datetime(house_data.index)
        idx_to_drop = house_data[house_data['Issues']==1].index
        house_data = house_data.drop(index=idx_to_drop, axis=0)
        house_data = house_data.resample(rule='10s').mean().ffill(limit=9) # Resample to minimum of 10sec and ffill for 1min30
        house_data[house_data < 5] = 0 # Remove small value
        house_data = house_data.clip(lower=0, upper=self.cutoff) # Apply general cutoff
        house_data = house_data.sort_index()

        if self.flag_week:
            tmp_min = house_data[(house_data.index.weekday == 0) & (house_data.index.hour == 0) & (house_data.index.minute == 0) & (house_data.index.second == 0)]
            house_data = house_data[house_data.index >= tmp_min.index[0]]
        elif self.flag_day:
            tmp_min = house_data[(house_data.index.hour == 0) & (house_data.index.minute == 0) & (house_data.index.second == 0)]
            house_data = house_data[house_data.index >= tmp_min.index[0]]
        
        for appliance in self.mask_app[1:]:
            # Check if appliance is in this house
            if appliance in house_data:

                # Replace nan values by -1 during appliance activation status filtering
                house_data[appliance] = house_data[appliance].replace(np.nan, -1)

                # Creating status
                initial_status = ((house_data[appliance] >= self.appliance_param[appliance]['min_threshold']) & (house_data[appliance] <= self.appliance_param[appliance]['max_threshold'])).astype(int).values
                
                if self.use_status_from_kelly_paper:
                    house_data[appliance+'_status'] = self._compute_status(initial_status, 
                                                                           self.appliance_param[appliance]['min_on_duration'], 
                                                                           self.appliance_param[appliance]['min_off_duration'],
                                                                           self.appliance_param[appliance]['min_activation_time'])
                else:
                    house_data[appliance+'_status'] = initial_status

                # Finally replacing nan values put to -1 by nan
                house_data[appliance] = house_data[appliance].replace(-1, np.nan)

        if self.sampling_rate!='10s':
            house_data = house_data.resample(self.sampling_rate).mean()

        tmp_list = ['Aggregate']
        for appliance in self.mask_app[1:]:
            tmp_list.append(appliance)
            tmp_list.append(appliance+'_status')
            if appliance in house_data:
                if not self.soft_label:
                    house_data[appliance+'_status'] = (house_data[appliance+'_status'] > 0).astype(int)
                else:
                    continue
            else:
                house_data[appliance] = 0
                house_data[appliance+'_status'] = 0

        house_data = house_data[tmp_list]

        return house_data
    
    def _check_appliance_names(self):
        """
        Check appliances names for UKDALE case.
        """
        for appliance in self.mask_app:
            assert appliance in ['WashingMachine', 'Dishwasher', 'Kettle', 'Microwave'], f"Selected applicance unknow for REFIT Dataset, got: {appliance}"
        return
            
    def _check_if_file_exist(self, file):
        """
        Check if file exist at provided path.
        """
        if os.path.isfile(file):
            pass
        else:
            raise FileNotFoundError
        return
    
    def _check_anynan(self, a):
        """
        Fast check of NaN in a numpy array.
        """
        return np.isnan(np.sum(a))
    

# ===================== DataBuilder for IDEAL dataset =====================#
class IDEAL_DataBuilder(object):
    def __init__(self, 
                 data_path,
                 mask_app,
                 window_size,
                 limit_ffill=20,
                 sampling_rate='1min',
                 threshold_app_activation=0,
                 window_stride=None,
                 soft_label=False,
                 flag_day=False, flag_week=False):

        # =============== Class variables =============== #
        self.data_path     = data_path
        self.mask_app      = mask_app
        self.sampling_rate = sampling_rate 
        self.limit_ffill   = limit_ffill
        self.window_size   = window_size
        self.soft_label    = soft_label

        if not isinstance(self.mask_app, str):
            assert ValueError('mask_app need to be a string, NB: list (i.e., multiple appliances) is NOT YET supported for IDEAL dataset.')

        if window_stride is not None:
            self.window_stride = window_stride
        else:
            self.window_stride = self.window_size

        self.flag_week = flag_week
        self.flag_day  = False if self.flag_week else flag_day

        self.threshold_app_activation = threshold_app_activation

        # ======= Load metadata ======= #
        self.metadata = pd.read_csv(self.data_path + 'metadata/appliance.csv')
        self.metadata = self.metadata.loc[self.metadata['powertype']=='electric']

        self.list_id_all_houses           = list(self.metadata['homeid'].unique()) # All houses (256)
        self.list_id_submeters_houses     = self._get_list_houses_submeters() # Households with submeters level info (39)
        self.list_id_submeters_houses_app = self._get_list_houses_submeters_app() # Get list of house's indicies for selected appliance

        # ======= Check appliance name and get IDEAL's appliance list ======= #
        self.list_possible_appliances = self._check_appliance_names()

    def get_classif_dataset(self, w=None):
        """
        Process dataset to get classif (binary) case using all data IDEAL weakly labeled data (256 houses, label given in metadata)

        Input:
            -w: window length
        
        Return : 
            -Time Series: np.ndarray of size [N, w]
            -Associated label for each subsequences: np.ndarray of size [N, 1] in 0 or 1 for each TS
            -st_date: pandas.Dataframe as :
                - index: id of each house
                - column 'start_date': Starting timestamp of each TS
        """
        label = self.metadata.copy()

        if w is None:
            warnings.warn(f'No w provide, use DataBuilder window size {self.window_size}')
            w = self.window_size

        label[self.mask_app] = 0
        if self.mask_app=='washingmachine':
            label.loc[(label['appliancetype']=='washingmachine') | (label['appliancetype']=='washingmachinetumbledrier'), self.mask_app] = 1
        else:
            label.loc[label['appliancetype']==self.mask_app, self.mask_app] = 1

        label = label.groupby('homeid', as_index=False).agg({self.mask_app: 'max', **{col: 'first' for col in label.columns if col != self.mask_app}})
        # label.drop_duplicates('homeid', keep="first", inplace=True)

        label = label[['homeid', self.mask_app]]
        label = label.loc[label['homeid'].isin(self.list_id_all_houses)] # Ensure we hhave only the 256 homeid
        appliance_houses_id_list = label['homeid'].unique()

        data    = np.array([])
        y       = np.array([])
        st_date = pd.DataFrame()

        for indice in appliance_houses_id_list:
            house_data = self._load_aggregate_data(indice)

            if not house_data.size:
                continue

            values = house_data['Aggregate'].values
            tmp_y  = label.loc[label['homeid']==indice][self.mask_app].values[0]

            # Extract the timestamps from the index
            timestamps = house_data.index.values

            # Calculate the number of full subsequences
            n_total        = len(values)
            n_subsequences = n_total // w

            # Truncate the arrays to make them divisible by w
            values_truncated     = values[:n_subsequences * w]
            timestamps_truncated = timestamps[:n_subsequences * w]

            # Reshape the arrays to (n_subsequences, w) and remove subsequences with missing values
            values_reshaped     = values_truncated.reshape(n_subsequences, w)
            timestamps_reshaped = timestamps_truncated.reshape(n_subsequences, w)
            valid_mask          = ~np.isnan(values_reshaped).any(axis=1)

            # Select valid subsequences
            tmp_data = values_reshaped[valid_mask]  # Shape: (n_valid_subsequences, w)

            # Get st_date
            st_date_values = timestamps_reshaped[valid_mask, 0].flatten()
            tmp_st_date    = pd.DataFrame(st_date_values, index=[indice for _ in range(len(st_date_values))], columns=['start_date'])
            tmp_y          = np.array([tmp_y for _ in range(len(st_date_values))])

            data    = np.concatenate((data, tmp_data), axis=0) if data.size else tmp_data
            y       = np.concatenate((y, tmp_y), axis=0) if y.size else tmp_y
            st_date = pd.concat([st_date, tmp_st_date], axis=0) if st_date.size else tmp_st_date
            
        return data, y, st_date
        
    def get_nilm_dataset(self, house_indicies=None):
        """
        Process data to build NILM dataset using the 39ths IDEAL's households equiped with submeters 
        
        Return : 
            - np.ndarray of size [N_ts, M_appliances, 2, Win_Size] as :
        
                -1st dimension : nb ts obtained after slicing the total load curve of chosen Houses
                -2nd dimension : nb chosen appliances
                                -> indice 0 for aggregate load curve
                                -> Other appliance in same order as given "appliance_names" list
                -3rd dimension : access to load curve (values of consumption in Wh) or states of activation 
                                of the appliance (0 or 1 for each time step)
                                -> indice 0 : access to load curve
                                -> indice 1 : access to states of activation (0 or 1 for each time step) or Probability (i.e. value in [0, 1]) if soft_label
                -4th dimension : values

            - pandas.Dataframe as :
                index: id of each house
                column 'start_date': Starting timestamp of each TS
        """
        
        if house_indicies is None:
            house_indicies = self.list_id_submeters_houses_app
            warnings.warn(f'No list of houses indices provide, return dataset for all possible houses for selected appliances according to metadata: {house_indicies}')

        data    = np.array([])
        st_date = pd.DataFrame()

        for house_id in house_indicies:
            tmp_data, tmp_st_date = self._extract_one_house(house_id)

            data    = np.concatenate((data, tmp_data), axis=0) if data.size else tmp_data
            st_date = pd.concat([st_date, tmp_st_date], axis=0) if st_date.size else tmp_st_date
                        
        return data, st_date
    
    def _extract_one_house(self, house_id):
        tmp = self._get_stems(house_id)

        # === If flag_day_or_week, remove values before the first monday midnight  === #
        if self.flag_day:
            tmp_day = tmp.index[(tmp.index.hour == 0) & (tmp.index.minute == 0)]
            if not tmp_day.empty:
                tmp = tmp[tmp.index >= tmp_day[0]]
            del tmp_day

        if self.flag_week:
            tmp_monday = tmp.index[(tmp.index.weekday == 0) & (tmp.index.hour == 0) & (tmp.index.minute == 0)]
            if not tmp_monday.empty:
                tmp = tmp[tmp.index >= tmp_monday[0]]
            del tmp_monday

        app_status = (tmp[self.mask_app].values > self.threshold_app_activation).astype(dtype=int)

        #if self.min_app_activation_time > 1:
        #    app_status = self._compute_status(app_status)

        if self.window_size==self.window_stride:
            n_wins = len(tmp) // self.window_stride
        else:
            n_wins = 1 + ((len(tmp) - self.window_size) // self.window_stride)
            
        if n_wins < 0:
            n_wins = 0 

        X = np.empty((n_wins, 2, 2, self.window_size))
        tmp_df_st_date = []

        cpt = 0
        for i in range(n_wins):
            tmp_win_cdc  = tmp['Aggregate'].values[i*self.window_stride:i*self.window_stride+self.window_size]
            tmp_win_app  = tmp[self.mask_app].values[i*self.window_stride:i*self.window_stride+self.window_size]
            tmp_win_date = tmp.index.values[i*self.window_stride]
            
            if not self._check_anynan(tmp_win_cdc) and not self._check_anynan(tmp_win_app):
                tmp_df_st_date.append(tmp_win_date)
                X[cpt, 0, 0, :] = tmp_win_cdc
                X[cpt, 1, 0, :] = tmp_win_app
                X[cpt, 0, 1, :] = (tmp_win_cdc > 0).astype(dtype=int)
                X[cpt, 1, 1, :] = app_status[i*self.window_stride:i*self.window_stride+self.window_size]
                cpt += 1
        
        df_st_date = pd.DataFrame(data=tmp_df_st_date, index=[house_id for j in range(cpt)], columns=['start_date'])
        
        return X[:cpt, :, :, :], df_st_date

    def _get_stems(self, house_id):
        agg = self._load_aggregate_data(house_id)
        app = self._load_appliance_data(house_id)
        data = pd.merge(agg, app, how='inner', on='time')
        tmp_ind = data[self.mask_app] > data['Aggregate']
        data.loc[tmp_ind, [self.mask_app]] = data[tmp_ind]['Aggregate']

        return data

    def _load_aggregate_data(self, house_id):
        """
        Load aggregate meters data 
        """
        root_dir = self.data_path + 'household_sensors/sensor_data/'

        file = None
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if (f'home{house_id}' in filename) and ('mains_electric-combined' in filename):
                    file = os.path.join(dirpath, filename)
        
        if file is not None:
            data = pd.read_csv(file, compression='gzip', header=None, names=['time', 'Aggregate'], parse_dates=['time'])
            data.set_index('time', inplace=True)
            # data.sort_index(inplace=True)
            if self.sampling_rate!='1min':
                data = data.resample(self.sampling_rate).mean()
            data.fillna(method='ffill', limit=self.limit_ffill, inplace=True)
        else:
            data = pd.DataFrame(columns=['Aggregate'], index=pd.Index([], name='time'))
            warnings.warn(f"Doesn't found {house_id} aggregate recorded data.")

        return data

    def _load_appliance_data(self, house_id):
        """
        Load appliances submeters data 
        """
        if ('shower' in self.mask_app) or ('cooker' in self.mask_app) or ('electrichob' in self.mask_app):
            root_dir = self.data_path + 'household_sensors/sensor_data/'
        else:
            root_dir = self.data_path + 'room_and_appliance_sensors/sensordata/'

        if 'shower' in self.mask_app:
            str_mask_app = 'shower'

        elif 'electrichob' in self.mask_app:
            str_mask_app = 'cooker'
        else:
            str_mask_app = self.mask_app

        file = None
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if (f'home{house_id}' in filename) and (str_mask_app in filename):
                    file = os.path.join(dirpath, filename)

        if file is not None:
            data = pd.read_csv(file, compression='gzip', header=None, names=['time', self.mask_app], parse_dates=['time'])
            data.set_index('time', inplace=True)
            #data.sort_index(inplace=True)
            if self.sampling_rate!='1min':
                data = data.resample(self.sampling_rate).mean()
            data.fillna(method='ffill', limit=self.limit_ffill, inplace=True)
            data.columns = [self.mask_app]
        else:
            data = pd.DataFrame(columns=[self.mask_app], index=pd.Index([], name='time'))
            warnings.warn(f"House {house_id} doesn't have recorded data for: {self.mask_app}.")

        return data

    def _get_list_houses_submeters_app(self):
        if 'shower' in self.mask_app:
            # List of houses that exclude some houses with not accurate label: [90, 128, 146, 259]
            return [63, 65, 73, 106, 145, 162, 171, 175, 208, 225, 227, 228, 249, 266, 311]

        elif ('electrichob' in self.mask_app) or ('cooker' in self.mask_app):
            # List of houses that exclude some houses with not accurate label: [169, 175, 266, 311]
            return [65, 90, 96, 106, 140, 162, 168, 171, 242, 255, 262, 264]
        
        else:
            label = self.metadata.loc[self.metadata['homeid'].isin(self.list_id_submeters_houses)]
            label = label[((label['appliancetype']==self.mask_app) & (label['number']==1))]

            return list(label['homeid'].unique())

    def _get_list_houses_submeters(self):
        """
        Get list of appliance houses with submeters
        """
        root_dir = self.data_path + 'room_and_appliance_sensors/sensordata/'
        
        houses_list = []
        for _, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if 'appliance' in filename:
                    houses_list.append(int(filename[4:int(filename.find('_'))]))

        houses_list = np.unique(np.array(houses_list))

        return houses_list

    def _compute_status(self, initial_status, min_on, min_off, min_activation_time):
        tmp_status = np.zeros_like(initial_status)
        status_diff = np.diff(initial_status)
        events_idx  = status_diff.nonzero()

        events_idx  = np.array(events_idx).squeeze()
        events_idx += 1

        if initial_status[0]:
            events_idx = np.insert(events_idx, 0, 0)

        if initial_status[-1]:
            events_idx = np.insert(events_idx, events_idx.size, initial_status.size)

        events_idx     = events_idx.reshape((-1, 2))
        on_events      = events_idx[:, 0].copy()
        off_events     = events_idx[:, 1].copy()
        assert len(on_events) == len(off_events)

        if len(on_events) > 0:
            off_duration = on_events[1:] - off_events[:-1]
            off_duration = np.insert(off_duration, 0, 1000)
            on_events    = on_events[off_duration > min_off]
            off_events   = off_events[np.roll(off_duration, -1) > min_off]

            on_duration  = off_events - on_events
            on_events    = on_events[on_duration  >= min_on]
            off_events   = off_events[on_duration >= min_on]
            assert len(on_events) == len(off_events)

        # Filter activations based on minimum continuous points after applying min_on and min_off
        activation_durations = off_events - on_events
        valid_activations = activation_durations >= min_activation_time
        on_events = on_events[valid_activations]
        off_events = off_events[valid_activations]

        for on, off in zip(on_events, off_events):
            tmp_status[on:off] = 1

        return tmp_status
    
    def _check_appliance_names(self):
        """
        Check appliances names for UKDALE case.
        """
        list_possible_appliances = list(self.metadata['appliancetype'].unique()) + ['cooker']

        assert self.mask_app in list_possible_appliances, f"Selected applicance unknow for IDEAL Dataset, got: {self.mask_app}, possible appliance are : {list_possible_appliances}"

        return list_possible_appliances
            
    def _check_if_file_exist(self, file):
        """
        Check if file exist at provided path.
        """
        if os.path.isfile(file):
            pass
        else:
            raise FileNotFoundError
        return
    
    def _check_anynan(self, a):
        """
        Fast check of NaN in a numpy array.
        """
        return np.isnan(np.sum(a))