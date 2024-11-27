import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ========================================= Graphics setting ========================================= #
def apply_graphics_setting(ax=None, legend_font_size=20, label_fontsize=20):

    if ax is None:
        ax = plt.gca()
        for pos in ['right', 'top', 'bottom', 'left']:
            ax.spines[pos].set_visible(False)

        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(label_fontsize)  
            
        
        plt.grid(linestyle='-.') 
        plt.legend(fontsize=legend_font_size)
        plt.tight_layout()
    else:
        for pos in ['right', 'top', 'bottom', 'left']:
            ax.spines[pos].set_visible(False)

        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(label_fontsize)  

        ax.grid(linestyle='-.') 
        ax.legend(fontsize=legend_font_size)
        ax.figure.tight_layout()


# ========================================= Managing path and file ========================================= #
def create_dir(path):
    os.makedirs(path, exist_ok=True)

    return path

def check_file_exist(path):
    return os.path.isfile(path)

def rename_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # Check if the filename ends with '.pt.pt'
            if filename.endswith('.pt.pt'):
                # Construct the old file path
                old_file = os.path.join(dirpath, filename)
                # Remove the extra '.pt' to get the new filename
                new_filename = filename[:-3]
                # Construct the new file path
                new_file = os.path.join(dirpath, new_filename)
                # Rename the file
                os.rename(old_file, new_file)
                print(f"Renamed '{old_file}' to '{new_file}'")


# ========================================= Math fun ========================================= #
def fmax(val):
    if val > 0 : return val
    else: return 0
def fmin(val):
    if val < 0 : return val
    else: return 0

def threshold_cam(vector, threshold=0.5): 
    output = np.zeros_like(vector).astype(dtype=int)
    output[vector >= threshold] = 1

    return output

def sigmoid(z):
    return 2 * (1.0/(1.0 + np.exp(-z))) -1

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w



# ========================================= Metrics ========================================= #
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

class Classifmetrics():
    """
    Basics metrics for classification
    """
    def __init__(self, round_to=5):
        self.round_to=round_to
        
    def __call__(self, y, y_hat):
        metrics = {}

        y_hat_round = y_hat.round()

        metrics['ACCURACY'] = round(accuracy_score(y, y_hat_round), self.round_to)
        metrics['BALANCED_ACCURACY'] = round(balanced_accuracy_score(y, y_hat_round), self.round_to)
        
        metrics['PRECISION'] = round(precision_score(y, y_hat_round), self.round_to)
        metrics['RECALL'] = round(recall_score(y,y_hat_round), self.round_to)
        metrics['F1_SCORE'] = round(f1_score(y, y_hat_round), self.round_to)
        metrics['F1_SCORE_MACRO'] = round(f1_score(y, y_hat_round, average='macro'), self.round_to)
        
        metrics['ROC_AUC_SCORE'] = round(roc_auc_score(y, y_hat), self.round_to)
        metrics['AP'] = round(average_precision_score(y, y_hat), self.round_to)

        return metrics
    

class NILMmetrics():
    """
    Basics metrics for NILM
    """
    def __init__(self, round_to=3):
        self.round_to=round_to
        
    def __call__(self, y=None, y_hat=None, y_state=None, y_hat_state=None):
        metrics = {}

        # ======= Basic regression Metrics ======= #
        if y is not None:
            assert y_hat is not None, 'Target y_hat not provided, please provide y_hat to compute regression metrics.'
            y     = np.nan_to_num(y.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            y_hat = np.nan_to_num(y_hat.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

            # MAE, MSE and RMSE
            metrics['MAE']  = round(mean_absolute_error(y, y_hat), self.round_to)
            metrics['MSE']  = round(mean_squared_error(y, y_hat), self.round_to)
            metrics['RMSE'] = round(np.sqrt(mean_squared_error(y, y_hat)), self.round_to)

            # =======  NILM Metrics ======= #

            # Total Energy Correctly Assigned (TECA)
            metrics['TECA'] = round(1 - ((np.sum(np.abs(y_hat - y))) / (2*np.sum(np.abs(y)))), self.round_to)
            # Normalized Disaggregation Error (NDE)
            metrics['NDE']  = round((np.sum((y_hat - y)**2)) / np.sum(y**2), self.round_to)
            # Signal Aggregate Error (SAE)
            metrics['SAE']  = round(np.abs(np.sum(y_hat) - np.sum(y)) / np.sum(y), self.round_to)
            # Matching Rate
            metrics['MR']   = round(np.sum(np.minimum(y_hat, y)) / np.sum(np.maximum(y_hat, y)), self.round_to)

        # =======  Event Detection Metrics ======= #
        if y_state is not None:
            assert y_hat_state is not None, 'Target y_hat_state not provided, please pass y_hat_state to compute classification metrics.'
            y_state     = np.nan_to_num(y_state.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            y_hat_state = np.nan_to_num(y_hat_state.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            y_hat_state = np.round(y_hat_state)

            # Accuracy and Balanced Accuracy
            metrics['ACCURACY'] = round(accuracy_score(y_state, y_hat_state), self.round_to)
            metrics['BALANCED_ACCURACY'] = round(balanced_accuracy_score(y_state, y_hat_state), self.round_to)
            # Pr, Rc and F1 Score
            metrics['PRECISION'] = round(precision_score(y_state, y_hat_state), self.round_to)
            metrics['RECALL']    = round(recall_score(y_state,y_hat_state), self.round_to)
            metrics['F1_SCORE']  = round(f1_score(y_state, y_hat_state), self.round_to)

        return metrics

class REGmetrics():
    """
    Basics regrssion metrics
    """
    def __init__(self, round_to=5):
        self.round_to=round_to
        
    def __call__(self, y, y_hat):
        metrics = {}

        metrics['MAE']  = round(mean_absolute_error(y, y_hat), self.round_to)
        metrics['MSE']  = round(mean_squared_error(y, y_hat), self.round_to)
        metrics['RMSE'] = round(np.sqrt(mean_squared_error(y, y_hat)), self.round_to)
        metrics['MAPE'] = round(mean_absolute_percentage_error(y, y_hat), self.round_to)

        return metrics


# ========================================= RPU metrics ========================================= #
def eval_win_energy_aggregation(input_data_test, 
                                input_st_date_test, 
                                model_trainer, 
                                scaler, 
                                metrics,
                                window_size, 
                                freq,
                                mask_metric='test_metrics',
                                list_exo_variables=[],
                                threshold_small_values=0,
                                use_temperature=False):

    data_test    = input_data_test.copy()
    st_date_test = input_st_date_test.copy()

    st_date_test.index.name = 'ID_PDL'
    st_date_test = st_date_test.reset_index()
    list_pdl_test = st_date_test['ID_PDL'].unique()

    for freq_agg in ['D', 'W', 'M']:
        df = pd.DataFrame()

        true_app_power = []
        pred_app_power = []
        true_ratio = []
        pred_ratio = []

        for pdl in list_pdl_test:
            tmp_st_date_test = st_date_test.loc[st_date_test['ID_PDL']==pdl]

            list_index = tmp_st_date_test.index

            list_date = []
            pdl_total_power = []
            pdl_true_app_power = []
            pdl_pred_app_power = []

            for k, val in enumerate(list(list_index)):

                if list_exo_variables is not None:
                    if use_temperature:
                        input_seq = torch.Tensor(create_exogene(data_test[val, 0, :2, :], tmp_st_date_test.iloc[k, 1], list_exo_variables=list_exo_variables, freq=freq))
                    else:
                        input_seq = torch.Tensor(create_exogene(data_test[val, 0, 0, :],  tmp_st_date_test.iloc[k, 1], list_exo_variables=list_exo_variables, freq=freq))
                else:
                    if use_temperature:
                        input_seq = torch.Tensor(np.expand_dims(data_test[val, 0, :2, :], axis=0))
                    else:
                        input_seq = torch.Tensor(np.expand_dims(np.expand_dims(data_test[val, 0, 0, :],  axis=0), axis=0))

                pred = model_trainer.model(input_seq.to(model_trainer.device))

                pred = scaler.inverse_transform_appliance(pred)

                pred[pred<threshold_small_values] = 0 

                inv_scale = scaler.inverse_transform(data_test[val, :, :, :])

                agg = inv_scale[0, 0, :]
                app = inv_scale[1, 0, :]

                pred = pred.detach().cpu().numpy().flatten()

                list_date.extend(list(pd.date_range(tmp_st_date_test.iloc[k, 1], periods=window_size, freq=freq)))
                pdl_total_power.extend(list(agg))
                pdl_true_app_power.extend(list(app))
                pdl_pred_app_power.extend(list(pred))

            df_inst = pd.DataFrame(list(zip(list_date, pdl_total_power, pdl_true_app_power, pdl_pred_app_power)), columns=['date', 'total_power', 'true_app_power', 'pred_app_power'])
            df_inst['date'] = pd.to_datetime(df_inst['date'])
            df_inst = df_inst.set_index('date')

            df_inst = df_inst.groupby(pd.Grouper(freq=freq_agg)).sum()
            df_inst += 1 # Prevent total power or appliance power is 0 to calculate ratio

            true_app_power.extend(df_inst['true_app_power'].tolist())
            pred_app_power.extend(df_inst['pred_app_power'].tolist())

            df_inst['true_ratio'] = df_inst['true_app_power'] / df_inst['total_power']
            df_inst['pred_ratio'] = df_inst['pred_app_power'] / df_inst['total_power']

            df_inst = df_inst.fillna(value=0)

            true_ratio.extend(df_inst['true_ratio'].tolist())
            pred_ratio.extend(df_inst['pred_ratio'].tolist())

            df = df_inst if not df.size else pd.concat((df, df_inst), axis=0)

        model_trainer.log[mask_metric+'_'+freq_agg] = metrics(np.array(true_app_power), np.array(pred_app_power))
        
        true_ratio = np.nan_to_num(np.array(true_ratio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        pred_ratio = np.nan_to_num(np.array(pred_ratio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        tmp_dict_ratio = metrics(true_ratio, pred_ratio)

        for name_m, values in tmp_dict_ratio.items():
            tmp_dict_ratio[name_m] = values * 100

        model_trainer.log[mask_metric+'_ratio_'+freq_agg] = tmp_dict_ratio
        model_trainer.log[mask_metric+'_ratio_'+freq_agg]['True_Ratio'] = np.mean(np.array(true_ratio)) * 100

    model_trainer.save()

    return

def create_exogene(values, st_date, list_exo_variables=['hour', 'dow'], freq='30T'):
        """
        Add Exogene to data

        values: 1D np.array
        st_date: First value Timestamp
        list_exo_variables: 'hour', 'dow', 'dom'
        """

        tmp = pd.date_range(start=st_date, periods=len(values[-1]) if len(values.shape) > 1 else len(values), freq=freq)

        np_extra = np.empty((1, 2*len(list_exo_variables), len(values[-1]) if len(values.shape) > 1 else len(values))).astype(np.float32)

        k=0
        for exo_var in list_exo_variables:
            if exo_var=='month':
                np_extra[0, k, :]   = np.sin(2 * np.pi * tmp.month.values/12.0)
                np_extra[0, k+1, :] = np.cos(2 * np.pi * tmp.month.values/12.0)
                k+=2
            elif exo_var=='dom':
                np_extra[0, k, :]   = np.sin(2 * np.pi * tmp.day.values/31.0)
                np_extra[0, k+1, :] = np.cos(2 * np.pi * tmp.day.values/31.0)
                k+=2
            elif exo_var=='dow':
                np_extra[0, k, :]   = np.sin(2 * np.pi * tmp.dayofweek.values/7.0)
                np_extra[0, k+1, :] = np.cos(2 * np.pi * tmp.dayofweek.values/7.0)
                k+=2
            elif exo_var=='hour':
                np_extra[0, k, :]   = np.sin(2 * np.pi * tmp.hour.values/24.0)
                np_extra[0, k+1, :] = np.cos(2 * np.pi * tmp.hour.values/24.0)
                k+=2
            elif exo_var=='minute':
                np_extra[0, k, :]   = np.sin(2 * np.pi * tmp.minute.values/60.0)
                np_extra[0, k+1, :] = np.cos(2 * np.pi * tmp.minute.values/60.0)
                k+=2
            else:
                raise ValueError("Embedding unknown for these Data. Only 'month', 'dow', 'dom', 'hour' and 'minute' supported.")   

        if len(values.shape)==1:
            values = np.expand_dims(np.expand_dims(values, axis=0), axis=0)
        elif len(values.shape)==2:
            values = np.expand_dims(values, axis=0)

        values = np.concatenate((values, np_extra), axis=1)
            
        return values