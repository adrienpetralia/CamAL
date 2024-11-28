import os
import numpy as np
import matplotlib.pyplot as plt

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


def create_dir(path):
    os.makedirs(path, exist_ok=True)

    return path


def check_file_exist(path):
    return os.path.isfile(path)


def fmax(val):
    if val > 0 : return val
    else: return 0
def fmin(val):
    if val < 0 : return val
    else: return 0


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
