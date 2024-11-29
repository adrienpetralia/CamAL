import os, lzma, io
import numpy as np
import pickle
import heapq
import time
import warnings
import torch

from Helpers.class_activation_map import CAM
from Helpers.data_processing import NILMdataset_to_Clf, UnderSampler
from Helpers.torch_dataset import SimpleDataset, TSDataset
from Helpers.torch_trainer import BasedClassifTrainer
from Helpers.other import NILMmetrics, Classifmetrics
from Models.Classifiers.CamALResNet import CamALResNet


from torch.utils.data import DataLoader

class CamAL(object):
    def __init__(self, 
                 path,
                 device='cpu',
                 kernel_sizes=[5, 7, 9, 15, 25],
                 n_best_clf=5,
                 n_try_clf=3,
                 batch_inference=128,
                 timestamp_metrics=NILMmetrics(),
                 clf_metrics=Classifmetrics()):

        self.path   = path
        self.device = device

        self.kernel_sizes = kernel_sizes
        self.n_best_clf   = n_best_clf
        self.n_try_clf    = n_try_clf

        self.batch_inference   = batch_inference
        self.timestamp_metrics = timestamp_metrics
        self.clf_metrics       = clf_metrics

        # ========== Hyperparameters ========== #
        self.p_es_ensemble   = 5
        self.p_rlr_ensemble  = 3
        self.n_warmup_epochs = 1


    def train(self,
              train_dataset, 
              valid_dataset,
              test_dataset,
              lr = 1e-3,
              batch_size=128,
              epochs=50,
              balance_class=True,
              compress_and_save_space=True):
        """
        Inputs:
        - param_training_global: [dict] with expes parameters
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

        # Init dicts result
        tmp_dict_results = {}
        dict_results     = {}


        if self.verbose:
            print('Train ResNet clf ensemble')

        tmp_time = time.time()

        idx_clf = 0
        for kernel_size in self.kernel_sizes:
            for i in range(self.n_try_clf):

                resnet_inst = CamALResNet(kernel_size=kernel_size)

                # Dataloader
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1,  shuffle=False)
                test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=1,  shuffle=False)

                # Init trainer
                clf_trainer = BasedClassifTrainer(resnet_inst,
                                                  train_loader=train_loader, valid_loader=valid_loader,
                                                  learning_rate=lr, weight_decay=0,
                                                  patience_es=self.p_es_ensemble, patience_rlr=self.p_rlr_ensemble,
                                                  n_warmup_epochs=self.n_warmup_epochs,
                                                  device=self.device, all_gpu=False,
                                                  verbose=False, plotloss=False, 
                                                  save_checkpoint=True, path_checkpoint=f'{self.path}ResNet_{idx_clf}')

                # Train 
                clf_trainer.train(epochs)

                # Eval
                clf_trainer.restore_best_weights()
                loss_eval, metrics = clf_trainer.evaluate(test_loader)

                if compress_and_save_space:
                    # Attention, model logs compressed! -> file end with ".xz" not ".pt"
                    clf_trainer.compress()

                if self.verbose:
                    print(f'Trained ResNet_{idx_clf} with Kernel size: {kernel_size} (nth_try {i}).')
                    print('Loss test:', loss_eval)
                    print('Score:', metrics)

                # Save loss value
                tmp_dict_results[f'ResNet_{idx_clf}'] = loss_eval
                dict_results[f'ResNet_{idx_clf}']     = {'LossTest': loss_eval, 'MetricTest': metrics, 'kernel_size': kernel_size, 'nth_try': i}

                # Update counters
                idx_clf +=1

        dict_results['TrainingTime'] = round((time.time() - tmp_time), 3)

        if self.verbose:
            print('Training time:', dict_results['TrainingTime'])
        
        heap = [(loss, name) for name, loss in tmp_dict_results.items()]
        heapq.heapify(heap)
        smallest = heapq.nsmallest(self.n_best_clf, heap)

        # Get the name of the best clf and select them for final ensemble
        list_best_clf = [name for _, name in smallest]
        if self.verbose:
            print('List best clf:', list_best_clf)
        dict_results['ListBestResNets'] = list_best_clf

        # Delete the clfs for the ensemble (worst performance)
        if self.verbose:
            print('Delete worst clf...')

        if compress_and_save_space:
            for i in range(idx_clf):
                if not (f'ResNet_{i}' in list_best_clf):
                    os.remove(f'{self.path}ResNet_{i}.xz')
                    if self.verbose:
                        print(f'ResNet_{i} removed.')
        if self.verbose:
            print('Done.')

        # Save ensemble dict_results in pkl file at provided path
        name_file = f'{self.path}LogResNetsEnsemble.pkl'

        print(f'Save ResNets ensemble log at path: {self.path}LogResNetsEnsemble.pkl')
        with open(name_file, 'wb') as f:
            pickle.dump(dict_results, f, protocol=pickle.HIGHEST_PROTOCOL)

        return dict_results
    

    def test(self,
             data_test,
             path_ensemble_clf=None,
             appliance_mean_on_power=None,
             scaler=None, 
             return_clf_perf=True):

        if path_ensemble_clf is None:
            path_ensemble_clf = self.path+'ResNetEnsemble/'
            
            if not os.path.exists(path_ensemble_clf):
                warnings.warn(f"No ensemble trained at provide path: {path_ensemble_clf}. be sure that the provided path include an")
                return
            
        if len(data_test.shape)<4:
            raise ValueError("Provided test data need to be given in NILM standard (4D numpy array)")

        # Create soft label
        soft_label, prob_detect = self.create_soft_label(path_ensemble_clf,
                                                         data_test[:, 0, 0, :],
                                                         return_prob=True)

        # Get y_state and y_hat_state
        y_state     = data_test[:, 1, 1, :].ravel().astype(dtype=int)
        y_hat_state = soft_label.ravel().astype(dtype=int)


        if appliance_mean_on_power is not None:
            if scaler is not None:
                data_test_rescale = scaler.inverse_transform(data_test)
            warnings.warn("No scaler provided for inverse_transform, are you sure you didn't scale the data for training?")
            # Get true appliance power
            
            tmp_agg = data_test_rescale[:, 0, 0, :].ravel()
            y       = data_test_rescale[:, 1, 0, :].ravel()

            # Create soft label power value according to mean appliance power param
            y_hat = y_hat_state * appliance_mean_on_power
            # Ensure that app consumption doesn't exceed aggregate
            y_hat[y_hat>tmp_agg] = tmp_agg[y_hat>tmp_agg]
            del data_test_rescale
            del tmp_agg

            # Compute metric (NILM reg + classif)
            metric_softlabel = self.timestamp_metrics(y, y_hat, y_state, y_hat_state)
        else: 
            # Compute metric (NILM classif)
            metric_softlabel = self.timestamp_metrics(y=None, y_hat=None, y_state=y_state, y_hat_state=y_hat_state)

        _, y_clf_true = NILMdataset_to_Clf(data_test)
        metric_classif = self.clf_metrics(y=y_clf_true.ravel(), y_hat=prob_detect.ravel())

        if return_clf_perf:
            return metric_softlabel, metric_classif
        else:
            return metric_softlabel
    

    def create_soft_label(self,
                          path_ensemble_clf,
                          data,
                          y=None,
                          return_prob=False):
        
        if path_ensemble_clf is None:
            path_ensemble_clf = self.path+'ResNetEnsemble/'
            
            if not os.path.exists(path_ensemble_clf):
                warnings.warn(f"No ensemble trained at provide path: {path_ensemble_clf}. be sure that the provided path include an")
                return

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
            resnet_inst = CamALResNet(kernel_size=dict_results[resnet_name]['kernel_size'])
            resnet_inst.to(self.device)

            if os.path.exists(f'{path_ensemble_clf}{resnet_name}.xz'):
                path_model = f'{path_ensemble_clf}{resnet_name}.xz'
                with lzma.open(path_model, 'rb') as file:
                    decompressed_file = file.read()
                log = torch.load(io.BytesIO(decompressed_file), map_location=torch.device(self.device))
                del decompressed_file
            elif os.path.exists(f'{path_ensemble_clf}{resnet_name}.pt'):
                path_model = f'{path_ensemble_clf}{resnet_name}.pt'
                log = torch.load(path_model, map_location=torch.device(self.device))
            else:
                raise ValueError(f'Provide folders {path_ensemble_clf} does not contain {resnet_name} clf.')

            resnet_inst.load_state_dict(log['model_state_dict'])
            resnet_inst.eval()

            # Get first the per window prediction labels using large batch
            if (self.device!='cpu') and (y is None):
                y = self._get_window_labels(resnet_inst, data)

            for idx in range(len(data)): 
                # Use provide window label to speed up (by not computing the CAM)
                if y is not None:
                    if y[idx]<1:
                        continue

                CAM_builder = CAM(model=resnet_inst, device=self.device, 
                                  last_conv_layer=resnet_inst._modules['layers'][2], 
                                  fc_layer_name=resnet_inst._modules['linear'])
                
                cam, y_pred, proba   = CAM_builder.run(instance=data[idx], returned_cam_for_label=1)
                prob_detect[idx][0] += proba[1]
                    
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
        soft_label  = self.SigmoidAttentionModule(data, soft_label)

        if return_prob:
            return soft_label, prob_detect
        else:
            return soft_label


    def SigmoidAttentionModule(self, data, soft_label, w_ma=5):
        # Apply moving average
        soft_label = np.apply_along_axis(lambda x: self.moving_average(x, w=w_ma), axis=1, arr=soft_label)

        # Apply sigmoid on the product of averaged soft labels
        soft_label = self.sigmoid(soft_label * data)

        # Thresholding to obtain binary labels
        return np.round(soft_label)


    def _get_window_labels(self, model, data):
        model.to(self.device)
        model.eval()
        data   = SimpleDataset(data)
        loader = DataLoader(data, batch_size=self.batch_inference, shuffle=False)

        y_hat = np.array([])
        with torch.no_grad():
            for ts in loader:
                logits       = model(torch.Tensor(ts.float()).to(self.device))
                _, predicted = torch.max(logits, 1)

                y_hat = np.concatenate((y_hat, predicted.detach().cpu().numpy().flatten())) if y_hat.size else predicted.detach().cpu().numpy().flatten()

        return y_hat.astype(np.int8)
    

    def _sigmoid(self, z):
        return 2 * (1.0/(1.0 + np.exp(-z))) -1


    def _moving_average(self, x, w):
        return np.convolve(x, np.ones(w), 'same') / w
