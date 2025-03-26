import os
import time
import warnings
import lzma
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from src.helpers.other import Classifmetrics, NILMmetrics
    

class BasedClassifTrainer(object):
    def __init__(self,
                 model, 
                 train_loader, valid_loader=None,
                 learning_rate=1e-3, weight_decay=1e-2, 
                 criterion=nn.CrossEntropyLoss(),
                 patience_es=None, patience_rlr=None,
                 device="cuda", all_gpu=False,
                 valid_criterion=None,
                 n_warmup_epochs=0,
                 f_metrics=Classifmetrics(),
                 verbose=False, plotloss=False, 
                 save_fig=False, path_fig=None,
                 save_checkpoint=False, path_checkpoint=None):
        """
        PyTorch Model Trainer Class for classification case
        """

        # =======================class variables======================= #
        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.f_metrics = f_metrics
        self.device = device
        self.all_gpu = all_gpu
        self.verbose = verbose
        self.plotloss = plotloss
        self.save_checkpoint = save_checkpoint
        self.path_checkpoint = path_checkpoint
        self.save_fig = save_fig
        self.path_fig = path_fig
        self.patience_rlr = patience_rlr
        self.patience_es = patience_es
        self.n_warmup_epochs = n_warmup_epochs
        self.scheduler = None
        
        self.train_criterion = criterion
        if valid_criterion is None:
            self.valid_criterion = criterion
        else:
            self.valid_criterion = valid_criterion
        
        if self.path_checkpoint is not None:
            self.path_checkpoint = path_checkpoint
        else:
            self.path_checkpoint = os.getcwd()+os.sep+'model' 
            
        if patience_rlr is not None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', 
                                                                        patience=patience_rlr, 
                                                                        eps=1e-7)

        self.log = {}
        self.train_time = 0
        self.eval_time = 0
        self.voter_time = 0
        self.passed_epochs = 0
        self.best_loss = np.inf
        self.loss_train_history = []
        self.loss_valid_history = []
        self.accuracy_train_history = []
        self.accuracy_valid_history = []
               
        if self.patience_es is not None:
            self.early_stopping = EarlyStopper(patience=self.patience_es)

        if self.all_gpu:
            # =========== Dummy forward to intialize Lazy Module if all GPU used =========== #
            self.model.to("cpu")
            for ts, _ in train_loader:
                self.model(torch.rand(ts.shape))
                break
            # =========== Data Parrallel Module call =========== #
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
    
    def train(self, n_epochs=10):
        """
        Public function : master training loop over epochs
        """
        
        #flag_es = 0
        tmp_time = time.time()
        
        for epoch in range(n_epochs):
            # =======================one epoch======================= #
            train_loss, train_accuracy = self.__train()
            self.loss_train_history.append(train_loss)
            self.accuracy_train_history.append(train_accuracy)
            if self.valid_loader is not None:
                valid_loss, valid_accuracy = self.__evaluate()
                self.loss_valid_history.append(valid_loss)
                self.accuracy_valid_history.append(valid_accuracy)
            else:
                valid_loss = train_loss
                
            # =======================reduce lr======================= #
            if self.scheduler:
                self.scheduler.step(valid_loss)

            # ===================early stoppping=================== #
            if self.patience_es is not None:
                if self.passed_epochs > self.n_warmup_epochs: # Avoid n_warmup_epochs first epochs
                    if self.early_stopping.early_stop(valid_loss):
                        #flag_es  = 1
                        es_epoch = epoch+1
                        self.passed_epochs+=1
                        if self.verbose:
                            print('Early stopping after {} epochs !'.format(epoch+1))
                        break
        
            # =======================verbose======================= #
            if self.verbose:
                print('Epoch [{}/{}]'.format(epoch+1, n_epochs))
                print('    Train loss : {:.4f}, Train acc : {:.2f}%'
                          .format(train_loss, train_accuracy*100))
                
                if self.valid_loader is not None:
                    print('    Valid  loss : {:.4f}, Valid  acc : {:.2f}%'
                              .format(valid_loss, valid_accuracy*100))

            # =======================save log======================= #
            if valid_loss <= self.best_loss and self.passed_epochs>=self.n_warmup_epochs:
                self.best_loss = valid_loss
                self.log = {'valid_metrics': valid_accuracy if self.valid_loader is not None else train_accuracy,
                            'best_model_state_dict': self.model.module.state_dict() if self.device=="cuda" and self.all_gpu else self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss_train_history': self.loss_train_history,
                            'loss_valid_history': self.loss_valid_history,
                            'accuracy_train_history': self.accuracy_train_history,
                            'accuracy_valid_history': self.accuracy_valid_history,
                            'value_best_loss': self.best_loss,
                            'epoch_best_loss': self.passed_epochs,
                            'time_best_loss': round((time.time() - tmp_time), 3),
                            }
                if self.save_checkpoint:
                    self.save()
                
            self.passed_epochs+=1
                    
        self.train_time = round((time.time() - tmp_time), 3)

        if self.plotloss:
            self.plot_history()
        
        # =======================update log======================= #
        self.log['training_time'] = self.train_time
        self.log['loss_train_history'] = self.loss_train_history
        self.log['loss_valid_history'] = self.loss_valid_history
        self.log['accuracy_train_history'] = self.accuracy_train_history
        self.log['accuracy_valid_history'] = self.accuracy_valid_history
        
        if self.save_checkpoint:
            self.save()

        return
    
    def evaluate(self, test_loader, mask='test_metrics', return_output=False):
        """
        Public function : model evaluation on test dataset
        """
        tmp_time = time.time()
        mean_loss_eval = []
        y = np.array([])
        y_hat = np.array([])
        with torch.no_grad():
            for ts, labels in test_loader:
                self.model.eval()
                # ===================variables=================== #
                ts = torch.Tensor(ts.float()).to(self.device)
                labels = torch.Tensor(labels.float()).to(self.device)
                # ===================forward===================== #
                logits = self.model(ts)
                loss = self.valid_criterion(logits.float(), labels.long())
                # =================concatenate=================== #
                _, predicted = torch.max(logits, 1)
                mean_loss_eval.append(loss.item())
                y_hat = np.concatenate((y_hat, predicted.detach().cpu().numpy())) if y_hat.size else predicted.detach().cpu().numpy()
                y = np.concatenate((y, torch.flatten(labels).detach().cpu().numpy())) if y.size else torch.flatten(labels).detach().cpu().numpy()
                
        metrics = self.f_metrics(y, y_hat)
        self.eval_time = round((time.time() - tmp_time), 3)
        self.log[mask+'_time'] = self.eval_time
        self.log[mask] = metrics
        
        if self.save_checkpoint:
            self.save()
        
        if return_output:
            return np.mean(mean_loss_eval), metrics, y, y_hat
        else:
            return np.mean(mean_loss_eval), metrics
    
    def save(self):
        """
        Public function : save log
        """
        torch.save(self.log, self.path_checkpoint+'.pt')
        return

    def compress(self):
        """
        Public function : compress log using lzma
        """
        file_path = self.path_checkpoint
        with open(file_path+'.pt', 'rb') as file:
            file_data = file.read()
        
        # Compress the file data using LZMA
        compressed_data = lzma.compress(file_data)
        
        # Save the compressed data to a new file in the same directory, change to .xz extension
        with open(file_path+'.xz', 'wb') as compressed_file:
            compressed_file.write(compressed_data)

        # Delete not compressed .pt file
        os.remove(file_path+'.pt')
        return
    
    def plot_history(self):
        """
        Public function : plot loss history
        """
        fig = plt.figure()
        plt.plot(range(self.passed_epochs), self.loss_train_history, label='Train loss')
        if self.valid_loader is not None:
            plt.plot(range(self.passed_epochs), self.loss_valid_history, label='Valid loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        if self.path_fig:
            plt.savefig(self.path_fig)
        else:
            plt.show()
        return
    
    def reduce_lr(self, new_lr):
        """
        Public function : update learning of the optimizer
        """
        for g in self.model.optimizer.param_groups:
            g['lr'] = new_lr
        return
            
    def restore_best_weights(self):
        """
        Public function : load best model state dict parameters met during training.
        """
        try:
            if self.all_gpu:
                self.model.module.load_state_dict(self.log['best_model_state_dict'])
            else:
                self.model.load_state_dict(self.log['best_model_state_dict'])
        except KeyError:
            warnings.warn('Error during loading log checkpoint state dict : no update.')
        return
    
    def __train(self):
        """
        Private function : model training loop over data loader
        """
        total_sample_train = 0
        mean_loss_train = []
        mean_accuracy_train = []
        
        for ts, labels in self.train_loader:
            self.model.train()
            # ===================variables=================== #
            ts = torch.Tensor(ts.float()).to(self.device)
            labels = torch.Tensor(labels.float()).to(self.device)
            # ===================forward===================== #
            self.optimizer.zero_grad()
            logits = self.model(ts)
            # ===================backward==================== #
            loss_train = self.train_criterion(logits.float(), labels.long())
            loss_train.backward()
            self.optimizer.step()
            # ================eval on train================== #
            total_sample_train += labels.size(0)
            _, predicted_train = torch.max(logits, 1)
            correct_train = (predicted_train.to(self.device) == labels.to(self.device)).sum().item()
            mean_loss_train.append(loss_train.item())
            mean_accuracy_train.append(correct_train)
            
        return np.mean(mean_loss_train), np.sum(mean_accuracy_train)/total_sample_train
    
    def __evaluate(self):
        """
        Private function : model evaluation loop over data loader
        """
        total_sample_valid = 0
        mean_loss_valid = []
        mean_accuracy_valid = []
        
        with torch.no_grad():
            for ts, labels in self.valid_loader:
                self.model.eval()
                # ===================variables=================== #
                ts = torch.Tensor(ts.float()).to(self.device)
                labels = torch.Tensor(labels.float()).to(self.device)
                logits = self.model(ts)
                loss_valid = self.valid_criterion(logits.float(), labels.long())
                # ================eval on test=================== #
                total_sample_valid += labels.size(0)
                _, predicted = torch.max(logits, 1)
                correct = (predicted.to(self.device) == labels.to(self.device)).sum().item()
                mean_loss_valid.append(loss_valid.item())
                mean_accuracy_valid.append(correct)

        return np.mean(mean_loss_valid), np.sum(mean_accuracy_valid)/total_sample_valid 
    



class ClassifTrainerSigmoid(object):
    def __init__(self,
                model, 
                train_loader, valid_loader=None,
                learning_rate=1e-3, weight_decay=1e-2, 
                criterion=nn.BCELoss(),
                patience_es=None, patience_rlr=None,
                device="cuda", all_gpu=False,
                valid_criterion=None,
                n_warmup_epochs=0,
                f_metrics=Classifmetrics(),
                verbose=True, plotloss=True, 
                save_fig=False, path_fig=None,
                save_checkpoint=False, path_checkpoint=None):
        """
        PyTorch Model Trainer Class for classification case
        """

        # =======================class variables======================= #
        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.f_metrics = f_metrics
        self.device = device
        self.all_gpu = all_gpu
        self.verbose = verbose
        self.plotloss = plotloss
        self.save_checkpoint = save_checkpoint
        self.path_checkpoint = path_checkpoint
        self.save_fig = save_fig
        self.path_fig = path_fig
        self.patience_rlr = patience_rlr
        self.patience_es = patience_es
        self.n_warmup_epochs = n_warmup_epochs
        self.scheduler = None
        
        self.train_criterion = criterion
        if valid_criterion is None:
            self.valid_criterion = criterion
        else:
            self.valid_criterion = valid_criterion
        
        if self.path_checkpoint is not None:
            self.path_checkpoint = path_checkpoint
        else:
            self.path_checkpoint = os.getcwd()+os.sep+'model' 
            
        if patience_rlr is not None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', 
                                                                        patience=patience_rlr, 
                                                                        verbose=self.verbose,
                                                                        eps=1e-7)
            
        #if n_warmup_epochs > 0 and self.scheduler is not None:
        #    self.scheduler = create_lr_scheduler_with_warmup(self.scheduler,
        #                                                     warmup_start_value=1e-6,
        #                                                     warmup_end_value=learning_rate,
        #                                                     warmup_duration=n_warmup_epochs)

        self.log = {}
        self.train_time = 0
        self.eval_time = 0
        self.voter_time = 0
        self.passed_epochs = 0
        self.best_loss = np.inf
        self.loss_train_history = []
        self.loss_valid_history = []
        self.accuracy_train_history = []
        self.accuracy_valid_history = []
               
        if self.patience_es is not None:
            self.early_stopping = EarlyStopper(patience=self.patience_es)

        if self.all_gpu:
            # =========== Dummy forward to intialize Lazy Module if all GPU used =========== #
            self.model.to("cpu")
            for ts, _ in train_loader:
                self.model(torch.rand(ts.shape))
                break
            # =========== Data Parrallel Module call =========== #
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
    
    def train(self, n_epochs=10):
        """
        Public function : master training loop over epochs
        """
        
        #flag_es = 0
        tmp_time = time.time()
        
        for epoch in range(n_epochs):
            # =======================one epoch======================= #
            train_loss, train_accuracy = self.__train()
            self.loss_train_history.append(train_loss)
            self.accuracy_train_history.append(train_accuracy)
            if self.valid_loader is not None:
                valid_loss, valid_accuracy = self.__evaluate()
                self.loss_valid_history.append(valid_loss)
                self.accuracy_valid_history.append(valid_accuracy)
            else:
                valid_loss = train_loss
                
            # =======================reduce lr======================= #
            if self.scheduler:
                self.scheduler.step(valid_loss)

            # ===================early stoppping=================== #
            if self.patience_es is not None:
                if self.passed_epochs > self.n_warmup_epochs: # Avoid n_warmup_epochs first epochs
                    if self.early_stopping.early_stop(valid_loss):
                        #flag_es  = 1
                        es_epoch = epoch+1
                        self.passed_epochs+=1
                        if self.verbose:
                            print('Early stopping after {} epochs !'.format(epoch+1))
                        break
        
            # =======================verbose======================= #
            if self.verbose:
                print('Epoch [{}/{}]'.format(epoch+1, n_epochs))
                print('    Train loss : {:.4f}, Train acc : {:.2f}%'
                          .format(train_loss, train_accuracy*100))
                
                if self.valid_loader is not None:
                    print('    Valid  loss : {:.4f}, Valid  acc : {:.2f}%'
                              .format(valid_loss, valid_accuracy*100))

            # =======================save log======================= #
            if valid_loss <= self.best_loss and self.passed_epochs>=self.n_warmup_epochs:
                self.best_loss = valid_loss
                self.log = {'valid_metrics': valid_accuracy if self.valid_loader is not None else train_accuracy,
                            'model_state_dict': self.model.module.state_dict() if self.device=="cuda" and self.all_gpu else self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss_train_history': self.loss_train_history,
                            'loss_valid_history': self.loss_valid_history,
                            'accuracy_train_history': self.accuracy_train_history,
                            'accuracy_valid_history': self.accuracy_valid_history,
                            'value_best_loss': self.best_loss,
                            'epoch_best_loss': self.passed_epochs,
                            'time_best_loss': round((time.time() - tmp_time), 3),
                            }
                if self.save_checkpoint:
                    self.save()
                
            self.passed_epochs+=1
                    
        self.train_time = round((time.time() - tmp_time), 3)

        if self.plotloss:
            self.plot_history()
            
        if self.save_checkpoint:
            self.log['best_model_state_dict'] = torch.load(self.path_checkpoint+'.pt')['model_state_dict']
        
        # =======================update log======================= #
        self.log['training_time'] = self.train_time
        self.log['loss_train_history'] = self.loss_train_history
        self.log['loss_valid_history'] = self.loss_valid_history
        self.log['accuracy_train_history'] = self.accuracy_train_history
        self.log['accuracy_valid_history'] = self.accuracy_valid_history
        
        #if flag_es != 0:
        #    self.log['final_epoch'] = es_epoch
        #else:
        #    self.log['final_epoch'] = n_epochs
        
        if self.save_checkpoint:
            self.save()

        return

    def save(self):
        """
        Public function : save log
        """
        torch.save(self.log, self.path_checkpoint+'.pt')
        return

    def compress(self):
        """
        Public function : compress log using lzma
        """
        file_path = self.path_checkpoint
        with open(file_path+'.pt', 'rb') as file:
            file_data = file.read()
        
        # Compress the file data using LZMA
        compressed_data = lzma.compress(file_data)
        
        # Save the compressed data to a new file in the same directory, change to .xz extension
        with open(file_path+'.xz', 'wb') as compressed_file:
            compressed_file.write(compressed_data)

        # Delete not compressed .pt file
        os.remove(file_path+'.pt')
        return
    
    def plot_history(self):
        """
        Public function : plot loss history
        """
        fig = plt.figure()
        plt.plot(range(self.passed_epochs), self.loss_train_history, label='Train loss')
        if self.valid_loader is not None:
            plt.plot(range(self.passed_epochs), self.loss_valid_history, label='Valid loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        if self.path_fig:
            plt.savefig(self.path_fig)
        else:
            plt.show()
        return
    
    def reduce_lr(self, new_lr):
        """
        Public function : update learning of the optimizer
        """
        for g in self.model.optimizer.param_groups:
            g['lr'] = new_lr
        return
            
    def restore_best_weights(self):
        """
        Public function : load best model state dict parameters met during training.
        """
        try:
            if self.all_gpu:
                self.model.module.load_state_dict(self.log['best_model_state_dict'])
            else:
                self.model.load_state_dict(self.log['best_model_state_dict'])
            print('Restored best model met during training.')
        except KeyError:
            print('Error during loading log checkpoint state dict : no update.')
        return
    
    def evaluate(self, test_loader, mask='test_metrics', return_output=False):
        """
        Public function : model evaluation on test dataset
        """
        tmp_time = time.time()
        mean_loss_eval = []
        y = np.array([])
        y_hat = np.array([])

        self.model.eval()
        with torch.no_grad():
            for ts, labels in test_loader:
                # ===================variables=================== #
                ts = torch.Tensor(ts.float()).to(self.device)
                labels = torch.unsqueeze(torch.Tensor(labels.float()), dim=1).to(self.device)
                # ===================forward===================== #
                pred  = self.model(ts)
                loss  = self.train_criterion(pred, labels)
                # =================concatenate=================== #
                pred = torch.round(pred)
                mean_loss_eval.append(loss.item())
                y_hat = np.concatenate((y_hat, pred.detach().cpu().numpy())) if y_hat.size else pred.detach().cpu().numpy()
                y = np.concatenate((y, torch.flatten(labels).detach().cpu().numpy())) if y.size else torch.flatten(labels).detach().cpu().numpy()
                
        metrics = self.f_metrics(y, y_hat)
        self.eval_time = round((time.time() - tmp_time), 3)
        self.log[mask+'_time'] = self.eval_time
        self.log[mask] = metrics
        
        if self.save_checkpoint:
            self.save()
        
        if return_output:
            return np.mean(mean_loss_eval), metrics, y, y_hat
        else:
            return np.mean(mean_loss_eval), metrics
    
    def __train(self):
        """
        Private function : model training loop over data loader
        """
        total_sample_train  = 0
        mean_loss_train     = []
        mean_accuracy_train = []
        
        self.model.train()

        for ts, labels in self.train_loader:
            # ===================variables=================== #
            ts     = torch.Tensor(ts.float()).to(self.device) 
            labels = torch.unsqueeze(torch.Tensor(labels.float()), dim=1).to(self.device)
            # ===================forward===================== #
            self.optimizer.zero_grad()
            pred = self.model.forward(ts)
            loss_train = self.train_criterion(pred, labels)
            # ===================backward==================== #
            loss_train.backward()
            mean_loss_train.append(loss_train.item())
            self.optimizer.step()
            # ===================accuracy==================== #
            total_sample_train += labels.size(0)
            correct_train = (torch.round(pred).to(self.device) == labels.to(self.device)).sum().item()
            mean_accuracy_train.append(correct_train)
            
        return np.mean(mean_loss_train), np.sum(mean_accuracy_train)/total_sample_train
    
    def __evaluate(self):
        """
        Private function : model evaluation loop over data loader
        """
        total_sample_valid = 0
        mean_loss_valid = []
        mean_accuracy_valid = []
        
        self.model.eval()
        with torch.no_grad():
            for ts, labels in self.valid_loader:
                # ===================variables=================== #
                ts     = torch.Tensor(ts.float()).to(self.device)
                labels = torch.unsqueeze(torch.Tensor(labels.float()), dim=1).to(self.device)
                # ===================forward=================== #
                pred = self.model(ts)
                loss_valid = self.valid_criterion(pred, labels)
                # ================eval on test=================== #
                total_sample_valid += labels.size(0)
                correct = (torch.round(pred).to(self.device) == labels.to(self.device)).sum().item()
                mean_loss_valid.append(loss_valid.item())
                mean_accuracy_valid.append(correct)

        return np.mean(mean_loss_valid), np.sum(mean_accuracy_valid)/total_sample_valid 



class SeqToSeqTrainer():
    def __init__(self,
                 model, 
                 train_loader, valid_loader=None,
                 learning_rate=1e-3, weight_decay=1e-2,
                 criterion=nn.MSELoss(),
                 consumption_pred=True,
                 patience_es=None, patience_rlr=None,
                 device="cuda", all_gpu=False,
                 valid_criterion=None,
                 training_in_model=False, loss_in_model=False, apply_sigmoid=False,
                 f_metrics=NILMmetrics(),
                 n_warmup_epochs=0,
                 verbose=True, plotloss=True, 
                 save_fig=False, path_fig=None,
                 save_checkpoint=False, path_checkpoint=None):
        """
        PyTorch Model Trainer Class for SeqToSeq NILM (per timestamps estimation)

        Can be either: classification, values in [0,1] or energy power estimation for each timesteps
        """
        
        # =======================class variables======================= #
        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.all_gpu = all_gpu
        self.verbose = verbose
        self.plotloss = plotloss
        self.save_checkpoint = save_checkpoint
        self.path_checkpoint = path_checkpoint
        self.save_fig = save_fig
        self.path_fig = path_fig
        self.patience_rlr = patience_rlr
        self.patience_es = patience_es
        self.n_warmup_epochs = n_warmup_epochs
        self.consumption_pred = consumption_pred
        self.f_metrics = f_metrics
        self.loss_in_model = loss_in_model
        self.training_in_model = training_in_model
        self.apply_sigmoid = apply_sigmoid
        self.apply_sigmoid_train = apply_sigmoid

        if self.training_in_model:
            assert hasattr(self.model, 'train_one_epoch')
        
        self.train_criterion = criterion
        self.valid_criterion = criterion if valid_criterion is None else valid_criterion

        if (self.apply_sigmoid and isinstance(criterion, nn.BCEWithLogitsLoss)) and (not self.training_in_model):
            warnings.warn(f"BCELossWithLogitsLoss provided for training and apply_sigmoid is True: set apply_sigmoid_train=False.")
            self.apply_sigmoid_train = False

        if self.apply_sigmoid and isinstance(self.valid_criterion, nn.BCEWithLogitsLoss):
            warnings.warn(f"BCELossWithLogitsLoss provided for validation and apply_sigmoid is True: set valid_criterion as BCELoss  to avoid re-applying sigmoid.")
            self.valid_criterion = nn.BCELoss()

        if self.consumption_pred and self.apply_sigmoid:
            warnings.warn(f"Inconsistent provided arguments: consumption_pred=True and apply_sigmoid=True. Set apply_sigmoid=False to perform training.")
            self.apply_sigmoid = False
        
        if self.path_checkpoint is not None:
            self.path_checkpoint = path_checkpoint
        else:
            self.path_checkpoint = os.getcwd()+os.sep+'model'
            
        if self.patience_rlr is not None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', 
                                                                        patience=self.patience_rlr, 
                                                                        verbose=self.verbose,
                                                                        eps=1e-7)
  
        self.log = {}
        self.train_time = 0
        self.eval_time = 0
        self.voter_time = 0
        self.passed_epochs = 0
        self.best_loss = np.inf
        self.loss_train_history = []
        self.loss_valid_history = []
        self.accuracy_train_history = []
        self.accuracy_valid_history = []
               
        if self.patience_es is not None:
            self.early_stopping = EarlyStopper(patience=self.patience_es)

        if self.all_gpu:
            # =========== Dummy forward to intialize Lazy Module =========== #
            self.model.to("cpu")
            for instances in train_loader:
                self.model(torch.rand(instances[0].shape))
                break
            # =========== Data Parrallel Module call =========== #
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
    
    def train(self, n_epochs=10):
        """
        Public function : master training loop over epochs
        """
        
        #flag_es = 0
        tmp_time = time.time()
        
        for epoch in range(n_epochs):
            # =======================one epoch======================= #
            if self.training_in_model:
                self.model.train()
                if self.all_gpu:
                    train_loss = self.model.module.train_one_epoch(loader=self.train_loader, optimizer=self.optimizer, device=self.device)
                else:
                    train_loss = self.model.train_one_epoch(loader=self.train_loader, optimizer=self.optimizer, device=self.device)
            else:
                train_loss = self.__train()
            self.loss_train_history.append(train_loss)
            if self.valid_loader is not None:
                valid_loss = self.__evaluate()
                self.loss_valid_history.append(valid_loss)
            else:
                valid_loss = train_loss
                
            # =======================reduce lr======================= #
            if self.patience_rlr:
                self.scheduler.step(valid_loss)

            # ===================early stoppping=================== #
            if self.patience_es is not None:
                if self.passed_epochs > self.n_warmup_epochs: # Avoid n_warmup_epochs first epochs
                    if self.early_stopping.early_stop(valid_loss):
                        #flag_es  = 1
                        es_epoch = epoch+1
                        self.passed_epochs+=1
                        if self.verbose:
                            print('Early stopping after {} epochs !'.format(epoch+1))
                        break
        
            # =======================verbose======================= #
            if self.verbose:
                print('Epoch [{}/{}]'.format(epoch+1, n_epochs))
                print('    Train loss : {:.4f}'
                          .format(train_loss))
                
                if self.valid_loader is not None:
                    print('    Valid  loss : {:.4f}'
                              .format(valid_loss))

            # =======================save log======================= #
            if valid_loss <= self.best_loss and self.passed_epochs>=self.n_warmup_epochs:
                self.best_loss = valid_loss
                self.log = {'model_state_dict': self.model.module.state_dict() if self.device=="cuda" and self.all_gpu else self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss_train_history': self.loss_train_history,
                            'loss_valid_history': self.loss_valid_history,
                            'value_best_loss': self.best_loss,
                            'epoch_best_loss': self.passed_epochs,
                            'time_best_loss': round((time.time() - tmp_time), 3),
                            }
                if self.save_checkpoint:
                    self.save()
                
            self.passed_epochs+=1
                    
        self.train_time = round((time.time() - tmp_time), 3)

        if self.plotloss:
            self.plot_history()

        if self.save_checkpoint:
            self.log['best_model_state_dict'] = torch.load(self.path_checkpoint+'.pt')['model_state_dict']
        
        # =======================update log======================= #
        self.log['training_time'] = self.train_time
        self.log['loss_train_history'] = self.loss_train_history
        self.log['loss_valid_history'] = self.loss_valid_history
        
        if self.save_checkpoint:
            self.save()
        return
    
    def evaluate(self, test_loader, scaler=None, factor_scaling=1, 
                 save_outputs=False, mask='test_metrics',
                 appliance_mean_on_power=None,
                 threshold_small_values=0, threshold_activation=None):
        """
        Public function : model evaluation on test dataset
        """
        loss_valid = 0
        
        y           = np.array([])
        y_hat       = np.array([])
        y_win       = np.array([])
        y_hat_win   = np.array([])
        y_state     = np.array([])
        y_hat_state = np.array([])
        
        start_time = time.time()
        self.model.eval()
        with torch.no_grad():
            for ts_agg, appl, state in test_loader:
                # ===================variables=================== #
                ts_agg = torch.Tensor(ts_agg.float()).to(self.device)

                if self.consumption_pred:
                    target = torch.Tensor(appl.float()).to(self.device)
                else:
                    target = torch.Tensor(state.float()).to(self.device)
                
                # ===================forward and loss===================== #
                if self.loss_in_model:
                    pred, _ = self.model(ts_agg, target)
                else:
                    pred = self.model(ts_agg)

                if self.apply_sigmoid:
                        pred = nn.Sigmoid()(pred)

                loss = self.valid_criterion(pred, target)
                loss_valid += loss.item()

                # ===================Evaluate using provided metrics===================== #
                if self.consumption_pred:
                    if scaler is not None:
                        target = scaler.inverse_transform_appliance(target)
                        pred   = scaler.inverse_transform_appliance(pred)
                    else:
                        target = target * factor_scaling
                        pred   = pred * factor_scaling

                    pred[pred<threshold_small_values] = 0 

                    target_win = target.sum(dim=-1)
                    pred_win   = pred.sum(dim=-1)

                    y         = np.concatenate((y, torch.flatten(target).detach().cpu().numpy())) if y.size else torch.flatten(target).detach().cpu().numpy()
                    y_hat     = np.concatenate((y_hat, torch.flatten(pred).detach().cpu().numpy())) if y_hat.size else torch.flatten(pred).detach().cpu().numpy()
                    y_win     = np.concatenate((y_win, torch.flatten(target_win).detach().cpu().numpy())) if y_win.size else torch.flatten(target_win).detach().cpu().numpy()
                    y_hat_win = np.concatenate((y_hat_win, torch.flatten(pred_win).detach().cpu().numpy())) if y_hat_win.size else torch.flatten(pred_win).detach().cpu().numpy()
                    y_state   = np.concatenate((y_state, state.flatten())) if y_state.size else state.flatten()

                else:
                    y_state     = np.concatenate((y_state, state.flatten())) if y_state.size else state.flatten()
                    y_hat_state = np.concatenate((y_hat_state, torch.flatten(pred).detach().cpu().numpy())) if y_hat_state.size else torch.flatten(pred).detach().cpu().numpy()
                
                    if appliance_mean_on_power is not None:
                        appl = scaler.inverse_transform_appliance(appl)
                        y = np.concatenate((y, appl.flatten())) if y.size else appl.flatten()



        loss_valid = loss_valid / len(self.valid_loader)
        
        if self.consumption_pred:
            metrics_timestamp = self.f_metrics(y, y_hat, y_state, y_hat_state=(y_hat > (threshold_activation if threshold_activation is not None else threshold_small_values)).astype(dtype=int))
            metrics_win       = self.f_metrics(y_win, y_hat_win)

            self.log[mask+'_timestamp'] = metrics_timestamp
            self.log[mask+'_win']       = metrics_win
        else:
            if appliance_mean_on_power is not None:
                y_hat_state = np.nan_to_num(y_hat_state.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
                y_hat       = np.round(y_hat_state) * appliance_mean_on_power
                metrics = self.f_metrics(y=y, y_hat=y_hat, y_state=y_state, y_hat_state=y_hat_state)
            else:
                metrics = self.f_metrics(y=None, y_hat=None, y_state=y_state, y_hat_state=y_hat_state)
            self.log[mask+'_timestamp'] = metrics

        self.eval_time = round((time.time() - start_time), 3)

        self.log[mask+'_time'] = self.eval_time

        if save_outputs:
            self.log[mask+'_yhat'] = y_hat

            if y_hat_win.size:
                self.log[mask+'_yhat_win'] = y_hat
        
        if self.save_checkpoint:
            self.save()
        
        return np.mean(loss_valid)
    
    def save(self):
        """
        Public function : save log
        """
        torch.save(self.log, self.path_checkpoint+'.pt')
        return

    def compress(self):
        """
        Public function : compress log using lzma
        """
        file_path = self.path_checkpoint
        with open(file_path+'.pt', 'rb') as file:
            file_data = file.read()
        
        # Compress the file data using LZMA
        compressed_data = lzma.compress(file_data)
        
        # Save the compressed data to a new file in the same directory, change to .xz extension
        with open(file_path+'.xz', 'wb') as compressed_file:
            compressed_file.write(compressed_data)

        # Delete not compressed .pt file
        os.remove(file_path+'.pt')
        return
    
    def delete(self):
        """
        Public function : delete model log
        """
        file_path = self.path_checkpoint
        
        if os.path.exists(file_path+'.xz'):
            os.remove(file_path+'.xz')
        elif os.path.exists(file_path+'.pt'):
            os.remove(file_path+'.pt')
        else:
            warnings.warn(f"Model's log doesn't exist at provide path, not deleted. Check path and file extension (supported .xz and .pt). Provide path: {file_path}")
        
        return    

    def plot_history(self):
        """
        Public function : plot loss history
        """
        fig = plt.figure()
        plt.plot(range(self.passed_epochs), self.loss_train_history, label='Train loss')
        if self.valid_loader is not None:
            plt.plot(range(self.passed_epochs), self.loss_valid_history, label='Valid loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        if self.path_fig:
            plt.savefig(self.path_fig)
        else:
            plt.show()
        return
    
    def reduce_lr(self, new_lr):
        """
        Public function : update learning of the optimizer
        """
        for g in self.model.optimizer.param_groups:
            g['lr'] = new_lr
            
        return
            
    def restore_best_weights(self):
        """
        Public function : load best model state dict parameters met during training.
        """
        try:
            if self.all_gpu:
                self.model.module.load_state_dict(self.log['best_model_state_dict'])
            else:
                self.model.load_state_dict(self.log['best_model_state_dict'])
            print('Restored best model met during training.')
        except KeyError:
            print('Error during loading log checkpoint state dict : no update.')
        return
    
    def __train(self):
        """
        Private function : model training loop over data loader
        """
        loss_train = 0
        self.model.train()
        
        for ts_agg, appl, states in self.train_loader:
            # ===================variables=================== #
            ts_agg = torch.Tensor(ts_agg.float()).to(self.device)    

            if self.consumption_pred:
                target = torch.Tensor(appl.float()).to(self.device)
            else:
                target = torch.Tensor(states.float()).to(self.device)

            # ===================forward===================== #
            self.optimizer.zero_grad()

            if self.loss_in_model:
                pred, loss = self.model(ts_agg, target)
            else:
                pred = self.model(ts_agg)

                if self.apply_sigmoid_train:
                    pred = nn.Sigmoid()(self.model(ts_agg))

                loss = self.train_criterion(pred, target)

            # ===================backward==================== #
            loss_train += loss.item()
            loss.backward()
            self.optimizer.step()

        loss_train = loss_train / len(self.train_loader)
            
        return loss_train
    
    def __evaluate(self):
        """
        Private function : model evaluation loop over data loader
        """
        loss_valid = 0
        self.model.eval()

        with torch.no_grad():
            for ts_agg, appl, states in self.valid_loader:
                # ===================variables=================== #
                    
                ts_agg = torch.Tensor(ts_agg.float()).to(self.device)    
                if self.consumption_pred:
                    target = torch.Tensor(appl.float()).to(self.device)
                else:
                    target = torch.Tensor(states.float()).to(self.device)

                # ===================forward=================== #
                if self.loss_in_model:
                    pred, loss = self.model(ts_agg, target)
                else:
                    pred = self.model(ts_agg)

                    if self.apply_sigmoid:
                        pred = nn.Sigmoid()(self.model(ts_agg))

                    loss = self.valid_criterion(pred, target)
                
                loss_valid += loss.item()
                
        loss_valid = loss_valid / len(self.valid_loader)

        return loss_valid

        
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
