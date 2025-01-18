import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Convolutional Dilated Embedding Block Functions
class Conv1dSamePadding(nn.Conv1d):
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)


class Dense(nn.Module):
    def __init__(self, in_features, out_features):
        super(Dense, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


class BiGRU(nn.Module):
    def __init__(self, window_size, c_in=1, out_channels=1, dropout=0.1, return_values='power', verbose_loss=False):
        super(BiGRU, self).__init__()

        self.return_values = return_values
        self.verbose_loss = verbose_loss
        
        self.drop = nn.Dropout(dropout)

        self.conv1 = Conv1dSamePadding(in_channels=c_in, out_channels=16, kernel_size=5, dilation=1, stride=1, bias=True)
        self.conv2 = Conv1dSamePadding(in_channels=16, out_channels=8, kernel_size=5, dilation=1, stride=1, bias=True)

        self.gru1 = nn.GRU(8, 64, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(128, 128, batch_first=True, bidirectional=True)

        self.dense = Dense(256, 64)
        self.regressor = nn.Conv1d(64, out_channels, kernel_size=1, padding=0)
        self.activation = nn.Conv1d(64, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(self.drop(x))

        x = self.gru1(self.drop(x.permute(0, 2, 1)))[0]
        x = self.gru2(self.drop(x))[0]

        x = self.drop(self.dense(self.drop(x)))

        power_logits  = self.regressor(self.drop(x.permute(0, 2, 1)))
        states_logits = self.activation(self.drop(F.relu(x.permute(0, 2, 1))))

        if self.return_values=='power':
            return power_logits
        elif self.return_values=='states':
            return states_logits
        else:
            return power_logits, states_logits
        
    def forward_loss(self, x, y_power, y_status):
        
        tmp_return_values = self.return_values
        self.return_values = 'dual'
        
        power_logits, states_logits = self.forward(x)
        
        mse_loss = nn.MSELoss()(power_logits, y_power)
        bce_loss = nn.BCEWithLogitsLoss()(states_logits, y_status)
        
        loss = mse_loss + bce_loss
        
        self.return_values = tmp_return_values
        
        return loss, mse_loss, bce_loss
        
    def train_one_epoch(self, loader, optimizer, device='cuda'):
        """
        Train for one epoch
        """
        self.train()

        total_loss = 0
        total_mse_loss = 0
        total_bce_loss = 0
        
        for seqs, labels_energy, status in loader:
            
            seqs, labels_energy, status = torch.Tensor(seqs.float()).to(device), torch.Tensor(labels_energy.float()).to(device), torch.Tensor(status.float()).to(device)
            
            optimizer.zero_grad()
            loss, mse_loss, bce_loss = self.forward_loss(seqs, labels_energy, status)
            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_bce_loss += bce_loss.item()

            loss.backward()
            optimizer.step()
                
        total_loss = total_loss / len(loader)

        if self.verbose_loss:
            print('Tot. loss:', total_loss, ' | MSE loss:', total_mse_loss / len(loader), ' | BCE loss:', total_bce_loss / len(loader))

        return total_loss
    
    def valid_one_epoch(self, loader, device='cuda'):
        """
        Train for one epoch
        """
        self.eval()
        
        total_loss = 0
        total_mse_loss = 0
        total_bce_loss = 0
        
        for seqs, labels_energy, status in loader:
            
            seqs, labels_energy, status = torch.Tensor(seqs.float()).to(device), torch.Tensor(labels_energy.float()).to(device), torch.Tensor(status.float()).to(device)
            
            loss, mse_loss, bce_loss = self.forward_loss(seqs, labels_energy, status)
            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_bce_loss += bce_loss.item()
                
        total_loss = total_loss / len(loader)

        if self.verbose_loss:
            print('Tot. loss:', total_loss, ' | MSE loss:', total_mse_loss / len(loader), ' | BCE loss:', total_bce_loss / len(loader))

        return total_loss