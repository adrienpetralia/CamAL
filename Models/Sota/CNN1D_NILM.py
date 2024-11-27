import torch
import torch.nn.functional as F
from torch import nn

class CNN1D(nn.Module):
    """
    Baseline 1D CNN implementation
    """

    def __init__(self, 
                 window_size=128, 
                 quantiles=[0.5], # [0.0025,0.1, 0.5, 0.9, 0.975]
                 c_in=1, 
                 num_classes=1, 
                 dropout=0.1, 
                 pooling_size=16,
                 return_values='power',
                 verbose_loss=False
                ):
        """
        Constructor for baseline CNN1D

        Args:
            window_size (int) : window size of the input
            quantiles (int) : number of quantiles used
            in_channels (int) : number of input channels
            dropout (float) : Dropout probability
            num_classes (int) : number of output classes / appliances
            pooling_size (int) : size of global average pooling filter
            return_values (str) : 'power', 'states' or 'both' -> return if tuple 
            verbose loss
        Returns:
            model (CNN1D) : CNN1D model object
        """
        super(CNN1D, self).__init__()

        self.num_classes = num_classes
        self.window_size = window_size
        self.num_quantiles = len(quantiles)
        self.quantiles = torch.Tensor(quantiles)
        self.prob_dropout = dropout
        self.pooling_size = pooling_size
        self.return_values = return_values
        self.verbose_loss = verbose_loss

        self.conv1 = nn.Conv1d(c_in, 16, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv1d(16, 32, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv1d(32, 64, 5, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.prelu3 = nn.PReLU()
        self.conv4 = nn.Conv1d(64, 128, 5, stride=2, padding=1)

        self.dropout = nn.Dropout(self.prob_dropout)
        self.adapool = nn.AdaptiveAvgPool1d(self.pooling_size)

        self.mlp1 = nn.Linear(128 * 16, 1024)
        self.prelu4 = nn.PReLU()
        self.mlp2 = nn.Linear(1024, self.num_classes * self.window_size)
        self.mlp3 = nn.Linear(1024, self.num_classes * self.window_size * self.num_quantiles)

        nn.utils.weight_norm(self.conv1)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.utils.weight_norm(self.conv2)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.utils.weight_norm(self.conv3)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.utils.weight_norm(self.conv4)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.utils.weight_norm(self.mlp1)
        nn.init.xavier_uniform_(self.mlp1.weight)
        nn.utils.weight_norm(self.mlp2)
        nn.init.xavier_uniform_(self.mlp2.weight)
        nn.utils.weight_norm(self.mlp3)
        nn.init.xavier_uniform_(self.mlp3.weight)

        self.mlp2.bias.data.fill_(0)
        self.mlp3.bias.data.fill_(0)

    def forward(self, x):

        x = self.prelu1(self.bn1(self.conv1(x)))
        x = self.prelu2(self.bn2(self.conv2(x)))
        x = self.prelu3(self.bn3(self.conv3(x)))
        x = self.dropout(self.conv4(x))

        x = self.adapool(x).reshape(x.size(0), -1)
        x = self.dropout(self.prelu4(self.mlp1(x)))

        states_logits = self.mlp2(x).reshape(x.size(0), self.window_size, self.num_classes).permute(0, 2, 1)
        power_logits = self.mlp3(x).reshape(x.size(0), self.window_size, self.num_quantiles, self.num_classes).permute(0, 3, 2, 1)

        if self.num_quantiles > 1:
            power_logits = power_logits[:, :, self.num_quantiles//2, :]
        else:
            power_logits = torch.squeeze(power_logits, dim=2)
            
        if self.return_values=='power':
            return power_logits
        elif self.return_values=='states':
            return states_logits
        else:
            return power_logits, states_logits
    
    def forward_loss(self, x, y_power, y_status):
        
        x = self.prelu1(self.bn1(self.conv1(x)))
        x = self.prelu2(self.bn2(self.conv2(x)))
        x = self.prelu3(self.bn3(self.conv3(x)))
        x = self.dropout(self.conv4(x))

        x = self.adapool(x).reshape(x.size(0), -1)
        x = self.dropout(self.prelu4(self.mlp1(x)))

        states_logits = self.mlp2(x).reshape(x.size(0), self.window_size, self.num_classes).permute(0, 2, 1)
        power_logits = self.mlp3(x).reshape(x.size(0), self.window_size, self.num_quantiles, self.num_classes).permute(0, 3, 2, 1)
        
        q_loss = self.quantile_regression_loss(power_logits.permute(0, 3, 2, 1), y_power.permute(0, 2, 1))
        bce_loss = nn.BCEWithLogitsLoss()(states_logits, y_status)
        
        return q_loss + bce_loss, q_loss, bce_loss
    
    def quantile_regression_loss(self, inputs, targets):
        """
        Function that computes the quantile regression loss

        Arguments:
            y_hat (torch.Tensor) : Shape (B x T x N x M) model regression predictions
            y (torch.Tensor) : Shape (B x T x M) ground truth targets
        Returns:
            loss (float): value of quantile regression loss
        """
        targets = targets.unsqueeze(2).expand_as(inputs)
        quantiles = self.quantiles.to(targets.device)
        error = (targets - inputs).permute(0, 1, 3, 2)
        loss = torch.max(quantiles*error, (quantiles-1)*error)
        
        return loss.mean()
        
    def train_one_epoch(self, loader, optimizer, device='cuda'):
        """
        Train for one epoch
        """
        self.train()
        
        total_q_loss = 0
        total_bce_loss = 0
        total_loss = 0
        
        for seqs, labels_energy, status in loader:
            
            seqs = torch.Tensor(seqs.float()).to(device)
            labels_energy = torch.Tensor(labels_energy.float()).to(device)
            status = torch.Tensor(status.float()).to(device)
            
            optimizer.zero_grad()
            loss, q_loss, bce_loss = self.forward_loss(seqs, labels_energy, status)
            total_q_loss += q_loss.item()
            total_bce_loss += bce_loss.item()
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
                
        total_loss = total_loss / len(loader)
        
        if self.verbose_loss:
            print('Tot. loss:', total_loss, ' | Quantiles loss:', total_q_loss / len(loader), ' | BCE loss:', total_bce_loss / len(loader))

        return total_loss
    
    def valid_one_epoch(self, loader, device='cuda'):
        """
        Train for one epoch
        """
        self.eval()

        total_q_loss = 0
        total_bce_loss = 0
        total_loss = 0
        
        for seqs, labels_energy, status in loader:
            
            seqs = torch.Tensor(seqs.float()).to(device)
            labels_energy = torch.Tensor(labels_energy.float()).to(device)
            status = torch.Tensor(status.float()).to(device)
            
            loss, q_loss, bce_loss = self.forward_loss(seqs, labels_energy, status)
            total_q_loss += q_loss.item()
            total_bce_loss += bce_loss.item()
            total_loss += loss.item()
                
        total_loss = total_loss / len(loader)

        if self.verbose_loss:
            print('Tot. loss:', total_loss, ' | Quantiles loss:', total_q_loss / len(loader), ' | BCE loss:', total_bce_loss / len(loader))

        return total_loss