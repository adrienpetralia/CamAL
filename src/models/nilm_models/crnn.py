import torch
import torch.nn as nn

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):        
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class LinSoftmaxPooling1D(nn.Module):
    '''
    PyTorch implementation of LinSoftmaxPooling1D.
    '''
    def __init__(self, axis=0):
        super(LinSoftmaxPooling1D, self).__init__()
        self.axis = axis

    def forward(self, x):
        square = x * x
        sum_square = torch.sum(square, dim=self.axis, keepdim=True)
        sum_x = torch.sum(x, dim=self.axis, keepdim=True)
        out = sum_square / (sum_x + 1e-6)  # Added small value to prevent division by zero
        
        return out


class CRNN_block(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dropout):
        super(CRNN_block,self).__init__()

        # Conv, BN, ReLU
        self.conv = nn.Conv1d(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn   = nn.BatchNorm1d(c_out)
        self.act  = nn.ReLU()

        # self.mp   = nn.MaxPool1d(1, stride=1) Useless
        self.dropout = nn.Dropout(p=dropout)

        # Init with weight with Xavier and bias to 0
        torch.nn.init.xavier_uniform_(self.conv.weight)
        torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.dropout(self.act(self.bn(self.conv(x))))
        
        return x

class SCRNN(nn.Module):
    """
    SCRNN model Pytorch implementation proposed in:
    "Multilabel Appliance Classification With Weakly Labeled Data for Non-Intrusive Load Monitoring"

    Backbone architecture based on a Convolutional Block and a BiGRU units layer.
    Two outputs level:
        1) Frame level (classic output of Seq-To-Seq models)
        2) Bag level (detection probabilites yes/no for n_classes)
    """
    def __init__(self, 
                 c_in=1,
                 n_classes=1, 
                 n_crnn_block=4,
                 h_gru_units=128, 
                 kernel_size=5, dp_rate=0.1):
        super().__init__()

        layers = []
        for i in range(n_crnn_block):
            n_filters = 2 ** (i+5)
            layers.append(CRNN_block(c_in=c_in if i==0 else n_filters//2, c_out=n_filters, kernel_size=kernel_size, dropout=dp_rate))
        self.conv_block = torch.nn.Sequential(*layers)

        self.bigru = nn.Sequential(Transpose(1, 2),
                                   nn.GRU(input_size=n_filters, hidden_size=h_gru_units, batch_first=True, bidirectional=True))

        self.fcl_sigmoid = nn.Sequential(nn.Linear(h_gru_units*2, n_classes),
                                         nn.Sigmoid())

    def forward(self, x):
        x = self.fcl_sigmoid(self.bigru(self.conv_block(x))[0])

        return x.permute(0, 2, 1)

class CRNN(nn.Module):
    """
    CRNN model Pytorch implementation proposed in:
    "Multilabel Appliance Classification With Weakly Labeled Data for Non-Intrusive Load Monitoring"

    Backbone architecture based on a Convolutional Block and a BiGRU units layer.
    Two outputs:
        1) Frame level (classic output of Seq-To-Seq models)
        2) Bag level (detection probabilites yes/no for current windows)
    """
    def __init__(self, 
                 window_size, 
                 c_in=1,
                 n_classes=1, 
                 n_crnn_block=4,
                 h_gru_units=265, 
                 kernel_size=5, dp_rate=0.1,
                 weight=1,
                 clip_smoothing=True, return_values='frame_level'):
        super().__init__()

        assert return_values in ['frame_level', 'bag_level', 'both']
        
        self.weight = weight # Init to 1
        self.clip_smoothing = clip_smoothing # Clip smoothing used by default
        self.return_values = return_values

        layers = []
        for i in range(n_crnn_block):
            n_filters = 2 ** (i+5)
            layers.append(CRNN_block(c_in=c_in if i==0 else n_filters//2, c_out=n_filters, kernel_size=kernel_size, dropout=dp_rate))
        self.conv_block = torch.nn.Sequential(*layers)

        self.bigru = nn.Sequential(Transpose(1, 2),
                                   nn.GRU(input_size=n_filters, hidden_size=h_gru_units, batch_first=True, bidirectional=True))

        self.fcl_sigmoid = nn.Sequential(nn.Linear(h_gru_units*2, n_classes),
                                         nn.Sigmoid())
        
        self.pooling = LinSoftmaxPooling1D(axis=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        frame_level = self.fcl_sigmoid(self.bigru(self.conv_block(x))[0])
        bag_level   = self.activation(self.pooling(frame_level))

        if self.clip_smoothing:
            frame_level = torch.mul(bag_level, frame_level)

        if self.return_values=='frame_level':
            return frame_level.permute(0, 2, 1)
        elif self.return_values=='bag_level':
            return bag_level.squeeze(1)
        else:
            return frame_level.permute(0, 2, 1), bag_level.squeeze(1)
        
    def train_one_epoch(self, loader, optimizer, device='cuda'):
        """
        Train for one epoch
        """
        self.train()

        tmp_return_values  = self.return_values
        self.return_values = 'both'

        total_loss = 0
        
        for instances in loader:
            optimizer.zero_grad()

            if len(instances)==2:
                # If only weak labels for training
                seqs, label_bag = instances
                seqs, label_bag = torch.Tensor(seqs.float()).to(device), torch.unsqueeze(torch.Tensor(label_bag.float()), dim=1).to(device)

                frame_level, bag_level = self.forward(seqs)
                loss = nn.BCELoss()(bag_level, label_bag)

            elif len(instances)==4:
                # If strong labels availables for training
                seqs, _, status, label_bag = instances
                seqs, status, label_bag = torch.Tensor(seqs.float()).to(device), torch.Tensor(status.float()).to(device), torch.unsqueeze(torch.Tensor(label_bag.float()), dim=1).to(device)

                frame_level, bag_level = self.forward(seqs)
                loss = nn.BCELoss()(frame_level, status) + self.weight * nn.BCELoss()(bag_level, label_bag)

            elif len(instances)==5:
                # If weak and strong labels for training
                seqs, _, status, label_bag, flag_weak = instances
                seqs, status = torch.Tensor(seqs.float()).to(device), torch.Tensor(status.float()).to(device)
                label_bag, flag_weak = torch.unsqueeze(torch.Tensor(label_bag.float()), dim=1).to(device), torch.Tensor(flag_weak.float()).unsqueeze(1).unsqueeze(1).to(device)

                frame_level, bag_level = self.forward(seqs)
                loss = nn.BCELoss(weight=flag_weak)(frame_level, status) + self.weight * nn.BCELoss()(bag_level, label_bag)
                # loss = int(flag_weak) * nn.BCELoss()(frame_level, status) + self.weight * nn.BCELoss()(bag_level, label_bag)

            else:
                raise ValueError(f'Invalid len {len(instances)} of tuple returned by loader - i.e. Tuple len not match 2: only weak labels for training, 4: same instances strong and weak labels for training or 5: strong and weak labels from different instances for training.')

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
                
        total_loss = total_loss / len(loader)

        self.return_values = tmp_return_values

        return total_loss
    