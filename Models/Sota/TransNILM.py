import torch
import torch.nn as nn
import torch.nn.functional as F


class PFFN(nn.Module):
    def __init__(self, dim, hidden_dim, dp_rate=0., activation=nn.ReLU(), bias1=True, bias2=True):
        super(PFFN, self).__init__()
        self.layer1 = nn.Linear(dim, hidden_dim, bias=bias1)
        self.layer2 = nn.Linear(hidden_dim, dim, bias=bias2)
        self.dropout = nn.Dropout(dp_rate)
        self.activation = activation

    def forward(self, x):
        x = self.layer2(self.dropout(self.activation(self.layer1(x))))
        return x


class FeatureLearningUnit(nn.Module):
    def __init__(self, in_features, out_features, 
                 kernel_size=3, padding=1, stride=1,
                 nhead=4, ratio_ff=4, dp_rate=0.1):
        super(FeatureLearningUnit, self).__init__()
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)

        self.transformerlayer = nn.TransformerEncoderLayer(d_model=out_features, nhead=nhead, dim_feedforward=out_features*ratio_ff, 
                                                           dropout=dp_rate, activation=nn.ReLU(), batch_first=True)
        self.bn   = nn.BatchNorm1d(out_features)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.conv(x)).permute(0, 2, 1)
        x = self.transformerlayer(x).permute(0, 2, 1)

        return self.drop(self.bn(x))
    

class ResidualGRUUnit(nn.Module):
    def __init__(self, d_model, ratio_ff=4, dp_rate=0.):
        super(ResidualGRUUnit, self).__init__()
        self.gru  = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)
        self.pffn = PFFN(d_model, d_model * ratio_ff, dp_rate=dp_rate)

    def forward(self, x_input):
        x = self.gru(x_input.permute(0, 2, 1))[0]
        x = self.pffn(x)

        return torch.add(x_input, x.permute(0, 2, 1))
    

class TemporalScaling(nn.Module):
    def __init__(self, in_features=1, out_features=1, kernel_size=2):
        super(TemporalScaling, self).__init__()
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size=self.kernel_size, stride=self.kernel_size)
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=1, padding=0)
        self.bn   = nn.BatchNorm1d(out_features)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.bn(F.relu(x))
        return self.drop(F.interpolate(x, scale_factor=self.kernel_size, mode='linear', align_corners=True))
    

class Encoder(nn.Module):
    def __init__(self, in_channels, features, kernel_size=3, kernel_mp=2):
        super(Encoder, self).__init__()

        self.encoder1 = FeatureLearningUnit(in_channels, features, kernel_size=kernel_size, padding=0)
        self.pool1    = nn.MaxPool1d(kernel_size=kernel_mp, stride=kernel_mp)
        self.encoder2 = FeatureLearningUnit(features * 1, features * 2, kernel_size=kernel_size, padding=0)
        self.pool2    = nn.MaxPool1d(kernel_size=kernel_mp, stride=kernel_mp)
        self.encoder3 = FeatureLearningUnit(features * 2, features * 4, kernel_size=kernel_size, padding=0)
        self.pool3    = nn.MaxPool1d(kernel_size=kernel_mp, stride=kernel_mp)
        self.encoder4 = FeatureLearningUnit(features * 4, features * 8, kernel_size=kernel_size, padding=0)

        self.tpool1 = TemporalScaling(features*8, features*2, kernel_size=5)
        self.tpool2 = TemporalScaling(features*8, features*2, kernel_size=10)
        self.tpool3 = TemporalScaling(features*8, features*2, kernel_size=20)
        self.tpool4 = TemporalScaling(features*8, features*2, kernel_size=30)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        tp1 = self.tpool1(enc4)
        tp2 = self.tpool2(enc4)
        tp3 = self.tpool3(enc4)
        tp4 = self.tpool4(enc4)

        return torch.cat([enc4, tp1, tp2, tp3, tp4], dim=1)


class Decoder(nn.Module):
    def __init__(self, in_features, out_features, 
                 kernel_size=2, stride=2,
                 nhead=4, ratio_ff=4, dp_rate=0.1):
        super(Decoder, self).__init__()

        self.resgru1 = ResidualGRUUnit(in_features)
        self.resgru2 = ResidualGRUUnit(in_features)
        self.resgru3 = ResidualGRUUnit(in_features)

        self.convtranspose    = nn.ConvTranspose1d(in_features, out_features, kernel_size=kernel_size, stride=stride, bias=False)
        self.transformerlayer = nn.TransformerEncoderLayer(d_model=out_features, nhead=nhead, dim_feedforward=out_features*ratio_ff, 
                                                           dropout=dp_rate, activation=nn.ReLU(), batch_first=True)

    def forward(self, x):
        x = self.resgru1(x)
        x = self.resgru2(x)
        x = self.resgru3(x)

        x = self.convtranspose(x)
        x = F.relu(self.transformerlayer(x.permute(0, 2, 1)).permute(0, 2, 1))
        
        return  x

class TransNILM(nn.Module):
    """
    Implementation of TransNILM: A Transformer-based Deep Learning Model for Non-intrusive Load Monitoring (https://ieeexplore.ieee.org/document/9991439)

    Proposed model is an adaptation of TPNILM enahnced with Transformer layer and ResGRU Units in the Encoder and Decoder modules

    As TPNILM, input length need to be 510 and model output window size of length 480 due to no padding in Temporal Pooling convolutional encoder block
    """
    def __init__(self, in_channels=1, out_channels=1, features=32, stride=2):
        super(TransNILM, self).__init__()
        self.encoder     = Encoder(in_channels, features, kernel_size=3, kernel_mp=2)
        self.decoder     = Decoder(2*features*8, features, kernel_size=stride**3, stride=stride**3)
        self.output_conv = nn.Conv1d(features, out_channels, kernel_size=1, padding=0)

        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output_conv(x)

        return x
    
    def train_one_epoch(self, loader, optimizer, device='cuda'):
        """
        Train for one epoch
        """
        self.train()
        
        total_loss = 0
        
        for seqs, _, status in loader:
            assert seqs.shape[-1]==510, 'TransNILM only handle input sequence of length 510.'

            if status.shape[-1]==510:
                status = status[:, :, 15:-15] # Remove padding due to strided conv
            elif status.shape[-1]!=480:
                raise ValueError('TransNILM training: Output target states sequence need to be either of length 480 or 510.')

            seqs   = torch.Tensor(seqs.float()).to(device)
            status = torch.Tensor(status.float()).to(device)
            
            optimizer.zero_grad()
            states_logits = self.forward(seqs)
            loss = nn.BCEWithLogitsLoss()(states_logits, status)

            total_loss += loss.item()

            loss.backward()
            optimizer.step()
                
        total_loss = total_loss / len(loader)

        return total_loss