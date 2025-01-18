import torch
import torch.nn as nn
import torch.nn.functional as F

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

class BiLSTM_NILM(nn.Module):
    def __init__(self, window_size, downstreamtask='seq2seq', c_in=1):
        # Refer to "KELLY J, KNOTTENBELT W. Neural NILM: Deep neural networks applied to energy disaggregation[C].The 2nd ACM International Conference on Embedded Systems for Energy-Efficient Built Environments".
        '''
        Please notice that our implementation is slightly different from the original paper, since the input of the first fully connected
        layer is the concat of all the hidden states instead of the last hidden state which was the way Kelly used. And our approach will
        result in improved accuracy.
        '''
        super(BiLSTM_NILM, self).__init__()
        self.window_size = window_size
        self.downstreamtask = downstreamtask

        self.conv = Conv1dSamePadding(in_channels=c_in, out_channels=16, kernel_size=4, dilation=1, stride=1, bias=True)
        self.lstm_1 = nn.LSTM(input_size=16, hidden_size=64, batch_first=True, bidirectional=True)
        self.lstm_2 = nn.LSTM(input_size=2*64, hidden_size=128, batch_first=True, bidirectional=True)
        self.fc_1 = nn.Linear(self.window_size*128*2, 128)

        if self.downstreamtask=='seq2seq':
            self.fc_2 = nn.Linear(128, self.window_size) # Seq2Seq
        else:
            self.fc_2 = nn.Linear(128, 1) # Seq2Point
        self.act = nn.Tanh()        

    def forward(self, x):
        x = self.conv(x).permute(0,2,1)
        x, _ = self.lstm_1(x)
        x, _ = self.lstm_2(x)
        out = self.fc_2(self.act(self.fc_1(x.contiguous().view(-1, self.window_size*256))))

        if self.downstreamtask=='seq2seq':
            return out.unsqueeze(1)
        else:
            return out