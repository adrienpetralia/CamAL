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

    
# ======================= Dilated ResNet =======================#
class FCN(nn.Module):
    def __init__(self, window_size, downstreamtask='seq2seq', c_in=1):
        super().__init__()
        self.downstreamtask = downstreamtask
        
        self.convlayer1 = nn.Sequential(Conv1dSamePadding(in_channels=c_in, out_channels=30, kernel_size=10, 
                                                          dilation=1, stride=1, bias=True),
                                        nn.ReLU())
        self.convlayer2 = nn.Sequential(Conv1dSamePadding(in_channels=30, out_channels=30, kernel_size=8, 
                                                          dilation=1, stride=1, bias=True),
                                        nn.ReLU())
        self.convlayer3 = nn.Sequential(Conv1dSamePadding(in_channels=30, out_channels=40, kernel_size=6, 
                                                          dilation=1, stride=1, bias=True),
                                        nn.ReLU())
        self.convlayer4 = nn.Sequential(Conv1dSamePadding(in_channels=40, out_channels=50, kernel_size=5, 
                                                          dilation=1, stride=1, bias=True),
                                        nn.ReLU())
        self.convlayer5 = nn.Sequential(Conv1dSamePadding(in_channels=50, out_channels=50, kernel_size=5, 
                                                          dilation=1, stride=1, bias=True),
                                        nn.ReLU())
        
        self.fc = nn.Sequential(nn.Flatten(start_dim=1),
                                nn.LazyLinear(1024),
                                nn.Dropout(0.2))

        if self.downstreamtask=='seq2seq':
            self.out = nn.Linear(1024, window_size)
        else:
            self.out = nn.Linear(1024, 1)

                      
    def forward(self, x) -> torch.Tensor:
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = self.convlayer3(x)
        x = self.convlayer4(x)
        x = self.convlayer5(x)
        x = self.fc(x)
                             
        out = self.out(x)

        if self.downstreamtask=='seq2seq':
            return out.unsqueeze(1)
        else:
            return out