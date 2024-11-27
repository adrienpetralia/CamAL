import torch
import torch.nn as nn
import torch.nn.functional as F

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
        

class ResUnit(nn.Module):       
    def __init__(self, c_in, c_out, k=3, dilation=1, stride=1, bias=True):
        super(ResUnit,self).__init__()
        
        self.layers = nn.Sequential(nn.BatchNorm1d(c_in),
                                    nn.ReLU(),
                                    Conv1dSamePadding(in_channels=c_in, out_channels=c_out,
                                                      kernel_size=k, dilation=dilation, stride=stride, bias=bias),
                                    nn.BatchNorm1d(c_out),
                                    nn.ReLU(),
                                    Conv1dSamePadding(in_channels=c_out, out_channels=c_out,
                                                      kernel_size=k, dilation=dilation, stride=stride, bias=bias)
                                    )
        if c_in > 1 and c_in!=c_out:
            self.match_residual=True
            self.conv = nn.Conv1d(in_channels=c_in, out_channels=c_out, kernel_size=1)
        else:
            self.match_residual=False
            
    def forward(self,x):
        if self.match_residual:
            x_bottleneck = self.conv(x)
            x = self.layers(x)
            
            return torch.add(x_bottleneck, x)
        else:
            return torch.add(x, self.layers(x))
        

class DilatedResGroup(nn.Module):  
    def __init__(self, c_in=30, c_inner=30, c_out=30, kernel_size=3, dilation_list=[1, 2, 3, 4]):
        super(DilatedResGroup,self).__init__()
 
        layers = []
        for i, dilation in enumerate(dilation_list):
            if i==0:
                layers.append(ResUnit(c_in, c_inner, k=kernel_size, dilation=dilation))
            elif i==(len(dilation_list)-1):
                layers.append(ResUnit(c_inner, c_out, k=kernel_size, dilation=dilation))
            else:
                layers.append(ResUnit(c_inner, c_inner, k=kernel_size, dilation=dilation))
        self.layers = torch.nn.Sequential(*layers)
      
        if c_in > 1 and c_in!=c_out:
            self.match_residual=True
            self.conv = nn.Conv1d(in_channels=c_in, out_channels=c_out, kernel_size=1)
        else:
            self.match_residual=False
            
    def forward(self,x):
        if self.match_residual:
            x_bottleneck = self.conv(x)
            x = self.layers(x)
            
            return torch.add(x_bottleneck, x)
        else:
            return torch.add(x, self.layers(x))
        

    

class AttentionBranch(nn.Module):  
    def __init__(self, c_in=50, c_inner=50, c_out=50, kernel_size=3):
        super(AttentionBranch,self).__init__()
        
        self.ResUnit1 = ResUnit(c_in, c_inner, k=kernel_size)
        self.ResUnit2 = ResUnit(c_inner, c_inner, k=kernel_size)
        self.ResUnit3 = ResUnit(c_inner, c_inner, k=kernel_size)
        self.ResUnit4 = ResUnit(c_inner, c_inner, k=kernel_size)
        self.ResUnit5 = ResUnit(c_inner, c_inner, k=kernel_size)
        self.ResUnit6 = ResUnit(c_inner, c_inner, k=kernel_size)
        
        self.max_pool1 = nn.MaxPool1d(kernel_size, return_indices=True)
        self.max_pool2 = nn.MaxPool1d(kernel_size, return_indices=True)
        
        self.max_unpool1 = nn.MaxUnpool1d(kernel_size)
        self.max_unpool2 = nn.MaxUnpool1d(kernel_size)

        self.conv1_1 = nn.Conv1d(in_channels=c_inner, out_channels=c_inner, kernel_size=1)
        self.conv1_2 = nn.Conv1d(in_channels=c_inner, out_channels=c_out, kernel_size=1)
        
        self.sigmoid = nn.Sigmoid()

        if c_in > 1 and c_in!=c_inner:
            self.match_residual=True
            self.conv = nn.Conv1d(in_channels=c_in, out_channels=c_inner, kernel_size=1)
        else:
            self.match_residual=False
        
    def forward(self, x_input):

        x = self.ResUnit1(x_input)
        S1 = x.size()
        new_x, indices1 = self.max_pool1(x)
        x = self.ResUnit2(new_x)
        x = self.ResUnit3(x)
        S2 = x.size()
        x, indices2 = self.max_pool2(x)
        x = self.ResUnit4(x)
        x = self.ResUnit5(x)
        x = self.max_unpool1(x, indices2, output_size=S2)
        x = self.ResUnit6(x + new_x)
        x = self.max_unpool2(x, indices1, output_size=S1)

        if self.match_residual:
            x_input = self.conv(x_input)
        x = torch.add(x_input, x)

        x = self.conv1_1(x)
        x = self.conv1_2(x)

        x = self.sigmoid(x)
        
        return x


class DResNet(nn.Module):
    """
    DResNet Pytorch implementation as described in the original paper "Non-intrusive load disaggregation based on deep dilated residual network, Min Xia et al.".

    Inspired by ResNet-34, the architecture use 4 residual groups with a dilation parameter.
    """
    def __init__(self, window_size=128, c_in=1, kernel_size=3):
        super(DResNet, self).__init__()
        
        self.input_conv  = nn.Conv1d(in_channels=c_in, out_channels=30, kernel_size=7, padding=3)

        self.first_group  = DilatedResGroup(c_in=30, c_inner=30, c_out=30, kernel_size=kernel_size, dilation_list=[1, 1, 1])
        self.second_group = DilatedResGroup(c_in=30, c_inner=40, c_out=40, kernel_size=kernel_size, dilation_list=[2, 2, 2, 2])
        self.third_group  = DilatedResGroup(c_in=40, c_inner=50, c_out=50, kernel_size=kernel_size, dilation_list=[3, 3])
        self.fourth_group = DilatedResGroup(c_in=50, c_inner=50, c_out=50, kernel_size=kernel_size, dilation_list=[4, 4, 4])

        self.fc = nn.Sequential(nn.Flatten(start_dim=1),
                                nn.LazyLinear(1024),
                                nn.Dropout(0.2))
        
        self.out = nn.Linear(1024, window_size)

    def forward(self, x) -> torch.Tensor:
        # Input as B, C, L (C=1, for aggregate power)
        x = self.input_conv(x)

        x = self.first_group(x)
        x = self.second_group(x)
        x = self.third_group(x)
        x = self.fourth_group(x)
        
        x = self.fc(x)        
        x = self.out(x).unsqueeze(1)

        return x
    
    
class DAResNet(nn.Module):
    """
    DAResNet Pytorch implementation as described in the original paper "Dilated residual attention network for load disaggregation".

    As DResNet, the architecture is based on ResNet34, the only difference is the use of forward attention mechanism in parrallel of the 3rd dilated convolutional group.
    """
    def __init__(self, window_size=128, c_in=1, kernel_size=3):
        super(DAResNet, self).__init__()
        
        # According to original ResNet34 architecture kernel size in the first Conv set to 7
        self.input_conv  = nn.Conv1d(in_channels=c_in, out_channels=30, kernel_size=7, padding=3)

        self.first_group  = DilatedResGroup(c_in=30, c_inner=30, c_out=30, kernel_size=kernel_size, dilation_list=[1, 1, 1])
        self.second_group = DilatedResGroup(c_in=30, c_inner=40, c_out=40, kernel_size=kernel_size, dilation_list=[2, 2, 2, 2])
        self.third_group  = DilatedResGroup(c_in=40, c_inner=50, c_out=50, kernel_size=kernel_size, dilation_list=[3, 3])
        self.fourth_group = DilatedResGroup(c_in=50, c_inner=50, c_out=50, kernel_size=kernel_size, dilation_list=[4, 4, 4])

        self.attention_branch = AttentionBranch(c_in=40, c_inner=40, c_out=50, kernel_size=kernel_size)
        
        self.fc = nn.Sequential(nn.Flatten(start_dim=1),
                                nn.LazyLinear(1024),
                                nn.Dropout(0.2))
        
        self.out = nn.Linear(1024, window_size)

    def forward(self, x) -> torch.Tensor:
        # Input as B, C, L (C=1, for aggregate power)
        x = self.input_conv(x)

        x = self.first_group(x)
        x = self.second_group(x)

        x_feature_branch   = self.third_group(x)
        x_attention_branch = self.attention_branch(x)
        x = torch.add(x_feature_branch, torch.mul(x_feature_branch, x_attention_branch))

        x = self.fourth_group(x)
        
        x = self.fc(x)
        x = self.out(x).unsqueeze(1)

        return x