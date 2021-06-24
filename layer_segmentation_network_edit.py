import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
def GaussianInitialize(layer):
    layer_size = list(layer.weight.size())
    layer_point = 1
    for i in range(len(layer_size)):
        layer_point = layer_point * layer_size[i]
    initialize = np.random.normal(0, (2/layer_point)**0.5, layer_point)
    #mean 0 variance 2/layer_point
    initialize = initialize.reshape(layer_size)
    layer.weight = nn.Parameter(torch.Tensor(initialize))
    return
class Conv3d_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv3d_block, self).__init__()
        conv_relu = []
        conv_relu.append(nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.BatchNorm3d(out_channels))
        conv_relu.append(nn.ReLU())
        conv_relu.append(nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.BatchNorm3d(out_channels))
        conv_relu.append(nn.ReLU())
        self.conv_ReLU = nn.Sequential(*conv_relu)
    pass
    def forward(self, x):
        out = self.conv_ReLU(x)
        return out
    pass
class U_Net(nn.Module):
    def __init__(self,n_classes=2):
        super(U_Net,self).__init__()
        self.left_conv_1 = Conv3d_block(in_channels=1, out_channels=16)
        self.pool_1 = nn.MaxPool3d(2, 2)                       #（64, 64, 64）
        self.drop1 = nn.Dropout(p=0.1)
        self.left_conv_2 = Conv3d_block(in_channels=16, out_channels=32)
        self.pool_2 = nn.MaxPool3d(2, 2)                        #（32, 32, 32）
        self.drop2 = nn.Dropout(p=0.1)
        self.left_conv_3 = Conv3d_block(in_channels=32, out_channels=64)
        self.pool_3 = nn.MaxPool3d(2, 2)                        #（16, 16, 16）
        self.drop3 = nn.Dropout(p=0.1)
        self.left_conv_4 = Conv3d_block(in_channels=64, out_channels=512)
        self.deconv_1 = nn.ConvTranspose3d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.right_conv_1 = Conv3d_block(in_channels=576, out_channels=64)
        self.deconv_2 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.right_conv_2 = Conv3d_block(in_channels=96, out_channels=32)
        self.deconv_3 = nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.right_conv_3 = Conv3d_block(in_channels=48, out_channels=16)
        self.right_conv_4 = Conv3d_block(in_channels=16, out_channels=n_classes)
    #Is called to compute network output
    def ParameterInitialize(self):
        GaussianInitialize(self.left_conv_1.conv_ReLU[0])
        GaussianInitialize(self.left_conv_1.conv_ReLU[3])
        GaussianInitialize(self.left_conv_2.conv_ReLU[0])
        GaussianInitialize(self.left_conv_2.conv_ReLU[3])
        GaussianInitialize(self.left_conv_3.conv_ReLU[0])
        GaussianInitialize(self.left_conv_3.conv_ReLU[3])
        GaussianInitialize(self.left_conv_4.conv_ReLU[0])
        GaussianInitialize(self.left_conv_4.conv_ReLU[3])
        GaussianInitialize(self.right_conv_1.conv_ReLU[0])
        GaussianInitialize(self.right_conv_1.conv_ReLU[3])
        GaussianInitialize(self.right_conv_2.conv_ReLU[0])
        GaussianInitialize(self.right_conv_2.conv_ReLU[3])
        GaussianInitialize(self.right_conv_4.conv_ReLU[0])
        GaussianInitialize(self.right_conv_4.conv_ReLU[3])
        print('ALL convolutional layer have been initialized')
    def forward(self,x):
        feature_1 = self.left_conv_1(x)
        # print('feature1:\n', type(feature_1), feature_1.size())
        feature_1_pool = self.pool_1(feature_1)
        drop_1 = self.drop1(feature_1_pool)
        feature_2 = self.left_conv_2(drop_1)
        feature_2_pool = self.pool_2(feature_2)
        drop_2 = self.drop2(feature_2_pool)
        feature_3 = self.left_conv_3(drop_2)
        feature_3_pool = self.pool_3(feature_3)
        drop_3 = self.drop3(feature_3_pool)
        feature_4 = self.left_conv_4(drop_3)
        de_feature_1 = self.deconv_1(feature_4)
        temp1 = torch.cat((feature_3, de_feature_1), dim=1)
        de_feature_1_conv = self.right_conv_1(temp1)
        de_feature_2 = self.deconv_2(de_feature_1_conv)
        temp2 = torch.cat((feature_2, de_feature_2), dim=1)
        de_feature_2_conv = self.right_conv_2(temp2)
        de_feature_3 = self.deconv_3(de_feature_2_conv)
        temp3 = torch.cat((feature_1, de_feature_3), dim=1)
        de_feature_3_conv = self.right_conv_3(temp3)
        # print('de_feature_3_cov:\n', type(de_feature_3_conv), de_feature_3_conv.size())
        output = self.right_conv_4(de_feature_3_conv)
        return output
    def classify(self,x):
        print('heyheyhey, you little bitch,here is classify')
        feature_1 = self.left_conv_1(x)
        feature_1_pool = self.pool_1(feature_1)
        drop_1 = self.drop1(feature_1_pool)
        feature_2 = self.left_conv_2(drop_1)
        feature_2_pool = self.pool_2(feature_2)
        drop_2 = self.drop2(feature_2_pool)
        feature_3 = self.left_conv_3(drop_2)
        feature_3_pool = self.pool_3(feature_3)
        drop_3 = self.drop3(feature_3_pool)
        feature_4 = self.left_conv_4(drop_3)
        de_feature_1 = self.deconv_1(feature_4)
        temp1 = torch.cat((feature_3, de_feature_1), dim=1)
        de_feature_1_conv = self.right_conv_1(temp1)
        de_feature_2 = self.deconv_2(de_feature_1_conv)
        temp2 = torch.cat((feature_2, de_feature_2), dim=1)
        de_feature_2_conv = self.right_conv_2(temp2)
        de_feature_3 = self.deconv_3(de_feature_2_conv)
        temp3 = torch.cat((feature_1, de_feature_3), dim=1)
        de_feature_3_conv = self.right_conv_3(temp3)
        output = self.right_conv_4(de_feature_3_conv)
        return output, feature_1, feature_2, feature_3, feature_4
if __name__ == "__main__":
    x1 = torch.rand(size=(1, 1, 128, 128, 128))
    net = U_Net(n_classes=3)
    print('net dir:\n', dir(net))
    print('The net:', net)
    print('*'*60)
    print('The state_dict:', net.state_dict())
    net.train()
    output = net(x1)
    print(type(output))
    print(output.size())

