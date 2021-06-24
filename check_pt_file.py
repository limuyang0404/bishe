# -*- coding: utf-8 -*-
import torch
def check_pt(file_path):
    image = torch.load(file_path)
    return image


if __name__ == '__main__':
    print('hello')
    a = check_pt('saved_model0_20000(20210616_3).pt')
    print(a)
    print(type(a))
    print(a['right_conv_4.conv_ReLU.4.running_mean'])
    print('%'*50)
    print(a['left_conv_1.conv_ReLU.0.bias', 'left_conv_1.conv_ReLU.1.bias'])