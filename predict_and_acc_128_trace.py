# -*- coding: utf-8 -*-
import torch
import numpy as np
from layer_segmentation_network_128_edit0622 import U_Net
from torch import nn
def predict_matrix_update(model_output, label, n):
    predict_matrix = np.zeros((n, n))

    m = model_output.shape[0] * model_output.shape[1] * model_output.shape[2]
    model_output = model_output.reshape(m,).astype(int)
    label = label.reshape(m,).astype(int)
    for i in range(m):
        # print(model_output[i], label[i])
        predict_matrix[model_output[i], label[i]] +=1

    return predict_matrix

# a = [0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0]
# b = [0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
# a = np.array(a).reshape((3, 3, 3))
# b = np.array(b).reshape((3, 3, 3))
#
# c = predict_matrix_update(a, b, 2).astype(int)
# print(c)
def accuracy_calculate(predict_matrix, mode='PA'):
    accuracy = 0
    class_number = predict_matrix.shape[0]
    if mode == 'PA':
        Denominator = 0
        Numerator = 0
        for i in range(class_number):
            Numerator += predict_matrix[i][i]
            for j in range(class_number):
                Denominator += predict_matrix[i][j]
        accuracy = Numerator/Denominator
    elif mode == 'MPA':
        ClassAccuracy = 0
        class_number_nan = 0
        for i in range(class_number):
            Denominator = 0
            Numerator = predict_matrix[i][i]
            for j in range(class_number):
                Denominator += predict_matrix[j][i]

            if Denominator != 0:
                ClassAccuracy += Numerator / Denominator
                class_number_nan += 1
                print('The ' + str(i) + 'class PA is:', Numerator / Denominator)
            else:
                print('The ' + str(i) + 'class PA is: nan')
        accuracy = ClassAccuracy/class_number_nan
    elif mode == 'MIOU':
        ClassAccuracy = 0
        for i in range(class_number):
            Denominator0 = 0
            Denominator1 = 0
            Numerator = predict_matrix[i][i]
            for j in range(class_number):
                Denominator0 += predict_matrix[i][j]
                Denominator1 += predict_matrix[j][i]
            ClassAccuracy += Numerator/(Denominator0 + Denominator1 - Numerator)
        accuracy = ClassAccuracy/class_number
    return accuracy
# d = accuracy_calculate(c, mode='PA')
# print('acc:', d)
# def predict(net_output):
#     '''(1, 9, 56, 56, 56)'''
#     net_output_array = np.zeros((128, 128, 128))
#     net_output = torch.nn.Softmax(dim=1)(net_output)
#     # check_array = np.zeros((128, 128, 128))
#     for i in range(1, 3):
#         check_array = net_output[0, i, :, :, :].cpu().detach().numpy()
#         check_array = check_array.reshape((128*128, 128))
#         net_output_array = net_output_array.reshape((128*128, 128))
#         for j in range(128*128):
#             max_value = 0
#             max_index = 0
#             for k in range(128):
#                 if check_array[j, k] > max_value:
#                     max_value = check_array[j, k]
#                     max_index = k
#             net_output_array[j, max_index] = i
#     net_output_array = net_output_array.reshape((128, 128, 128))
#
#     # _, class_no = torch.max(net_output, 1, keepdim=True)
#     # net_output = torch.squeeze(class_no).cpu().numpy()
#     # print('predict_info:', type(net_output), net_output.shape, '\n', net_output)
#     return net_output_array
# def predict(net_output):
#     '''(1, 9, 56, 56, 56)'''
#     net_output = torch.nn.Softmax(dim=1)(net_output)
#     class_pos, class_no = torch.max(net_output, 1, keepdim=True)
#     net_output_value = torch.squeeze(class_pos).cpu().detach().numpy()
#     net_output_index = torch.squeeze(class_no).cpu().detach().numpy()
#     net_output_value = net_output_value.reshape((128*128, 128))
#     net_output_index = net_output_index.reshape((128*128, 128))
#     predict_out = np.zeros((128, 128, 128))
#     predict_out = predict_out.reshape((128*128, 128))
#     for i in range(1, 3):
#         for j in range(128*128):
#             max_value = 0
#             max_index = 0
#             for k in range(128):
#                 if net_output_index[j, k] == i:
#                     if net_output_value[j, k] > max_value:
#                         max_value = net_output_value[j, k]
#                         max_index = k
#             predict_out[j, max_index] = i
#     predict_out = predict_out.reshape((128, 128, 128))
#     # print('predict_info:', type(net_output), net_output.shape, '\n', net_output)
#     return predict_out
def predict(net_output):
    '''(1, 9, 56, 56, 56)'''
    net_output = torch.nn.Softmax(dim=1)(net_output)
    class_pos, class_no = torch.max(net_output, 1, keepdim=True)
    print(class_no.size())
    net_output_value = torch.squeeze(class_pos).cpu().detach().numpy()
    net_output_index = torch.squeeze(class_no).cpu().detach().numpy()
    # print(net_output_index.shape)
    # net_output_value = net_output_value.reshape((128*128, 128))
    # net_output_index = net_output_index.reshape((128*128, 128))
    # predict_out = np.zeros((128, 128, 128))
    # predict_out = predict_out.reshape((128*128, 128))
    # for i in range(1, 3):
    #     for j in range(128*128):
    #         max_value = 0
    #         max_index = 0
    #         for k in range(128):
    #             if net_output_index[j, k] == i:
    #                 if net_output_value[j, k] > max_value:
    #                     max_value = net_output_value[j, k]
    #                     max_index = k
    #         predict_out[j, max_index] = i
    # predict_out = predict_out.reshape((128, 128, 128))
    # print('predict_info:', type(net_output), net_output.shape, '\n', net_output)
    return net_output_index
def input_process(array, batch_size=128):
    data_output = np.zeros((1, 1, batch_size, batch_size, batch_size))
    data_output[0, 0, :, :, :] = array
    return data_output
def label_process(array, batch_size=128):
    data_output = np.zeros((1, batch_size, batch_size, batch_size))
    data_output[0, :, :, :] = array
    return data_output
if __name__ == '__main__':
    # torch.cuda.empty_cache()
    predict_matrix_sum = 0
    # network = U_Net(n_classes=3)
    # network.ParameterInitialize()
    # model_path = 'saved_model0_20000(20210603).pt'
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(torch.cuda.device_count())
    # if torch.cuda.device_count() > 1:
    #     network = torch.nn.DataParallel(network)
    # network.to(device)
    # # network.eval()
    # network.module.load_state_dict(torch.load(model_path))
    # network.eval()
    torch.cuda.empty_cache()
    network = U_Net(n_classes=3)
    network.ParameterInitialize()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # cross_entropy = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.06140522704449368, 0.04453565917512728, 0.0450304998326287, 0.04824696410638789, 0.04553646050490542, 0.050033888702920776, 0.052633051752423145, 0.04605392028337025, 0.05551705458817236, 0.0471249416853091, 0.00015375768210549294, 0.054766824120764626, 0.06048873111845646, 0.043577903063834224, 0.054766824120764626, 0.05708091528079694, 0.04453565917512728, 0.0450304998326287, 0.054766824120764626, 0.04266047352564824, 0.04605392028337025] ).to(
    #     device))
    cross_entropy = nn.CrossEntropyLoss(
        weight=torch.FloatTensor([0.13095815496304367, 1.3581026003557533, 1.510939244681203]).to(device))
    print(torch.cuda.device_count())
    # Transfer model to gpu
    if torch.cuda.device_count() > 1:
        network = nn.DataParallel(network)
    network.to(device)
    # network.load_state_dict = (checkpoint)    #pytorch调用先前的模型参数
    network.eval()
    # optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)  # Adam method
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9, last_epoch=-1)
    model_path = 'saved_model0_20000(20210623).pt'
    # optimizer.load_state_dict(torch.load(join('F3','optimizer.pth')))
    # checkpoint = {'model': network.state_dict(), 'optimizer': optimizer.state_dict()}
    # model_path = 'saved_model3.pt'
    # optimizer_path = 'optimizer0_20000(20210603).pth'
    network.module.load_state_dict(torch.load(model_path))
    # for state in optimizer.state.values():
    #     for k, v in state.items():  # for k, v in d.items()  Iterate the key and value simultaneously
    #         if torch.is_tensor(v):
    #             state[k] = v.cuda()
    #
    #         pass
    #     pass
    # pass
    network.eval()
    for z in range(23, 25):
        output = 0
        # data_file = np.loadtxt(r'layer_segmentation_test_model_fold_fault/' + str(z) + '.txt')
        data_file = np.loadtxt(r'layer_segmentation_test_model_fold_fault_multi/' + str(z) + '.txt')
        # data_file_1 = np.loadtxt(r'training_model/90.txt')
        # print(data_file_1[0:50, :])
        # print('data_file shape:', data_file.shape)
        # print(data_file[0:50, :])
        # data_file = np.moveaxis(data_file, -1, 0)
        print('data_file shape:', data_file.shape)
        data = data_file[0, :].reshape((128, 128, 128))
        predict_result = np.zeros((128, 128, 128))
        label = data_file[2, :].reshape((128, 128, 128))
        print(label[0,0,0], '1')
        torch.cuda.empty_cache()
        # for i in range(3):
        #     for j in range(3):
        #         for k in range(3):
        #             input_cube = input_process(data[i*42:i*42+56, j*42:j*42+56, k*42:k*42+56])
        #             input_cube = (torch.autograd.Variable(torch.Tensor(input_cube).float())).to(device)
        #             output = network(input_cube)
        #             predict_result[i*42:i*42+56, j*42:j*42+56, k*42:k*42+56] = predict(output)
        input_cube = input_process(data)
        print('input type,size', input_cube.shape, type(input_cube))
        input_cube = (torch.autograd.Variable(torch.Tensor(input_cube).float())).to(device)
        output = network(input_cube)
        # label_cube = label_process(label)
        # label_cube = (torch.autograd.Variable(torch.Tensor(label_cube).long())).to(device)
        # loss = cross_entropy(output, label_cube)
        # print('loss:', loss)
        predict_result = predict(output)
        # np.savetxt(r'layer_segmentation_test_model_fold_fault_predict/' + str(z) + '.txt', np.vstack([data.flatten(), label.flatten(), predict_result.flatten()]))
        np.savetxt(r'layer_segmentation_test_model_fold_fault_multi_predict/' + str(z) + '.txt', np.vstack([data.flatten(), label.flatten(), predict_result.flatten()]))
        predict_matrix = predict_matrix_update(predict_result, label, 3)
        predict_matrix_sum += predict_matrix
        print(z, '!!')
        torch.cuda.empty_cache()
        print(predict_matrix_sum)
        output = 0

    acc_PA = accuracy_calculate(predict_matrix_sum, mode='PA')
    acc_MPA = accuracy_calculate(predict_matrix_sum, mode='MPA')
    acc_MIOU = accuracy_calculate(predict_matrix_sum, mode='MIOU')
    print('acc_PA is :\n', acc_PA)
    print('acc_MPA is :\n', acc_MPA)
    print('acc_MIOU is :\n', acc_MIOU)

