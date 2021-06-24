# -*- coding: utf-8 -*-
import torch
import numpy as np
from layer_segmentation_network import U_Net

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
def predict(net_output):
    '''(1, 9, 56, 56, 56)'''
    net_output = torch.nn.Softmax(dim=1)(net_output)
    _, class_no = torch.max(net_output, 1, keepdim=True)
    net_output = torch.squeeze(class_no).cpu().numpy()
    # print('predict_info:', type(net_output), net_output.shape, '\n', net_output)
    return net_output
def input_process(array, batch_size=56):
    data_output = np.zeros((1, 1, batch_size, batch_size, batch_size))
    data_output[0, 0, :, :, :] = array
    return data_output
if __name__ == '__main__':
    predict_matrix_sum = 0
    network = U_Net(n_classes=3)
    network.ParameterInitialize()
    model_path = 'saved_model0_20000(20210601).pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        network = torch.nn.DataParallel(network)
    network.to(device)
    # network.eval()
    network.module.load_state_dict(torch.load(model_path))
    network.eval()
    for z in range(1, 21):
        data_file = np.loadtxt(r'layer_segmentation_test_model/' + str(z) + '.txt')
        # data_file_1 = np.loadtxt(r'training_model/90.txt')
        # print(data_file_1[0:50, :])
        # print('data_file shape:', data_file.shape)
        # print(data_file[0:50, :])
        data_file = np.moveaxis(data_file, -1, 0)
        print('data_file shape:', data_file.shape)
        data = data_file[0, :].reshape((140, 140, 140))
        predict_result = np.zeros((140, 140, 140))
        label = data_file[1, :].reshape((140, 140,140))
        torch.cuda.empty_cache()
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    input_cube = input_process(data[i*42:i*42+56, j*42:j*42+56, k*42:k*42+56])
                    input_cube = (torch.autograd.Variable(torch.Tensor(input_cube).float())).to(device)
                    output = network(input_cube)
                    predict_result[i*42:i*42+56, j*42:j*42+56, k*42:k*42+56] = predict(output)
        # np.savetxt(r'layer_segmentation_test_model_predict/' + str(z) + '.txt', predict_result.flatten())
        predict_matrix = predict_matrix_update(predict_result, label, 3)
        predict_matrix_sum += predict_matrix
        print(z, '!!')
    acc_PA = accuracy_calculate(predict_matrix_sum, mode='PA')
    acc_MPA = accuracy_calculate(predict_matrix_sum, mode='MPA')
    acc_MIOU = accuracy_calculate(predict_matrix_sum, mode='MIOU')
    print('acc_PA is :\n', acc_PA)
    print('acc_MPA is :\n', acc_MPA)
    print('acc_MIOU is :\n', acc_MIOU)

