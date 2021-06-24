# coding=UTF-8
import numpy as np
from collections import Counter
from functools import reduce

def weight(file_path, file_number):
    counter = Counter()
    new_counter = Counter()
    for i in range(file_number):
        model_open = np.loadtxt(file_path + '/' + str(i+1) + '.txt')
        new_counter = Counter(model_open[1, :])
        print('This is the ' + str(i+1) + 'th file and the counter is:', new_counter)
        counter += new_counter
    print('The whole counter is :', counter)
    return
def wei_calculate(list):
    beta = 1 - 1 / 41943040
    f1 = lambda x:(1 - beta)/(1 - beta**x)
    list1 = map(f1, list)
    return list1

def data_cube_create(file_path):
    # data_cube = np.zeros((10, 2, 140, 140, 140))
    counter = Counter()
    for i in range(20):
        data_cube = np.zeros((10, 2, 140, 140, 140))
        for j in range(10):
            model = np.moveaxis(np.loadtxt(file_path + '/' + str(i * 10 + j + 1) + '.txt'), -1, 0)
            data_cube[j, 0, :, :, :] = model[0, :].reshape((140, 140, 140))
            data_cube[j, 1, :, :, :] = model[1, :].reshape((140, 140, 140))
            print('the ' + str(i * 10 + j +1) + 'th txt file have been loaded!')
            new_counter = Counter(model[1, :].flatten())
            print('This is the ' + str(i * 10 + j +1) + 'th file and the counter is:', new_counter)
            counter += new_counter
        data_cube = data_cube.reshape(54880000,)
        np.savetxt(file_path + '/training_model_amp_norm_simp_1/' + str(i * 10 + 1) + '_' + str(i * 10 + 10) + '.txt', data_cube)
        print('the ' + str(i+1) + 'th file have been saved!')
    print('The whole counter is :', counter)
    return
if __name__ == '__main__':
    # weight('layer_segmentation_test_model_fold_fault', 200)
    # c = wei_calculate([196000, 137200, 196000, 98000, 176400, 137200, 196000, 137200, 215600, 156800, 51626400, 117600, 137200, 176400, 176400, 156800, 137200, 156800, 294000, 137200, 78400])
    # # a = [1293600, 1783600, 1764000, 1646400, 1744400, 1587600, 1509200, 1724800, 1430800, 1685600, 516616800, 1450400, 1313200, 1822800, 1450400, 1391600, 1783600, 1764000, 1450400, 1862000, 1724800]
    # # print(list(c), type(c))
    # d = list(c)
    # print(d)
    # f = lambda x,y:x+y
    # e = reduce(f, d)
    # print(e)
    # f = []
    # for i in d:
    #     f.append(i*21/e)
    # print(f)
    # # d = list(c)
    # # print(d, type(d), d)
    # # data_cube_create(r'training_model_amp_norm_simp')
    # # print(sum(a))
    # # print(140*140*140*200)
    c = wei_calculate([37292831, 2452452, 2197757])
    d = list(c)
    print(d)
    e = d[0] + d[1] + d[2]
    print(e)
    f = d * 1
    f[0] = f[0] * (3/e)
    f[1] = f[1] * (3/e)
    f[2] = f[2] * (3/e)
    print(d)
    print(f)