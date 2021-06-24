# coding=UTF-8
import numpy as np
from layer_segmentation_network_128_edit0622 import U_Net
import torch
from torch import nn
import time
def data_cube_create(file_path):
    data_cube = np.zeros((20, 2, 128, 128, 128))
    for i in range(20):
        model = np.loadtxt(file_path + '/' + str(i+1) + '.txt')
        data_cube[i, 0, :, :, :] = model[0, :].reshape((128, 128, 128))
        data_cube[i, 1, :, :, :] = model[2, :].reshape((128, 128, 128))
        print('the ' + str(i+1) + 'th txt file have been loaded!')
        load_time = time.localtime()
        print('load_time:', load_time)
    return data_cube

def random_batch(data_cube):
    index_number = np.random.randint(0, 200)
    batch_number = 14
    batch_size = 128
    data_output = np.zeros((batch_number, 1, batch_size, batch_size, batch_size))
    label_output = np.zeros((batch_number, batch_size, batch_size, batch_size))
    data_output[0, 0, :, :, :] = data_cube[index_number, 0, :, :, :]
    label_output[0, :, :, :] = data_cube[index_number, 1, :, :, :]
    return data_output, label_output
# def random_batch_edit(data_cube):
#     batch_number = 2
#     batch_size = 128
#     data_output = np.zeros((batch_number, 1, batch_size, batch_size, batch_size))
#     label_output = np.zeros((batch_number, batch_size, batch_size, batch_size))
#     index_list = []
#     while len(index_list)<batch_number:
#         random_index = np.random.randint(20)
#         if not (random_index in index_list):
#             index_list.append(random_index)
#     for i in range(batch_number):
#         data_output[i, 0, :, :, :] = data_cube[index_list[i], 0, :, :, :]
#         label_output[i, :, :, :] = data_cube[index_list[i], 1, :, :, :]
#     return data_output, label_output
def random_batch_edit(data_cube, index):
    batch_number = 2
    batch_size = 128
    sample_index = index % 200
    data_output = np.zeros((batch_number, 1, batch_size, batch_size, batch_size))
    label_output = np.zeros((batch_number, batch_size, batch_size, batch_size))
    data_output[0, 0, :, :, :] = data_cube[sample_index, 0, :, :, :]
    label_output[0, :, :, :] = data_cube[sample_index, 1, :, :, :]
    index_list = [sample_index]
    while len(index_list)<batch_number:
        random_index = np.random.randint(200)
        if not (random_index in index_list):
            index_list.append(random_index)
    for i in range(1, batch_number):
        data_output[i, 0, :, :, :] = data_cube[index_list[i], 0, :, :, :]
        label_output[i, :, :, :] = data_cube[index_list[i], 1, :, :, :]
    return data_output, label_output
if __name__ == '__main__':
    start = time.localtime()
    print('start:', start)
    # data_cube = data_cube_create('training_model_amp_norm_simp/training_model_amp_norm_simp_1')
    # torch.cuda.empty_cache()
    network = U_Net(n_classes=3)
    network.ParameterInitialize()
    total_params = sum(p.numel() for p in network.parameters())
    print('parameter number:\n', total_params)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # cross_entropy = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.06140522704449368, 0.04453565917512728, 0.0450304998326287, 0.04824696410638789, 0.04553646050490542, 0.050033888702920776, 0.052633051752423145, 0.04605392028337025, 0.05551705458817236, 0.0471249416853091, 0.00015375768210549294, 0.054766824120764626, 0.06048873111845646, 0.043577903063834224, 0.054766824120764626, 0.05708091528079694, 0.04453565917512728, 0.0450304998326287, 0.054766824120764626, 0.04266047352564824, 0.04605392028337025] ).to(
    #     device))
    # cross_entropy = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.008532196145313574, 1.4945117611710765, 1.49695604268361]).to(device))
    cross_entropy = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.13095815496304367, 1.3581026003557533, 1.510939244681203]).to(device))
    print(torch.cuda.device_count())
    # Transfer model to gpu
    if torch.cuda.device_count() > 1:
        print('2 cards!')
        network = nn.DataParallel(network, device_ids=[0, 1])
    network.to(device)
    # network.load_state_dict = (checkpoint)    #pytorch调用先前的模型参数
    # network.eval()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.00003)  # Adam method
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99, last_epoch=-1)
    model_path = 'saved_model0_20000(20210604_2).pt'
    # optimizer.load_state_dict(torch.load(join('F3','optimizer.pth')))
    # checkpoint = {'model': network.state_dict(), 'optimizer': optimizer.state_dict()}
    # model_path = 'saved_model3.pt'
    optimizer_path = 'optimizer0_20000(20210604_2).pth'
    # network.module.load_state_dict(torch.load(model_path))
    # network.module.load_state_dict(torch.load(model_path), map_location=torch.device('cpu')
    for state in optimizer.state.values():
        for k, v in state.items():  # for k, v in d.items()  Iterate the key and value simultaneously
            if torch.is_tensor(v):
                state[k] = v.cuda()

            pass
        pass
    pass
    # optimizer.load_state_dict(torch.load(optimizer_path))
    loss_list = []
    loss_list_1 = []
    network.train()
    part1 = time.localtime()
    print('part1:', part1)
    # print('part1 spend %f'%(part1 - start))
    data_cube = data_cube_create('layer_segmentation_test_model_fold_fault_multi')
    for z in range(50):
        for i in range(100):
            # scheduler.step()
            # part2 = time.localtime()
            # print('part2:', part2)
            data, label = random_batch_edit(data_cube)
            # part21 = time.localtime()
            # print('part21:', part21)
            data = (torch.autograd.Variable(torch.Tensor(data).float())).to(device)
            # part22 = time.localtime()
            # print('part22:', part22)
            label = (torch.autograd.Variable(torch.Tensor(label).long())).to(device)
            # part3 = time.localtime()
            # print('part3:', part3)
            output = network(data)
            # part4 = time.localtime()
            # print('part4:', part4)
            loss = cross_entropy(output, label)
            data = 0
            label = 0
            output = 0
            print(r"The %dth epoch's %dth batch's loss is:" % (z, i+1), loss)
            loss_list.append(loss)
            # if i == 99:
            #     loss_list_1.append(loss)
            loss_list_1.append(loss)
            loss.backward()
            loss = 0
            # if (i + 1) % 2 == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()
            optimizer.step()
            optimizer.zero_grad()
            # part5 = time.localtime()
            # print('part5:', part5)
            # scheduler.step()
        torch.save(network.module.state_dict(), 'saved_model0_20000(20210623_2).pt')  # 网络保存为saved_model.pt
        torch.save(optimizer.state_dict(), 'optimizer0_20000(20210623_2).pth')

        # part6 = time.localtime()
        # print('part6:', part6)
        scheduler.step()
    np.savetxt(r'loss_value_0_20000(20210623_2).txt', np.array(loss_list))
    np.savetxt(r'loss_value1_0_20000(20210623_2).txt', np.array(loss_list_1))
    pass




    # start = time.localtime()
    # print('start:', start)
    # # data_cube = data_cube_create('training_model_amp_norm_simp/training_model_amp_norm_simp_1')
    # torch.cuda.empty_cache()
    # network = U_Net(n_classes=3)
    # network.ParameterInitialize()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # # cross_entropy = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.06140522704449368, 0.04453565917512728, 0.0450304998326287, 0.04824696410638789, 0.04553646050490542, 0.050033888702920776, 0.052633051752423145, 0.04605392028337025, 0.05551705458817236, 0.0471249416853091, 0.00015375768210549294, 0.054766824120764626, 0.06048873111845646, 0.043577903063834224, 0.054766824120764626, 0.05708091528079694, 0.04453565917512728, 0.0450304998326287, 0.054766824120764626, 0.04266047352564824, 0.04605392028337025] ).to(
    # #     device))
    # cross_entropy = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.008532196145313574, 1.4945117611710765, 1.49695604268361]).to(device))
    # print(torch.cuda.device_count())
    # # Transfer model to gpu
    # if torch.cuda.device_count() > 1:
    #     network = nn.DataParallel(network)
    # network.to(device)
    # # network.load_state_dict = (checkpoint)    #pytorch调用先前的模型参数
    # network.eval()
    # optimizer = torch.optim.Adam(network.parameters(), lr=0.00001)  # Adam method
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9, last_epoch=-1)
    # model_path = 'saved_model0_20000(20210531).pt'
    # # optimizer.load_state_dict(torch.load(join('F3','optimizer.pth')))
    # # checkpoint = {'model': network.state_dict(), 'optimizer': optimizer.state_dict()}
    # # model_path = 'saved_model3.pt'
    # optimizer_path = 'optimizer0_20000(20210531).pth'
    # # network.module.load_state_dict(torch.load(model_path))
    # for state in optimizer.state.values():
    #     for k, v in state.items():    #for k, v in d.items()  Iterate the key and value simultaneously
    #         if torch.is_tensor(v):
    #             state[k] = v.cuda()
    #
    #         pass
    #     pass
    # pass
    # # optimizer.load_state_dict(torch.load(optimizer_path))
    # loss_list = []
    # network.train()
    # part1 = time.localtime()
    # print('part1:', part1)
    # # print('part1 spend %f'%(part1 - start))
    # data_cube = data_cube_create('layer_segmentation_test_model_fold_fault')
    # for z in range(1000):
    #     scheduler.step()
    #     # part2 = time.localtime()
    #     # print('part2:', part2)
    #     data, label = random_batch(data_cube)
    #     # part21 = time.localtime()
    #     # print('part21:', part21)
    #     data = (torch.autograd.Variable(torch.Tensor(data).float())).to(device)
    #     # part22 = time.localtime()
    #     # print('part22:', part22)
    #     label = (torch.autograd.Variable(torch.Tensor(label).long())).to(device)
    #     # part3 = time.localtime()
    #     # print('part3:', part3)
    #     output = network(data)
    #     # part4 = time.localtime()
    #     # print('part4:', part4)
    #     loss = cross_entropy(output, label)
    #     print(r"The %d epoch's loss is:" % z, loss)
    #     loss_list.append(loss)
    #     loss.backward()
    #     if (z+1)%2 == 0:
    #         optimizer.step()
    #         optimizer.zero_grad()
    #     # part5 = time.localtime()
    #     # print('part5:', part5)
    #     if z % 300 == 0 and z > 0:
    #         torch.save(network.module.state_dict(), 'saved_model0_20000(20210603_2).pt')  # 网络保存为saved_model.pt
    #         torch.save(optimizer.state_dict(), 'optimizer0_20000(20210603_2).pth')
    #     pass
    #     # part6 = time.localtime()
    #     # print('part6:', part6)
    # np.savetxt(r'loss_value_0_20000(20210603_2).txt', np.array(loss_list))
    # pass

