# coding=UTF-8
import numpy as np
from layer_segmentation_network import U_Net
import torch
from torch import nn
import time
def data_cube_create(file_path):
    data_cube = np.zeros((200, 2, 140, 140, 140))
    for i in range(200):
        model = np.loadtxt(file_path + '/' + str(i+1) + '.txt')
        data_cube[i, 0, :, :, :] = model[0, :].reshape((140, 140, 140))
        data_cube[i, 1, :, :, :] = model[1, :].reshape((140, 140, 140))
        print('the ' + str(i+1) + 'th txt file have been loaded!')
        load_time = time.localtime()
        print('load_time:', load_time)
    return data_cube

def random_batch(data_cube, batch_size, batch_number):
    counter = 0
    batch_size_half = batch_size//2
    data_output = np.zeros((batch_number, 1, batch_size, batch_size, batch_size))
    label_output = np.zeros((batch_number, batch_size, batch_size, batch_size))
    random_number = np.random.randint(0, 200)
    data = data_cube[random_number, 0, :, :, :]
    label = data_cube[random_number, 1, :, :, :]
    while counter < batch_number:
        random_index_0 = np.random.randint(batch_size_half, 140 - batch_size_half)
        random_index_1 = np.random.randint(batch_size_half, 140 - batch_size_half)
        random_index_2 = np.random.randint(batch_size_half, 140 - batch_size_half)
        data_output[counter, 0, :, :, :] = data[random_index_0 - batch_size_half:random_index_0 + batch_size_half,
                                           random_index_1 - batch_size_half:random_index_1 + batch_size_half,
                                           random_index_2 - batch_size_half:random_index_2 + batch_size_half]
        label_output[counter, :, :, :] = label[random_index_0 - batch_size_half:random_index_0 + batch_size_half,
                                         random_index_1 - batch_size_half:random_index_1 + batch_size_half,
                                         random_index_2 - batch_size_half:random_index_2 + batch_size_half]
        counter += 1
    return data_output, label_output
if __name__ == '__main__':
    start = time.localtime()
    print('start:', start)
    # data_cube = data_cube_create('training_model_amp_norm_simp/training_model_amp_norm_simp_1')
    torch.cuda.empty_cache()
    network = U_Net(n_classes=3)
    network.ParameterInitialize()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # cross_entropy = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.06140522704449368, 0.04453565917512728, 0.0450304998326287, 0.04824696410638789, 0.04553646050490542, 0.050033888702920776, 0.052633051752423145, 0.04605392028337025, 0.05551705458817236, 0.0471249416853091, 0.00015375768210549294, 0.054766824120764626, 0.06048873111845646, 0.043577903063834224, 0.054766824120764626, 0.05708091528079694, 0.04453565917512728, 0.0450304998326287, 0.054766824120764626, 0.04266047352564824, 0.04605392028337025] ).to(
    #     device))
    cross_entropy = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.008471212101742048, 1.495764393949129, 1.495764393949129]).to(device))
    print(torch.cuda.device_count())
    # Transfer model to gpu
    if torch.cuda.device_count() > 1:
        network = nn.DataParallel(network)
    network.to(device)
    # network.load_state_dict = (checkpoint)    #pytorch调用先前的模型参数
    network.eval()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.03)  # Adam method
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.9, last_epoch=-1)
    model_path = 'saved_model0_20000(20210531).pt'
    # optimizer.load_state_dict(torch.load(join('F3','optimizer.pth')))
    # checkpoint = {'model': network.state_dict(), 'optimizer': optimizer.state_dict()}
    # model_path = 'saved_model3.pt'
    optimizer_path = 'optimizer0_20000(20210531).pth'
    # network.module.load_state_dict(torch.load(model_path))
    for state in optimizer.state.values():
        for k, v in state.items():    #for k, v in d.items()  Iterate the key and value simultaneously
            if torch.is_tensor(v):
                state[k] = v.cuda()

            pass
        pass
    pass
    # optimizer.load_state_dict(torch.load(optimizer_path))
    loss_list = []
    network.train()
    part1 = time.localtime()
    print('part1:', part1)
    # print('part1 spend %f'%(part1 - start))
    data_cube = data_cube_create('layer_segmentation_test_model_2')
    for z in range(10000):
        scheduler.step()
        # part2 = time.localtime()
        # print('part2:', part2)
        data, label = random_batch(data_cube, 56, 20)
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
        print(r"The %d epoch's loss is:" % z, loss)
        loss_list.append(loss)
        loss.backward()
        if (z+1)%6 == 0:
            optimizer.step()
            optimizer.zero_grad()
        # part5 = time.localtime()
        # print('part5:', part5)
        if z % 300 == 0 and z > 0:
            torch.save(network.module.state_dict(), 'saved_model0_20000(20210602).pt')  # 网络保存为saved_model.pt
            torch.save(optimizer.state_dict(), 'optimizer0_20000(20210602).pth')
        pass
        # part6 = time.localtime()
        # print('part6:', part6)
    np.savetxt(r'loss_value_0_20000(20210602).txt', np.array(loss_list))
    pass

