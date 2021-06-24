# coding=UTF-8
import numpy as np
import math
from fold_and_test3 import TrainingData
import pyvista as pv
from collections import Counter
from interpolation_2 import model_structure_grid1
def WaveletConvolve(wavelet, rc):
    convolve_output = np.zeros(np.shape(rc))
    if len(rc.shape) == 2:
        for j in range(0, rc.shape[0]):
            convolve_output[j, :] = np.convolve(rc[j, :], wavelet, mode='same')
            pass
        pass
    elif len(rc.shape) == 3:
        for i in range(rc.shape[0]):
            for j in range(rc.shape[1]):
                convolve_output[i, j, :] = np.convolve(rc[i, j, :], wavelet, mode='same')
    return convolve_output
# def Layerstrcture(X_range, Y_range, Z_range):
#     reflect_rate_list = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
#     X, Y, Z = np.meshgrid(np.arange(0, X_range, dtype='float32'), np.arange(0, Y_range, dtype='float32'), np.arange(0, Z_range, dtype='float32'))
#     X, Y, Z = X.reshape(X_range * Y_range * Z_range), Y.reshape(X_range * Y_range * Z_range), Z.reshape(X_range * Y_range * Z_range)
#     r = X * 0
#     cls = X * 0 + 10
#     i = np.random.randint(10, 21)
#     while i <Z_range:
#         random_para = np.random.randint(0, 21)
#         r[Z == i] = reflect_rate_list[random_para]
#         cls[Z == i] = random_para
#         i += np.random.randint(6, 26)
#     return r, cls, X, Y, Z
# def face_add(model_array, horizon_number):
#     cls_array = np.zeros(shape=model_array.shape)
#     model_size = model_array.shape[2]
#     horizon_z_index = []
#     horizon_z_index_add = []
#     for i in range(horizon_number):
#         counter_1 = 0
#         while counter_1<1:
#             random_horizon_position = np.random.randint(int(0.35 * model_size), int(0.65 * model_size))
#             if not random_horizon_position in horizon_z_index + horizon_z_index_add:
#                 counter_1 += 1
#                 horizon_z_index.append(random_horizon_position)
#                 horizon_z_index_add.extend([random_horizon_position-1, random_horizon_position-2,random_horizon_position-3,random_horizon_position+1,random_horizon_position+2,random_horizon_position+3])
#         # random_horizon_position = np.random.randint(int(0.35*model_size), int(0.65*model_size))
#         print(random_horizon_position)
#         random_horizon_reflect_value1 = np.random.uniform(low=0.1, high=0.9)
#         random_horizon_reflect_value2 = np.random.uniform(low=-0.9, high=-0.1)
#         horizon_type = np.random.randint(1, 3)
#         if horizon_type == 1:
#             for i in range(model_array.shape[0]):
#                 for j in range(model_array.shape[1]):
#                     model_array[i, j, random_horizon_position] = random_horizon_reflect_value1 + np.random.uniform(low=-0.1, high=0.1)
#                     cls_array[i, j, random_horizon_position] = 1
#         elif horizon_type == 2:
#             for i in range(model_array.shape[0]):
#                 for j in range(model_array.shape[1]):
#                     model_array[i, j, random_horizon_position] = random_horizon_reflect_value2 + np.random.uniform(low=-0.1, high=0.1)
#                     cls_array[i, j, random_horizon_position] = 2
#     return model_array, cls_array
def face_add(model_array):
    cls_array = np.zeros(shape=model_array.shape)
    model_size = model_array.shape[2]
    horizon_z_index = []
    horizon_z_index_add = []
    random_list = [0, 0, 0, 0, 1, 2]
    for i in range(model_size):
        if not i in horizon_z_index + horizon_z_index_add:
            horizon_z_index.append(i)
            horizon_z_index_add.extend([i+1, i+2, i-1, i-2])
            class_random = np.random.randint(6)
            if random_list[class_random] != 0:
                if random_list[class_random] == 1:
                    random_horizon_reflect_value1 = np.random.uniform(low=0.1, high=0.9)
                    for j in range(model_array.shape[0]):
                        for k in range(model_array.shape[1]):
                            model_array[j, k, i] = random_horizon_reflect_value1 + np.random.uniform(low=-0.05, high=0.05)
                            cls_array[j, k, i] = 1
                elif random_list[class_random] == 2:
                    random_horizon_reflect_value2 = np.random.uniform(low=-0.9, high=-0.1)
                    for j in range(model_array.shape[0]):
                        for k in range(model_array.shape[1]):
                            model_array[j, k, i] = random_horizon_reflect_value2 + np.random.uniform(low=-0.05, high=0.05)
                            cls_array[j, k, i] = 2
        else:
            pass
        # counter_1 = 0
        # while counter_1<1:
        #     random_horizon_position = np.random.randint(int(0.35 * model_size), int(0.65 * model_size))
        #     if not random_horizon_position in horizon_z_index + horizon_z_index_add:
        #         counter_1 += 1
        #         horizon_z_index.append(random_horizon_position)
        #         horizon_z_index_add.extend([random_horizon_position-1, random_horizon_position-2,random_horizon_position-3,random_horizon_position+1,random_horizon_position+2,random_horizon_position+3])
        # # random_horizon_position = np.random.randint(int(0.35*model_size), int(0.65*model_size))
        # print(random_horizon_position)
        # random_horizon_reflect_value1 = np.random.uniform(low=0.1, high=0.9)
        # random_horizon_reflect_value2 = np.random.uniform(low=-0.9, high=-0.1)
        # horizon_type = np.random.randint(1, 3)
        # if horizon_type == 1:
        #     for i in range(model_array.shape[0]):
        #         for j in range(model_array.shape[1]):
        #             model_array[i, j, random_horizon_position] = random_horizon_reflect_value1 + np.random.uniform(low=-0.1, high=0.1)
        #             cls_array[i, j, random_horizon_position] = 1
        # elif horizon_type == 2:
        #     for i in range(model_array.shape[0]):
        #         for j in range(model_array.shape[1]):
        #             model_array[i, j, random_horizon_position] = random_horizon_reflect_value2 + np.random.uniform(low=-0.1, high=0.1)
        #             cls_array[i, j, random_horizon_position] = 2
    return model_array, cls_array
class LayerModel(object):
    def __init__(self, layer_size, wavelet):
        super(LayerModel).__init__()
        self.wavelet = wavelet
        self.layer_size = layer_size
        self.output_model = np.zeros((self.layer_size, self.layer_size, self.layer_size))
        self.output_model, self.output_model_cls = face_add(self.output_model)
        self.output_model_amp = WaveletConvolve(wavelet, self.output_model)
        self.output_model_amp_mean = np.mean(self.output_model_amp)
        self.output_model_amp_deviation = np.var(self.output_model_amp) ** 0.5
        self.output_model_amp_norm = (self.output_model_amp - self.output_model_amp_mean) / self.output_model_amp_deviation
        # self.mean_value = np.mean(self.output_model_amp)
        # self.standard_deviation = np.var(self.output_model_amp)
    def fold_and_fault_generate(self, random_seed=np.random.randint(20000)):
        self.XYZ = TrainingData(random_seed=random_seed, work_area = [self.layer_size, self.layer_size, self.layer_size])
        self.XYZ.layer_strcture()
        self.XYZ.random_parameter()
        self.XYZ.fold_generate()
        self.XYZ.fault_generate()
        self.saved_model = np.vstack([self.XYZ.fault_model[0], self.XYZ.fault_model[1], self.XYZ.fault_model[2], self.output_model.flatten(), self.output_model_amp_norm.flatten(), self.output_model_cls.flatten()])
    def structured_grid(self):
        _, _, _, self.structured_grid_r, self.structured_grid_cls, self.structured_grid_amp, _ = model_structure_grid1(self.saved_model[0, :], self.saved_model[1, :], self.saved_model[2, :], self.saved_model[3, :], self.saved_model[5, :], self.saved_model[4, :], origin_data_size=self.layer_size, structure_grid_size=128)
        print('b', Counter(self.structured_grid_r.flatten()), type(self.structured_grid_r), self.structured_grid_r.shape)
        print('c', self.wavelet)
        self.output_model_amp1 = WaveletConvolve(self.wavelet, self.structured_grid_r.reshape((128, 128, 128)))
        print('a:', Counter(self.output_model_amp1.flatten()))
        self.output_model_amp_mean1 = np.mean(self.output_model_amp1)
        self.output_model_amp_deviation1 = np.var(self.output_model_amp1) ** 0.5
        self.output_model_amp_norm1 = (self.output_model_amp1 - self.output_model_amp_mean1) / self.output_model_amp_deviation1
        self.structured_grid_amp2 = self.output_model_amp_norm1

        pass



if __name__ == '__main__':
    dt = np.arange(-0.026, 0.026, 0.002)
    fm = 30
    wavelet = (1 - 2 * (math.pi * fm * dt) ** 2) * np.exp(-1 * (math.pi * fm * dt) ** 2)
    # for z in range(20):
    #     model = LayerModel(layer_size=140, wavelet=wavelet)
    #     model.fold_and_fault_generate(random_seed=np.random.randint(20000))
    #     model.structured_grid()
    #     np.savetxt(r'layer_segmentation_test_model_fold_fault/' + str(z+1) + '.txt', np.vstack([model.structured_grid_r, model.structured_grid_amp, model.structured_grid_cls]))
    #     print('The ' + str(z+1) + 'th model have generated!')

    for z in range(0, 200):
        model = LayerModel(layer_size=160, wavelet=wavelet)
        model.fold_and_fault_generate(random_seed=np.random.randint(20000))
        model.structured_grid()
        # np.savetxt(r'layer_segmentation_test_model_fold_fault/' + str(z+1) + '.txt', np.vstack([model.output_model_amp_norm.flatten(), model.output_model_cls.flatten()]))
        np.savetxt(r'layer_segmentation_test_model_fold_fault_multi2/' + str(z+1) + '.txt', np.vstack([model.structured_grid_amp2.flatten(), model.structured_grid_r.flatten(), model.structured_grid_cls.flatten()]))
        # np.savetxt(r'layer_segmentation_test_model_fold_fault_multi/' + str(z+1) + '.txt', np.vstack([model.structured_grid_amp.flatten(), model.structured_grid_r.flatten(), model.structured_grid_cls.flatten()]))
        print('The ' + str(z+1) + 'th model have generated!')




    # model = LayerModel(layer_size=180, wavelet=wavelet)
    # model.fold_and_fault_generate()
    # model.structured_grid()
    # a = np.vstack([model.saved_model[0, :], model.saved_model[1, :], model.saved_model[2, :], model.saved_model[3, :], model.saved_model[4, :], model.saved_model[5, :]])
    # np.savetxt('check1.txt', a)
    # b = np.vstack([model.structured_grid_r, model.structured_grid_amp, model.structured_grid_cls])
    # np.savetxt('check2.txt', b)
    # a = np.loadtxt('check1.txt')
    # a1 = a[0, :]
    # a2 = a[1, :]
    # b = np.
    #
    # X = np.arange(140)
    # Y = np.arange(140)
    # Z = np.arange(140)
    # XX, YY, ZZ = np.meshgrid(X, Y, Z)
    # XX, YY, ZZ = XX.flatten().reshape((140, 140, 140), order='F'), YY.flatten().reshape((140, 140, 140), order='F'), ZZ.flatten().reshape((140, 140, 140), order='F')
    # p = pv.Plotter(shape=(2, 2))
    # p.subplot(0, 0)
    # img1 = pv.PolyData(np.moveaxis(model.saved_model[0:3, :], -1, 0))
    # img1['amp1'] = model.saved_model[4, :]
    # _ = p.add_mesh(img1)
    # p.subplot(0, 1)
    # img2 = pv.PolyData(np.moveaxis(model.saved_model[0:3, :], -1, 0))
    # img2['cls1'] = model.saved_model[5, :]
    # _ = p.add_mesh(img2)
    # p.subplot(1, 0)
    # img3 = pv.StructuredGrid(XX, YY, ZZ)
    # img3['amp2'] = model.structured_grid_amp
    # _ = p.add_mesh(img3)
    # p.subplot(1, 1)
    # img4 = pv.StructuredGrid(XX, YY, ZZ)
    # img4['cls2'] = model.structured_grid_cls
    # _ = p.add_mesh(img4)
    #
    # # a = np.moveaxis(np.vstack([XX.flatten(), YY.flatten(), ZZ.flatten()]), -1, 0)
    # # p = pv.PolyData(a)
    # # p['amp'] = model.output_model_amp.flatten()
    # # p.plot()
    # p.show()

