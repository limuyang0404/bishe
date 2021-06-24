# coding=UTF-8
import numpy as np
import math
import pyvista as pv
from biharmonic_spline_interpolation import BiharmonicSplineInterpolation


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
def fault_displacement_field(x, y, lx, ly, d_max):
    # print('x', x)
    # print('y', y)
    # print('x / lx', x / lx)
    rxy = ((x / lx) ** 2 + (y / ly) ** 2) ** 0.5
    # print('rxy', rxy, type(rxy), rxy.shape)
    rxy[rxy > 1] = 1
    # print('rxy', rxy, type(rxy), rxy.shape)
    displacement_out = 2 * d_max * (1 - rxy) * ((1 + rxy) ** 2 / 4 - rxy ** 2) ** 0.5
    return displacement_out
def fault_displacement_y(displacement_field, z, fxy, reverse_drag_radius, hwfwradio):
    fault_displacement_y_out = displacement_field * 0
    # print('z', type(z), z.shape)
    # print('fxy', type(fxy), fxy.shape)
    z_fxy = z - fxy
    # print('z - fxy', type(z_fxy), z_fxy.shape)
    fault_displacement_y_out[np.where((0 <= z_fxy) & (z_fxy < reverse_drag_radius))] = hwfwradio * displacement_field[
        np.where((0 <= z_fxy) & (z_fxy < reverse_drag_radius))] * alpha_function(fxy[np.where((0 <= z_fxy) & (z_fxy < reverse_drag_radius))], z[np.where((0 <= z_fxy) & (z_fxy < reverse_drag_radius))],
                                                                 reverse_drag_radius)
    fault_displacement_y_out[np.where((-1 * reverse_drag_radius <= z_fxy) & (z_fxy < 0))] = (hwfwradio-1) * displacement_field[
        np.where((-1 * reverse_drag_radius <= z_fxy) & (z_fxy < 0))] * alpha_function(fxy[np.where((-1 * reverse_drag_radius <= z_fxy) & (z_fxy < 0))], z[np.where((-1 * reverse_drag_radius <= z_fxy) & (z_fxy < 0))],
                                                                 reverse_drag_radius)
    return fault_displacement_y_out
def fault_displacement_z(x, y, displacement_y, fxy, fxy_example):
    y_edit = y + displacement_y
    # print('y_edit', type(y_edit), y_edit.shape)
    fxy_edit = fxy_example(x, y_edit)
    # print('fxy_edit', type(fxy_edit), fxy_edit.shape)
    fxy_edit = fxy_edit.reshape(fxy_edit.shape[0],)
    displacement_z = fxy_edit - fxy
    return displacement_z
def R_matrix(strike_angle, dip_angle):
    list_matrix_value = []
    list_matrix_value.append(math.sin(strike_angle))
    list_matrix_value.append(-1 * math.cos(strike_angle))
    list_matrix_value.append(0)
    list_matrix_value.append(math.cos(strike_angle) * math.cos(dip_angle))
    list_matrix_value.append(math.sin(strike_angle) * math.cos(dip_angle))
    list_matrix_value.append(math.sin(dip_angle))
    list_matrix_value.append(math.cos(strike_angle) * math.sin(dip_angle))
    list_matrix_value.append(math.sin(strike_angle) * math.sin(dip_angle))
    list_matrix_value.append(-1 * math.cos(dip_angle))
    matrix = np.mat(np.round(np.array(list_matrix_value), 4).reshape(3, 3))
    return matrix

def alpha_function(fxy, z, reverse_drag_radius):
    #alpha is used in displacement_y
    # print('Here is alpha_function')
    # print('fxy', type(fxy), fxy.shape)
    alpha_xyz = (1 - abs(z - fxy) / reverse_drag_radius) ** 2
    # print('the alpha is ', alpha_xyz)
    return alpha_xyz
class TrainingData(object):
    def __init__(self, random_seed, work_area):
        print('random_seed', random_seed)
        np.random.seed(random_seed)
        self.dt = np.arange(-0.02, 0.02, 0.002)
        self.fm = 45
        self.work_area = work_area
        self.wavelet = None
        self.layer, self.cls, self.X, self.Y, self.Z, self.XYZ_whole_number, self.convolve_shape = None, None, None, None, None, None, None
        self.amplitude, self.amplitude_norm = None, None
        self.noise = None
        self.number_of_fold, self.number_of_fault = None, None
        self.fold_parameter = []
        self.fault_parameter = []
        self.a0 = 0
        self.bk, self.ck, self.dk, self.ek, self.a, self.b = None, None, None, None, None, None
        self.strike_angle, self.dip_angle, self.origin, self.fault_surface_seed, self.d_max, self.lxly, self.reverse_drag_radius, self.hw_fw_radio= None, None, None, None, None, None, None, None
        self.s1_shift, self.s2_shift = None, None
        self.layer_model, self.fold_model, self.fault_model, self.show_model, self.saved_model = None, None, None, None, None
        self.transform_matrix, self.inverse_transform_matrix, self.X_location, self.Y_location, self.Z_location, self.XYZ_location, self.xyz_location = None, None, None, None, None, None, None
        self.spine_surface, self.fxy, self.displacement_field, self.displacement_y, self.displacement_z = None, None, None, None, None
        self.mean_value, self.standard_deviation = None, None
    def layer_strcture(self):
        self.layer, self.cls, self.X, self.Y, self.Z = Layerstrcture(self.work_area[0], self.work_area[1], self.work_area[2])
        self.XYZ_whole_number = self.work_area[0] * self.work_area[1] * self.work_area[2]
        self.convolve_shape = [self.work_area[0] * self.work_area[1], self.work_area[2]]
        self.layer_model = [self.X, self.Y, self.Z, self.layer]
        return
    def ricker_wavelet(self):
        self.wavelet = (1 - 2 * (math.pi * self.fm * self.dt) ** 2) * np.exp(-1 * (math.pi * self.fm * self.dt) ** 2)
        # print('Oh here is wavelet:\n', self.wavelet, type(self.wavelet), self.wavelet.size)
        return
    def wavelet_convolve(self):
        # print('Oh here is ')
        # self.amplitude = WaveletConvolve(wavelet=self.wavelet, rc=self.layer.reshape(self.work_area[0] * self.work_area[1], self.work_area[2])).reshape(self.XYZ_whole_number,)
        self.amplitude = WaveletConvolve(wavelet=self.wavelet, rc=self.layer.reshape(self.work_area[0] * self.work_area[1], self.work_area[2])).reshape(self.XYZ_whole_number,)
        # print('Layer:\n', self.layer)
        # print('Amplitude:\n', self.amplitude)
        return
    def random_noise(self):
        # self.noise = np.random.normal(0, 0.01, self.amplitude.shape)
        self.noise = np.random.normal(0, 0.0000001, self.amplitude.shape)
        self.amplitude = (self.amplitude + self.noise).reshape(self.XYZ_whole_number,)
        self.mean_value = np.mean(self.amplitude)
        self.standard_deviation = np.var(self.amplitude) ** 0.5
        self.amplitude_norm = (self.amplitude - self.mean_value) / self.standard_deviation
        return
    def random_parameter(self):
        self.number_of_fold = np.random.randint(10, 25)
        self.bk = np.random.uniform(-15, 15, size=self.number_of_fold)
        self.ck = np.random.uniform(0, self.work_area[0], size=self.number_of_fold)
        self.dk = np.random.uniform(0, self.work_area[0], size=self.number_of_fold)
        # self.ek = np.random.uniform(0.8 * np.min([self.work_area[0], self.work_area[1]]), 1 * np.min([self.work_area[0], self.work_area[1]]), size=self.number_of_fold)
        self.ek = np.random.uniform(40, 80, size=self.number_of_fold)
        self.a = np.random.uniform(-0.1, 0.1)
        self.b = np.random.uniform(-0.1, 0.1)
        self.fold_parameter.extend([self.a0, self.bk, self.ck, self.dk, self.ek, self.a, self.b])
        self.number_of_fault = np.random.randint(1, 3)
        # self.number_of_fault = 2
        self.strike_angle = np.random.uniform(0, math.pi * 2, size=self.number_of_fault)
        # self.strike_angle[0] = math.pi / 2
        self.dip_angle = np.random.uniform(0.01, math.pi / 3, size=self.number_of_fault)
        # self.dip_angle[0] = math.pi / 4
        self.origin = np.array([np.random.uniform(0.2 * self.work_area[0], 0.8 * self.work_area[0], size=self.number_of_fault),
                                np.random.uniform(0.2 * self.work_area[1], 0.8 * self.work_area[1], size=self.number_of_fault),
                               np.random.uniform(0.2 * self.work_area[2], 0.8 * self.work_area[2], size=self.number_of_fault)])
        self.fault_surface_seed = np.array([np.random.uniform(-100, 100, size=(self.number_of_fault, 40)), np.random.uniform(-100, 100, size=(self.number_of_fault, 40)), np.random.uniform(-3, 3, size=(self.number_of_fault, 40))])
        self.d_max = np.random.uniform(0, 70, size=self.number_of_fault) * (-1) ** np.random.randint(0, 2, size=self.number_of_fault)
        self.lxly = np.array([np.random.uniform(150, 200, self.number_of_fault), np.random.uniform(100, 150, self.number_of_fault)])
        self.reverse_drag_radius = np.random.uniform(10, 80, size=self.number_of_fault)
        self.hw_fw_radio = np.random.uniform(0.01, 1, size=self.number_of_fault)
        self.fault_parameter.extend([self.strike_angle, self.dip_angle, self.origin, self.fault_surface_seed, self.d_max, self.lxly, self.reverse_drag_radius, self.hw_fw_radio])

        return
    def fold_generate(self):
        self.s1_shift = self.Z * 0
        for i in range(self.number_of_fold):
            self.s1_shift = self.s1_shift + self.bk[i] * np.exp(-1 * ((self.X - self.ck[i]) ** 2 + (self.Y - self.dk[i]) ** 2) / (2 * self.ek[i] ** 2))
            pass
        self.s1_shift = self.s1_shift * self.Z / self.work_area[2]
        self.s2_shift = self.Z * 0
        # self.s2_shift = self.s2_shift + self.a * self.X + self.b * self.Y + (-1 * self.X * self.work_area[0]//2 - self.Y * self.work_area[1]//2)
        self.s2_shift = self.s2_shift + self.a * self.X + self.b * self.Y - self.a * self.work_area[0]//2 - self.b * self.work_area[1]//2
        self.Z = self.Z + self.s1_shift + self.s2_shift
        self.fold_model = [self.X, self.Y, self.Z]
        return
    def fault_generate(self):
        for i in range(self.number_of_fault):
            self.transform_matrix = R_matrix(self.strike_angle[i], self.dip_angle[i])
            self.inverse_transform_matrix = self.transform_matrix.I
            self.X_location, self.Y_location, self.Z_location = self.X - self.origin[0, i], self.Y - self.origin[1, i], self.Z - self.origin[2, i]
            self.XYZ_location = np.mat(np.vstack([self.X_location, self.Y_location, self.Z_location]))
            self.xyz_location = self.transform_matrix * self.XYZ_location
            self.spine_surface = BiharmonicSplineInterpolation(self.fault_surface_seed[0, i, :], self.fault_surface_seed[1, i, :], self.fault_surface_seed[2, i, :])
            # print('check:\n', self.xyz_location[0, :].getA().shape, self.xyz_location[1, :].getA().shape)
            self.fxy = self.spine_surface(self.xyz_location[0, :].getA().reshape(self.XYZ_whole_number,), self.xyz_location[1, :].getA().reshape(self.XYZ_whole_number,))
            self.fxy = self.fxy.reshape(self.fxy.shape[0],)
            self.displacement_field = fault_displacement_field(self.xyz_location[0, :].getA().reshape(self.XYZ_whole_number,), self.xyz_location[1, :].getA().reshape(self.XYZ_whole_number,), self.lxly[0, i], self.lxly[1, i], self.d_max[i])
            # print('self.displacement_field:\n', self.displacement_field.shape, self.fxy.shape)
            self.displacement_y = fault_displacement_y(self.displacement_field, self.xyz_location[2, :].getA().reshape(self.XYZ_whole_number,), self.fxy, self.reverse_drag_radius[i],
                                                  self.hw_fw_radio[i])
            self.displacement_z = fault_displacement_z(self.xyz_location.getA()[0, :], self.xyz_location.getA()[1, :], self.displacement_y, self.fxy, self.spine_surface)
            self.XYZ_location = self.inverse_transform_matrix * (self.xyz_location + np.mat(np.vstack([self.xyz_location[0, :] * 0, self.displacement_y, self.displacement_z]))) + np.mat(
                np.array([self.origin[0, i], self.origin[1, i], self.origin[2, i]]).reshape(3, 1))
            self.X, self.Y, self.Z = self.XYZ_location.getA()[0, :].reshape(self.XYZ_whole_number,), self.XYZ_location.getA()[1, :].reshape(self.XYZ_whole_number,), self.XYZ_location.getA()[2, :].reshape(self.XYZ_whole_number,)
        self.fault_model = [self.X, self.Y, self.Z]

        return
    def model_3d_show(self, show_mode):
        if show_mode == 'Layer':
            self.show_model = pv.StructuredGrid(self.layer_model[0].reshape([self.work_area[0], self.work_area[1], self.work_area[2]], order = 'F'), self.layer_model[1].reshape([self.work_area[0], self.work_area[1], self.work_area[2]], order = 'F'), self.layer_model[2].reshape([self.work_area[0], self.work_area[1], self.work_area[2]], order = 'F'))
            self.show_model['R'] = self.cls
            # print('Oh here is amplitude:\n', self.amplitude.shape, type(self.amplitude), self.layer.shape, type(self.layer))
            self.show_model.plot()
        elif show_mode == 'Fold':
            # print('fold_model[0]:\n', self.fold_model[0].shape)
            # print('fold_model[1]:\n', self.fold_model[1].shape)
            # print('fold_model[2]:\n', self.fold_model[2].shape)
            self.show_model = pv.StructuredGrid(self.fold_model[0].reshape([self.work_area[0], self.work_area[1], self.work_area[2]], order = 'F'), self.fold_model[1].reshape([self.work_area[0], self.work_area[1], self.work_area[2]], order = 'F'), self.fold_model[2].reshape([self.work_area[0], self.work_area[1], self.work_area[2]], order = 'F'))
            # self.show_model = pv.StructuredGrid(self.fold_model[0], self.fold_model[1], self.fold_model[2])
            self.show_model['R'] = self.amplitude
            self.show_model.plot()
        elif show_mode == 'Fault':
            # print(r"fault's info:\n", self.strike_angle, self.dip_angle, self.d_max)
            self.show_model = pv.StructuredGrid(self.fault_model[0].reshape([self.work_area[0], self.work_area[1], self.work_area[2]], order = 'F'), self.fault_model[1].reshape([self.work_area[0], self.work_area[1], self.work_area[2]], order = 'F'), self.fault_model[2].reshape([self.work_area[0], self.work_area[1], self.work_area[2]], order = 'F'))
            # print('Oh here is amplitude:\n', self.amplitude.shape, type(self.amplitude), self.layer.shape, type(self.layer))
            # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            self.show_model['R'] = self.amplitude
            # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            # print(self.show_model)
            # self.show_model.plot()
            pv.plot(self.show_model)
            # print('%%%%%%%')
        return
    def model_save(self):
        self.saved_model = np.moveaxis(np.vstack([self.fault_model[0], self.fault_model[1], self.fault_model[2], self.layer, self.amplitude_norm, self.cls, self.amplitude]), -1, 0)
        return
def Layerstrcture(X_range, Y_range, Z_range):
    reflect_rate_list = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    X, Y, Z = np.meshgrid(np.arange(0, X_range, dtype='float32'), np.arange(0, Y_range, dtype='float32'), np.arange(0, Z_range, dtype='float32'))
    X, Y, Z = X.reshape(X_range * Y_range * Z_range), Y.reshape(X_range * Y_range * Z_range), Z.reshape(X_range * Y_range * Z_range)
    r = X * 0
    cls = X * 0 + 10
    i = np.random.randint(10, 21)
    while i <Z_range:
        random_para = np.random.randint(0, 21)
        r[Z == i] = reflect_rate_list[random_para]
        cls[Z == i] = random_para
        i += np.random.randint(6, 26)
    return r, cls, X, Y, Z
def read_saved_model(file_path):
    model_array = np.moveaxis(np.loadtxt(file_path), -1, 0)
    # print('model_array shape', model_array.shape)
    # print(model_array[0, :].shape)
    X = model_array[0, :]
    Y = model_array[1, :]
    Z = model_array[2, :]
    r = model_array[3, :]
    amp = model_array[4, :]
    X = X.reshape([140, 140, 140], order='F')
    Y = Y.reshape([140, 140, 140], order='F')
    Z = Z.reshape([140, 140, 140], order='F')
    p = pv.Plotter(shape=(1, 2))
    p.subplot(0, 0)
    # _ = p.add_mesh(pv.StructuredGrid(X, Y, Z))
    model_show = pv.StructuredGrid(X, Y, Z)
    model_show['R'] = r
    _ = p.add_mesh(model_show)
    # model_show.plot()
    p.subplot(0, 1)
    model_show1 = pv.StructuredGrid(X, Y, Z)
    model_show1['R'] = amp
    _ = p.add_mesh(model_show1)
    # model_show1.plot()
    p.show()
    return
if __name__ == '__main__':
    data = TrainingData(7914, [180, 180, 180])
    data.layer_strcture()
    data.ricker_wavelet()
    data.wavelet_convolve()
    data.random_noise()
    data.random_parameter()
    print('data.number_of_fold:\n', data.number_of_fold)
    print('data.bk:\n', data.bk)
    print('data.ck:\n', data.ck)
    print('data.dk:\n', data.dk)
    print('data.ek:\n', data.ek)
    print('data.fold_parameter:\n', data.fold_parameter)
    print(data.fold_parameter[1])
    print('%'*60)
    print('&'*60)
    print('%' * 60)
    print('&' * 60)
    print('self.a0, self.bk, self.ck, self.dk, self.ek, self.a, self.b\n', data.fold_parameter)
    print('self.strike_angle, self.dip_angle, self.origin, self.fault_surface_seed, self.d_max, self.lxly, self.reverse_drag_radius, self.hw_fw_radio\n', data.fault_parameter)
    print('self.fault_surface_seed', data.fault_surface_seed, data.fault_surface_seed.shape)
    data.fold_generate()
    data.fault_generate()
    print('data.fold_model:\n')
    # data.model_3d_show(show_mode='Fault')
    data.model_save()
    print('data.saved:', data.saved_model.shape, type(data.saved_model[0, 0]), type(data.saved_model[0, 1]), type(data.saved_model[0, 2]), type(data.saved_model[0, 3]), type(data.saved_model[0, 4]))
    np.savetxt(r'1.txt', data.saved_model)
    # read_saved_model(r'C:\Users\Administrator\Desktop\1.txt')

