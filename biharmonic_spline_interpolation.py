# coding=UTF-8
# -*- coding: utf-8 -*-
import numpy as np
from scipy import interpolate
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pyvista as pv
import random
# def func(x, y):
#     return (x**2 + y**2)
# # X-Y轴分为20*20的网格
# x = np.arange(-10, 10, 2)
# print('x:', x, type(x))
# y = np.arange(-10,10,2)
# xx, yy = np.meshgrid(x, y)#20*20的网格数据
# print('xx:\n', xx, type(xx), xx.shape)
# fvals = func(xx,yy) # 计算每个网格点上的函数值  15*15的值
# print('fvals:\n', fvals, type(fvals), fvals.shape)
# fig = plt.figure(figsize=(9, 6))
# #Draw sub-graph1
# ax=plt.subplot(1, 2, 1,projection = '3d')
# surf = ax.plot_surface(xx, yy, fvals, rstride=2, cstride=2, cmap=cm.coolwarm,linewidth=0.5, antialiased=True)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('f(x, y)')
# plt.colorbar(surf, shrink=0.5, aspect=5)#标注
# #二维插值
# newfunc = interpolate.interp2d(xx, yy, fvals, kind='cubic')#newfunc为一个函数
#
# # 计算100*100的网格上的插值
# xnew = np.arange(-10, 10, 1)#x
# ynew = np.arange(-10, 10, 1)#y
# fnew = newfunc(xnew, ynew)#仅仅是y值   100*100的值  np.shape(fnew) is 100*100
# xnew, ynew = np.meshgrid(xnew, ynew)
# ax2=plt.subplot(1, 2, 2,projection = '3d')
# surf2 = ax2.plot_surface(xnew, ynew, fnew, rstride=2, cstride=2, cmap=cm.coolwarm,linewidth=0.5, antialiased=True)
# ax2.set_xlabel('xnew')
# ax2.set_ylabel('ynew')
# ax2.set_zlabel('fnew(x, y)')
# plt.colorbar(surf2, shrink=0.5, aspect=5)#标注
# plt.show()

def GreenFunction(x, dim=2):
    #green function |x| ** 2 * (ln|x| - 1)
    if dim == 2:
        green_function_out = x * 1.0
        green_function_out[x != 0] = ((x[x != 0]) ** 2) * (np.log((x[x != 0])) - 1)
        green_function_out[x == 0] = 0
        return green_function_out
    elif dim == 3:
        green_function_out = x * 1.0
        # green_function_out = map(abs, green_function_out)
        return green_function_out

def vector_into_matrix(x, y):
    #transform seed point into |x| matrix
    xx = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        xx[:, i] = ((x - x[i]) ** 2 + (y - y[i]) ** 2) ** 0.5
    pass
    return xx

def vector_into_matrix_3d(x, y, z):
    #transform seed point into |x| matrix
    xx = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):

        xx[:, i] = ((x - x[i]) ** 2 + (y - y[i]) ** 2 + (z - z[i]) ** 2) ** 0.5
    pass
    # print('xx:\n', xx)
    return xx
def vector_into_matrix2(x, y, x0, y0):
    #transform interpolation point into |x| matrix
    xx = np.zeros((x.shape[0], x0.shape[0]))
    for i in range(x0.shape[0]):
        xx[:, i] = ((x - x0[i]) ** 2 + (y - y0[i]) ** 2) ** 0.5
    return xx
def vector_into_matrix2_3d(x, y, z, x0, y0, z0):
    #transform interpolation point into |x| matrix
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    x0 = x0.flatten()
    y0 = y0.flatten()
    z0 = z0.flatten()
    xx = np.zeros((x.shape[0], x0.shape[0]))
    for i in range(x0.shape[0]):
        xx[:, i] = ((x - x0[i]) ** 2 + (y - y0[i]) ** 2 + (z - z0[i]) ** 2) ** 0.5
    # print('yy:\n', xx)
    return xx
class BiharmonicSplineInterpolation(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.matrix0 = np.mat(GreenFunction(vector_into_matrix(x, y), dim=2))
        #the matrix of green_function(|x|)
        self.inverse_matrix0 = self.matrix0.I
        #the inverse matrix
        # print('%'*50, '\n', self.matrix0 * self.inverse_matrix0)
        # print('%'*50, '\n')
        self.aj = self.inverse_matrix0 * np.mat(z).T      #[a1, a2, ...].T
        #the vector of aj
        pass
    pass
    def __call__(self, x_linspace, y_linspace):
        self.parameter_matrix = np.mat(GreenFunction(vector_into_matrix2(x_linspace, y_linspace, self.x, self.y)))
        #the matrix of green_function(|x|)
        # print('self.parameter_matrix', self.parameter_matrix)
        self.output = self.parameter_matrix * self.aj
        #Ax = b
        # print(self.output)
        return self.output.getA()
class BiharmonicSplineInterpolation_3d(object):
    def __init__(self, x, y, z, r):
        self.x = x
        self.y = y
        self.z = z
        self.matrix0 = np.mat(GreenFunction(vector_into_matrix_3d(x, y, z), dim=3))
        # print('self.matrix0:\n', self.matrix0)
        #the matrix of green_function(|x|)
        self.inverse_matrix0 = self.matrix0.I
        #the inverse matrix
        # print('%'*50, '\n', self.matrix0 * self.inverse_matrix0)
        # print('%'*50, '\n')
        self.aj = self.inverse_matrix0 * np.mat(r).T      #[a1, a2, ...].T
        # print('aj:\n', self.aj, type(self.aj))
        #the vector of aj
        pass
    pass
    def __call__(self, x_linspace, y_linspace, z_linspace):
        self.parameter_matrix = np.mat(GreenFunction(vector_into_matrix2_3d(x_linspace, y_linspace, z_linspace, self.x, self.y, self.z), dim=3))
        # print('self.parameter_matrix:\n', self.parameter_matrix)
        #the matrix of green_function(|x|)
        # print('self.parameter_matrix', self.parameter_matrix)
        self.output = self.parameter_matrix * self.aj
        #Ax = b
        # print(self.output)
        return self.output
def SeedPoint_generate_fault_spine(seed_point_number):
    #generate random seed point to interpolation the fault spine
    x = []
    y = []
    z = []
    for i in range(seed_point_number):
        x.append(np.random.uniform(-100, 100))
        y.append(np.random.uniform(-100, 100))
        z.append(np.random.uniform(-5, 5))
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    return x, y, z

def fault_spine_generate(seed_point_number):
    seedpoint_x, seedpoint_y, seedpoint_z = SeedPoint_generate_fault_spine(seed_point_number)
    interpolate_surface = BiharmonicSplineInterpolation(seedpoint_x, seedpoint_y, seedpoint_z)
    return
seedpoint_x, seedpoint_y,  seedpoint_z= SeedPoint_generate_fault_spine(20)

if __name__ == '__main__':
    seedpoint_x, seedpoint_y, seedpoint_z = SeedPoint_generate_fault_spine(20)
    # seedpoint_x = np.arange(10)
    # seedpoint_y = np.arange(10)
    # seedpoint_z = np.arange(10)
    print('seedpoint_x:\n', seedpoint_x)
    print('seedpoint_y:\n', seedpoint_y)
    print('seedpoint_z:\n', seedpoint_z)
    aba = BiharmonicSplineInterpolation(seedpoint_x, seedpoint_y, seedpoint_z)
    #interpolation by seed points
    print('self.x:\n', aba.x)
    print('self.y:\n', aba.y)
    print('self.matrix0:\n', aba.matrix0)
    print('self.inverse_matrix0:\n', aba.inverse_matrix0)
    x = np.arange(-50, 50, 0.1)
    y = np.arange(-50, 50, 0.1)
    xx, yy = np.meshgrid(x, y)
    #interpolation grid points
    print('x shape', x.shape)
    t = xx.shape[0]
    xx = xx.reshape(xx.shape[0] * xx.shape[1], )
    yy = yy.reshape(yy.shape[0] * yy.shape[1], )
    z = aba(xx, yy)
    xx = xx.reshape(t, t)
    yy = yy.reshape(t, t)
    z = z.reshape(t, t)
    print('z:\n', z)
    print('z.shape', z.shape)
    mesh = pv.StructuredGrid(xx, yy, z)
    mesh['R'] = z.reshape(t*t,)
    mesh.plot()
    # a = np.arange(0, 5, 1)
    # print('a:\n', a, type(a), a.shape)
    # b = GreenFunction(a)
    # print('b:\n', b, type(b), b.shape)
    #
    # a = np.array([1, 2, 3])
    # print('a:\n', a, type(a), a.shape)
    # c = vector_into_matrix(a, a)
    # print('c:\n', c, type(c), c.shape)
    # d = GreenFunction(c)
    # print('d:\n', d, type(d), d.shape)
    #
    # a = np.arange(0, 10, 1)
    # e = BiharmonicSplineInterpolation(a, a, a)
    # print('e:   e.matrix0, type(e.matrix0), e.matrix0.shape, e.inverse_matrix0, type(e.inverse_matrix0), e.inverse_matrix.shape \n', e.matrix0, type(e.matrix0), e.matrix0.shape, '\n', e.inverse_matrix0, type(e.inverse_matrix0), e.inverse_matrix0.shape)
    # print('e.c\n', e.aj, type(e.aj), e.aj.shape)
    # g = np.arange(0, 4, 1)
    # print('%'*70)
    # f = vector_into_matrix2(a, a, g, g)
    # print(f, '\n', type(f), f.shape)
    # h = e(a, a)
    # print('h:\n', h, type(h), h.shape)
    #
    # np.random.seed(7914)
    # a = np.random.randint(-30, 30, 20)
    # b = np.random.randint(-30, 30, 20)
    # c = np.random.randn(20)*20
    # d = np.vstack([a, b, c])
    # print(d, d.shape)
    # d = np.moveaxis(d, 0, -1)
    # print(d, d.shape)



    # a = np.arange(5)
    # b = np.arange(5)
    # c = vector_into_matrix(a, b)
    # print(c)
    #
    #
    # # a = np.array([0, 0, 1, 1])
    # # b = np.array([0, 1, 0, 1])
    # # c = np.array([0.1, -0.2, 0.3, -0.25])
    # # mesh = pv.StructuredGrid(a, b, c)
    # # mesh['R'] = a + 1
    # # mesh.plot()
    # cloud = pv.PolyData(d)
    # # cloud.plot(point_size=15)
    # e = BiharmonicSplineInterpolation(a, b, c)
    # a0 = np.arange(-30, 30, 0.1)
    # b0 = np.arange(-30, 30, 0.1)
    # a1, b1 = np.meshgrid(a0, b0)
    # print('a0', a1, a1.shape)
    # print('b0', b1, b1.shape)
    # a1 = a1.reshape(360000,)
    # b1 = b1.reshape(360000,)
    # print('a1', a1, a1.shape)
    # print('b1', b1, b1.shape)
    # f = np.squeeze(e(a1, b1), axis=1)
    # print('f', f.shape)
    # f1 = np.vstack([a1, b1, f])
    # f1 = np.moveaxis(f1, 0, -1)
    # # cloud = pv.PolyData(f1)
    # # cloud.plot(point_size=15)
    # grid = pv.StructuredGrid(a1, b1, f)
    # # grid.plot()
    # plotter = pv.Plotter(shape = (1, 2))
    # plotter.subplot(0, 0)
    # plotter.add_mesh(cloud)
    # plotter.subplot(0, 1)
    # plotter.add_mesh(grid)
    # plotter.show()


    # ax=plt.subplot(projection = '3d')
    # surf = ax.plot_surface(x, y, z)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('f(x, y)')
    # plt.colorbar(surf, shrink=0.5, aspect=5)#标注
    # plt.show()

