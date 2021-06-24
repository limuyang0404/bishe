# -*- coding: utf-8 -*-
from biharmonic_spline_interpolation import BiharmonicSplineInterpolation_3d
import numpy as np
import pyvista as pv
import math
import scipy
from datetime import datetime
from collections import Counter
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
def model_structure_grid(X_array, Y_array, Z_array, r_array, amp_array, origin_data_size=180, structure_grid_size=140):
    slice_start = int((origin_data_size - structure_grid_size) // 2)
    slice_end = origin_data_size - slice_start
    wavelet = RickerWavelet()
    structure_grid_r = np.zeros((structure_grid_size, structure_grid_size, structure_grid_size))
    print('datetime_now:', datetime.now())
    for i in range(X_array.shape[0]):
        if slice_start<X_array[i]<slice_end and slice_start<Y_array[i]<slice_end and slice_start<Z_array[i]<slice_end:
            if r_array[i]!=0:
                x = int((X_array[i] - slice_start) // 1)
                y = int((Y_array[i] - slice_start) // 1)
                z = int((Z_array[i] - slice_start) // 1)
                structure_grid_r[x, y, z] = r_array[i]
                if z < structure_grid_size-1 and y < structure_grid_size-1 and x < structure_grid_size-1:
                    structure_grid_r[x, y, z+1] = r_array[i]
                    structure_grid_r[x, y+1, z] = r_array[i]
                    structure_grid_r[x, y+1, z+1] = r_array[i]
                    structure_grid_r[x+1, y, z] = r_array[i]
                    structure_grid_r[x+1, y, z+1] = r_array[i]
                    structure_grid_r[x+1, y+1, z] = r_array[i]
                    structure_grid_r[x+1, y+1, z+1] = r_array[i]

    X_1 = np.arange(structure_grid_size)
    Y_1 = np.arange(structure_grid_size)
    Z_1 = np.arange(structure_grid_size)
    XX1, YY1, ZZ1 = np.meshgrid(X_1, Y_1, Z_1)
    amp = WaveletConvolve(wavelet, structure_grid_r)

    #############################################################
    amp_interpolation = scipy.interpolate.griddata(np.moveaxis(np.vstack([X_array, Y_array, Z_array]), -1, 0), np.moveaxis(amp_array, -1, 0), np.moveaxis(np.vstack([XX1.flatten(), YY1.flatten(), ZZ1.flatten()]), -1, 0), method='nearest')

    return YY1.flatten(), XX1.flatten(), ZZ1.flatten(), structure_grid_r.flatten(), amp.flatten(), amp_interpolation.flatten(), np.moveaxis(np.vstack([XX1.flatten(), YY1.flatten(), ZZ1.flatten()]), -1, 0)
def nearest_cube(i, j, k, whole_array, error_value):
    counter_list = []
    counter_list.extend([whole_array[i+1, j, k], whole_array[i-1, j, k], whole_array[i, j+1, k], whole_array[i, j-1, k], whole_array[i, j, k+1], whole_array[i, j, k-1]])
    counter_dict = Counter(counter_list)
    if counter_dict.most_common(2)[0][0]!=error_value:
        output = counter_dict.most_common(2)[0][0]
    elif counter_dict.most_common(2)[0][0] == error_value and len(counter_dict.most_common(2)) == 2:
        output = counter_dict.most_common(2)[1][0]
    else:
        output = 0
    return output

def model_structure_grid1(X_array, Y_array, Z_array, r_array, cls_array, amp_array, origin_data_size=180, structure_grid_size=140):
    slice_start = int((origin_data_size - structure_grid_size) // 2)
    slice_end = origin_data_size - slice_start
    wavelet = RickerWavelet()
    structure_grid_r = np.zeros((structure_grid_size, structure_grid_size, structure_grid_size)) * 3
    structure_grid_cls = np.ones((structure_grid_size, structure_grid_size, structure_grid_size)) * 9
    structure_grid_amp_interpolation = np.ones((structure_grid_size, structure_grid_size, structure_grid_size)) * 9
    print('datetime_now:', datetime.now())
    for i in range(X_array.shape[0]):
        if slice_start+1 <= X_array[i] < slice_end-1 and slice_start+1 <= Y_array[i] < slice_end-1 and slice_start+1 <= Z_array[
            i] < slice_end-1:
            x = int((X_array[i] - slice_start) // 1)
            y = int((Y_array[i] - slice_start) // 1)
            z = int((Z_array[i] - slice_start) // 1)
            if X_array[i] - x < 0.5:
                if Y_array[i] - y < 0.5:
                    if Z_array[i] - z < 0.5:
                        structure_grid_r[x, y, z] = r_array[i]
                        structure_grid_amp_interpolation[x, y, z] = amp_array[i]
                        structure_grid_cls[x, y, z] = cls_array[i]
                    elif Z_array[i] - z > 0.5:
                        structure_grid_r[x, y, z+1] = r_array[i]
                        structure_grid_amp_interpolation[x, y, z+1] = amp_array[i]
                        structure_grid_cls[x, y, z+1] = cls_array[i]
                elif Y_array[i] - y > 0.5:
                    if Z_array[i] - z < 0.5:
                        structure_grid_r[x, y+1, z] = r_array[i]
                        structure_grid_amp_interpolation[x, y+1, z] = amp_array[i]
                        structure_grid_cls[x, y+1, z] = cls_array[i]
                    elif Z_array[i] - z > 0.5:
                        structure_grid_r[x, y+1, z+1] = r_array[i]
                        structure_grid_amp_interpolation[x, y+1, z+1] = amp_array[i]
                        structure_grid_cls[x, y+1, z+1] = cls_array[i]
            elif X_array[i] - x > 0.5:
                if Y_array[i] - y < 0.5:
                    if Z_array[i] - z < 0.5:
                        structure_grid_r[x+1, y, z] = r_array[i]
                        structure_grid_amp_interpolation[x+1, y, z] = amp_array[i]
                        structure_grid_cls[x+1, y, z] = cls_array[i]
                    elif Z_array[i] - z > 0.5:
                        structure_grid_r[x+1, y, z+1] = r_array[i]
                        structure_grid_amp_interpolation[x+1, y, z+1] = amp_array[i]
                        structure_grid_cls[x+1, y, z+1] = cls_array[i]
                elif Y_array[i] - y > 0.5:
                    if Z_array[i] - z < 0.5:
                        structure_grid_r[x+1, y+1, z] = r_array[i]
                        structure_grid_amp_interpolation[x+1, y+1, z] = amp_array[i]
                        structure_grid_cls[x+1, y+1, z] = cls_array[i]
                    elif Z_array[i] - z > 0.5:
                        structure_grid_r[x+1, y+1, z+1] = r_array[i]
                        structure_grid_amp_interpolation[x+1, y+1, z+1] = amp_array[i]
                        structure_grid_cls[x+1, y+1, z+1] = cls_array[i]
        elif slice_start <= X_array[i] < slice_end and slice_start <= Y_array[i] < slice_end and slice_start <= Z_array[i] < slice_end and (slice_end-1 <= X_array[i] or slice_end-1 <= Y_array[i] or slice_end-1 <= Z_array[i]):
            x = int((X_array[i] - slice_start) // 1)
            y = int((Y_array[i] - slice_start) // 1)
            z = int((Z_array[i] - slice_start) // 1)
            structure_grid_r[x, y, z] = r_array[i]
            structure_grid_amp_interpolation[x, y, z] = amp_array[i]
            structure_grid_cls[x, y, z] = cls_array[i]
        elif slice_start <= X_array[i] < slice_end and slice_start <= Y_array[i] < slice_end and slice_start <= Z_array[
            i] < slice_end and (X_array[i] <slice_start + 1 or Y_array[i] <slice_start + 1 or Z_array[i]<slice_start + 1):
            x = int((X_array[i] - slice_start) // 1)
            y = int((Y_array[i] - slice_start) // 1)
            z = int((Z_array[i] - slice_start) // 1)
            structure_grid_r[x, y, z] = r_array[i]
            structure_grid_amp_interpolation[x, y, z] = amp_array[i]
            structure_grid_cls[x, y, z] = cls_array[i]


    structure_grid_amp_interpolation1 = structure_grid_amp_interpolation*1
    structure_grid_cls1 = structure_grid_cls*1
    for i in range(structure_grid_size):
        for j in range(structure_grid_size):
            for k in range(structure_grid_size):
                if structure_grid_amp_interpolation1[i, j, k] == 9:
                    if 0<i<structure_grid_size-1:
                        if 0<j<structure_grid_size-1:
                            if 0<k<structure_grid_size-1:
                                structure_grid_amp_interpolation1[i, j, k] = nearest_cube(i, j, k, structure_grid_amp_interpolation, error_value=9)
                            elif k == 0:
                                structure_grid_amp_interpolation1[i, j, k] = nearest_cube(i, j, k+1, structure_grid_amp_interpolation, error_value=9)
                            elif k == structure_grid_size-1:
                                structure_grid_amp_interpolation1[i, j, k] = nearest_cube(i, j, k-1, structure_grid_amp_interpolation, error_value=9)
                        elif j == 0:
                            if 0<k<structure_grid_size-1:
                                structure_grid_amp_interpolation1[i, j, k] = nearest_cube(i, j+1, k, structure_grid_amp_interpolation, error_value=9)
                            elif k == 0:
                                structure_grid_amp_interpolation1[i, j, k] = nearest_cube(i, j+1, k+1, structure_grid_amp_interpolation, error_value=9)
                            elif k == structure_grid_size-1:
                                structure_grid_amp_interpolation1[i, j, k] = nearest_cube(i, j+1, k-1, structure_grid_amp_interpolation, error_value=9)
                        elif j == structure_grid_size-1:
                            if 0<k<structure_grid_size-1:
                                structure_grid_amp_interpolation1[i, j, k] = nearest_cube(i, j-1, k, structure_grid_amp_interpolation, error_value=9)
                            elif k == 0:
                                structure_grid_amp_interpolation1[i, j, k] = nearest_cube(i, j-1, k+1, structure_grid_amp_interpolation, error_value=9)
                            elif k == structure_grid_size-1:
                                structure_grid_amp_interpolation1[i, j, k] = nearest_cube(i, j-1, k-1, structure_grid_amp_interpolation, error_value=9)
                    elif i == 0:
                        if 0<j<structure_grid_size-1:
                            if 0<k<structure_grid_size-1:
                                structure_grid_amp_interpolation1[i, j, k] = nearest_cube(i+1, j, k, structure_grid_amp_interpolation, error_value=9)
                            elif k == 0:
                                structure_grid_amp_interpolation1[i, j, k] = nearest_cube(i+1, j, k+1, structure_grid_amp_interpolation, error_value=9)
                            elif k == structure_grid_size - 1:
                                structure_grid_amp_interpolation1[i, j, k] = nearest_cube(i+1, j, k-1, structure_grid_amp_interpolation, error_value=9)
                        elif j == 0:
                            if 0<k<structure_grid_size-1:
                                structure_grid_amp_interpolation1[i, j, k] = nearest_cube(i+1, j+1, k, structure_grid_amp_interpolation, error_value=9)
                            elif k == 0:
                                structure_grid_amp_interpolation1[i, j, k] = nearest_cube(i + 1, j + 1, k + 1,structure_grid_amp_interpolation, error_value=9)
                            elif k == structure_grid_size-1:
                                structure_grid_amp_interpolation1[i, j, k] = nearest_cube(i+1, j+1, k-1, structure_grid_amp_interpolation, error_value=9)
                        elif j == structure_grid_size-1:
                            if 0<k<structure_grid_size-1:
                                structure_grid_amp_interpolation1[i, j, k] = nearest_cube(i+1, j-1, k, structure_grid_amp_interpolation, error_value=9)
                            elif k == 0:
                                structure_grid_amp_interpolation1[i, j, k] = nearest_cube(i+1, j-1, k+1, structure_grid_amp_interpolation, error_value=9)
                            elif k == structure_grid_size-1:
                                structure_grid_amp_interpolation1[i, j, k] = nearest_cube(i+1, j-1, k-1, structure_grid_amp_interpolation, error_value=9)
                    elif i == structure_grid_size-1:
                        if 0<j<structure_grid_size-1:
                            if 0<k<structure_grid_size-1:
                                structure_grid_amp_interpolation1[i, j, k] = nearest_cube(i-1, j, k, structure_grid_amp_interpolation, error_value=9)
                            elif k == 0:
                                structure_grid_amp_interpolation1[i, j, k] = nearest_cube(i-1, j, k+1, structure_grid_amp_interpolation, error_value=9)
                            elif k == structure_grid_size - 1:
                                structure_grid_amp_interpolation1[i, j, k] = nearest_cube(i-1, j, k-1, structure_grid_amp_interpolation, error_value=9)
                        elif j == 0:
                            # structure_grid_amp_interpolation1[i, j, k] = structure_grid_amp_interpolation[
                            #     i + 1, j + 1, k]
                            if 0<k<structure_grid_size-1:
                                structure_grid_amp_interpolation1[i, j, k] = nearest_cube(i-1, j+1, k, structure_grid_amp_interpolation, error_value=9)
                            elif k == 0:
                                structure_grid_amp_interpolation1[i, j, k] = nearest_cube(i - 1, j + 1, k + 1,structure_grid_amp_interpolation, error_value=9)
                            elif k == structure_grid_size-1:
                                structure_grid_amp_interpolation1[i, j, k] = nearest_cube(i-1, j+1, k-1, structure_grid_amp_interpolation, error_value=9)
                        elif j == structure_grid_size-1:
                            if 0<k<structure_grid_size-1:
                                structure_grid_amp_interpolation1[i, j, k] = nearest_cube(i-1, j-1, k, structure_grid_amp_interpolation, error_value=9)
                            elif k == 0:
                                structure_grid_amp_interpolation1[i, j, k] = nearest_cube(i-1, j-1, k+1, structure_grid_amp_interpolation, error_value=9)
                            elif k == structure_grid_size-1:
                                structure_grid_amp_interpolation1[i, j, k] = nearest_cube(i-1, j-1, k-1, structure_grid_amp_interpolation, error_value=9)
                if structure_grid_cls1[i, j, k] == 9:
                    if 0<i<structure_grid_size-1:
                        if 0<j<structure_grid_size-1:
                            if 0<k<structure_grid_size-1:
                                structure_grid_cls1[i, j, k] = nearest_cube(i, j, k, structure_grid_cls, error_value=9)
                            elif k == 0:
                                structure_grid_cls1[i, j, k] = nearest_cube(i, j, k+1, structure_grid_cls, error_value=9)
                            elif k == structure_grid_size-1:
                                structure_grid_cls1[i, j, k] = nearest_cube(i, j, k-1, structure_grid_cls, error_value=9)
                        elif j == 0:
                            if 0<k<structure_grid_size-1:
                                structure_grid_cls1[i, j, k] = nearest_cube(i, j+1, k, structure_grid_cls, error_value=9)
                            elif k == 0:
                                structure_grid_cls1[i, j, k] = nearest_cube(i, j+1, k+1, structure_grid_cls, error_value=9)
                            elif k == structure_grid_size-1:
                                structure_grid_cls1[i, j, k] = nearest_cube(i, j+1, k-1, structure_grid_cls, error_value=9)
                        elif j == structure_grid_size-1:
                            if 0<k<structure_grid_size-1:
                                structure_grid_cls1[i, j, k] = nearest_cube(i, j-1, k, structure_grid_cls, error_value=9)
                            elif k == 0:
                                structure_grid_cls1[i, j, k] = nearest_cube(i, j-1, k+1, structure_grid_cls, error_value=9)
                            elif k == structure_grid_size-1:
                                structure_grid_cls1[i, j, k] = nearest_cube(i, j-1, k-1, structure_grid_cls, error_value=9)
                    elif i == 0:
                        if 0<j<structure_grid_size-1:
                            if 0<k<structure_grid_size-1:
                                structure_grid_cls1[i, j, k] = nearest_cube(i+1, j, k, structure_grid_cls, error_value=9)
                            elif k == 0:
                                structure_grid_cls1[i, j, k] = nearest_cube(i+1, j, k+1, structure_grid_cls, error_value=9)
                            elif k == structure_grid_size - 1:
                                structure_grid_cls1[i, j, k] = nearest_cube(i+1, j, k-1, structure_grid_cls, error_value=9)
                        elif j == 0:
                            if 0<k<structure_grid_size-1:
                                structure_grid_cls1[i, j, k] = nearest_cube(i+1, j+1, k, structure_grid_cls, error_value=9)
                            elif k == 0:
                                structure_grid_cls1[i, j, k] = nearest_cube(i + 1, j + 1, k + 1,structure_grid_cls, error_value=9)
                            elif k == structure_grid_size-1:
                                structure_grid_cls1[i, j, k] = nearest_cube(i+1, j+1, k-1, structure_grid_cls, error_value=9)
                        elif j == structure_grid_size-1:
                            if 0<k<structure_grid_size-1:
                                structure_grid_cls1[i, j, k] = nearest_cube(i+1, j-1, k, structure_grid_cls, error_value=9)
                            elif k == 0:
                                structure_grid_cls1[i, j, k] = nearest_cube(i+1, j-1, k+1, structure_grid_cls, error_value=9)
                            elif k == structure_grid_size-1:
                                structure_grid_cls1[i, j, k] = nearest_cube(i+1, j-1, k-1, structure_grid_cls, error_value=9)
                    elif i == structure_grid_size-1:
                        if 0<j<structure_grid_size-1:
                            if 0<k<structure_grid_size-1:
                                structure_grid_cls1[i, j, k] = nearest_cube(i-1, j, k, structure_grid_cls, error_value=9)
                            elif k == 0:
                                structure_grid_cls1[i, j, k] = nearest_cube(i-1, j, k+1, structure_grid_cls, error_value=9)
                            elif k == structure_grid_size - 1:
                                structure_grid_cls1[i, j, k] = nearest_cube(i-1, j, k-1, structure_grid_cls, error_value=9)
                        elif j == 0:
                            # structure_grid_amp_interpolation1[i, j, k] = structure_grid_amp_interpolation[
                            #     i + 1, j + 1, k]
                            if 0<k<structure_grid_size-1:
                                structure_grid_cls1[i, j, k] = nearest_cube(i-1, j+1, k, structure_grid_cls, error_value=9)
                            elif k == 0:
                                structure_grid_cls1[i, j, k] = nearest_cube(i - 1, j + 1, k + 1,structure_grid_cls, error_value=9)
                            elif k == structure_grid_size-1:
                                structure_grid_cls1[i, j, k] = nearest_cube(i-1, j+1, k-1, structure_grid_cls, error_value=9)
                        elif j == structure_grid_size-1:
                            if 0<k<structure_grid_size-1:
                                structure_grid_cls1[i, j, k] = nearest_cube(i-1, j-1, k, structure_grid_cls, error_value=9)
                            elif k == 0:
                                structure_grid_cls1[i, j, k] = nearest_cube(i-1, j-1, k+1, structure_grid_cls, error_value=9)
                            elif k == structure_grid_size-1:
                                structure_grid_cls1[i, j, k] = nearest_cube(i-1, j-1, k-1, structure_grid_cls, error_value=9)
    X_1 = np.arange(structure_grid_size)
    Y_1 = np.arange(structure_grid_size)
    Z_1 = np.arange(structure_grid_size)
    XX1, YY1, ZZ1 = np.meshgrid(X_1, Y_1, Z_1)
    X_2 = np.arange(structure_grid_size - 2)
    Y_2 = np.arange(structure_grid_size - 2)
    Z_2 = np.arange(structure_grid_size - 2)
    XX2, YY2, ZZ2 = np.meshgrid(X_2, Y_2, Z_2)
    #############################################################
    print('datetime_now:', datetime.now())
    return YY1.flatten(), XX1.flatten(), ZZ1.flatten(), structure_grid_r.flatten(), structure_grid_cls1.flatten(), structure_grid_amp_interpolation1.flatten(), np.moveaxis(
        np.vstack([YY2.flatten(), XX2.flatten(), ZZ2.flatten()]), -1, 0)
def model_interpolation2(file_path):
    wavelet = RickerWavelet()
    output_cube = np.zeros((140, 140, 140))
    counter = 0
    counter_1 = 0
    file_data = np.moveaxis(np.loadtxt(file_path), -1, 0)
    X, Y, Z, r, am = file_data[0, :], file_data[1, :], file_data[2, :], file_data[3, :], file_data[6, :]
    X_1, Y_1, Z_1, r_1 = [], [], [], []
    X_3, Y_3, Z_3, am2 = [], [], [], []
    for i in range(X.shape[0]):
        if 20<X[i]<160 and 20<Y[i]<160 and 20<Z[i]<160:
            counter += 1
            X_3.append(X[i])
            Y_3.append(Y[i])
            Z_3.append(Z[i])
            am2.append(am[i])
            if r[i]!=0:
                counter_1 += 1
                x = int((X[i] - 20) // 1)
                y = int((Y[i] - 20) // 1)
                z = int((Z[i] - 20) // 1)
                output_cube[x, y, z] = r[i]
                X_1.append(X[i])
                Y_1.append(Y[i])
                Z_1.append(Z[i])
                r_1.append(r[i])
                if z<139 and y<139 and x<139:
                    output_cube[x, y, z+1] = r[i]
                    output_cube[x, y+1, z] = r[i]
                    output_cube[x, y+1, z+1] = r[i]
                    output_cube[x+1, y, z] = r[i]
                    output_cube[x+1, y, z+1] = r[i]
                    output_cube[x+1, y+1, z] = r[i]
                    output_cube[x+1, y+1, z+1] = r[i]
                # if 0<z<139:
                #     output_cube[x, y, z-1] = r[i]
                #     output_cube[x, y, z+1] = r[i]
                # if 0<x<139:
                #     output_cube[x-1, y, z] = r[i]
                #     output_cube[x+1, y, z] = r[i]
                # if 0<y<139:
                #     output_cube[x, y-1, z] = r[i]
                #     output_cube[x, y+1, z] = r[i]
    print('counter:\n', counter, '\ncounter1:\n', counter_1)
    # output_cube1 = output_cube[::5, ::5, ::5]
    # print(output_cube1.shape)
    amp = WaveletConvolve(wavelet, output_cube)
    X_2, Y_2, Z_2, r_2 = [], [], [], []
    for i in range(140):
        for j in range(140):
            for k in range(140):
                if output_cube[i, j, k]!=0:
                    X_2.append(i)
                    Y_2.append(j)
                    Z_2.append(k)
                    r_2.append(output_cube[i, j, k])
    X_4, Y_4, Z_4 = [], [], []
    for i in range(140):
        for j in range(140):
            for k in range(140):
                X_4.append(i)
                Y_4.append(j)
                Z_4.append(k)

    # X1 = np.linspace(0, 140, 140)
    # Y1 = np.linspace(0, 140, 140)
    # Z1 = np.linspace(0, 140, 140)
    X1 = np.arange(140)
    Y1 = np.arange(140)
    Z1 = np.arange(140)
    X2, Y2, Z2 = np.meshgrid(X1, Y1, Z1)
    p = pv.Plotter(shape=(2, 2))
    p.subplot(0, 0)
    points1 = np.moveaxis(np.vstack([np.array(X_1), np.array(Y_1), np.array(Z_1)]), -1, 0)
    point_cloud1 = pv.PolyData(points1)
    point_cloud1['v1'] = np.array(r_1)
    _ = p.add_mesh(point_cloud1)
    p.subplot(0, 1)
    points2 = np.moveaxis(np.vstack([np.array(X_2), np.array(Y_2), np.array(Z_2)]), -1, 0)
    point_cloud2 = pv.PolyData(points2)
    point_cloud2['v2'] = np.array(r_2)
    _ = p.add_mesh(point_cloud2)
    p.subplot(1, 0)
    points3 = np.moveaxis(np.vstack([np.array(X_3), np.array(Y_3), np.array(Z_3)]), -1, 0)
    point_cloud3 = pv.PolyData(points3)
    point_cloud3['v3'] = np.array(am2)
    _ = p.add_mesh(point_cloud3)
    # points4 = np.moveaxis(np.vstack([X2.flatten(), Y2.flatten(), Z2.flatten()]), -1, 0)
    # point_cloud4 = pv.PolyData(points4)
    # point_cloud4['v4'] = output_cube.flatten()
    # _ = p.add_mesh(point_cloud4)
    p.subplot(1, 1)
    # points4 = np.moveaxis(np.vstack([X2.flatten(), Y2.flatten(), Z2.flatten()]), -1, 0)
    # point_cloud4 = pv.PolyData(points4)
    # point_cloud4['v4'] = amp.flatten()
    # _ = p.add_mesh(point_cloud4)
    points4 = np.moveaxis(np.vstack([np.array(X_4), np.array(Y_4), np.array(Z_4)]), -1, 0)
    point_cloud4 = pv.PolyData(points4)
    point_cloud4['v4'] = amp.flatten()
    _ = p.add_mesh(point_cloud4)
    p.show()
    return
def RickerWavelet():
    dt = np.arange(-0.02, 0.02, 0.002)
    fm = 45
    wavelet = (1 - 2 * (math.pi * fm * dt) ** 2) * np.exp(-1 * (math.pi * fm * dt) ** 2)
    return wavelet
if __name__ == '__main__':
    file_data = np.moveaxis(np.loadtxt('1.txt'), -1, 0)
    X, Y, Z, r, am = file_data[0, :], file_data[1, :], file_data[2, :], file_data[3, :], file_data[6, :]
    X1, Y1, Z1, r1, cls1, amp_inter, grid = model_structure_grid1(X, Y, Z, r, am)
    print(type(X1))
    print(type(Y1))
    print(type(Z1))
    print(type(r1))
    print(type(amp_inter))
    points = np.moveaxis(np.vstack([X1, Y1, Z1]), -1, 0)
    p = pv.Plotter(shape=(1, 3))
    p.subplot(0, 0)
    point_cloud = pv.PolyData(points)
    point_cloud['r'] = r1
    _ = p.add_mesh(point_cloud)
    p.subplot(0, 1)
    point_cloud2 = pv.PolyData(grid)
    point_cloud2['amp_inside'] = ((amp_inter.reshape((140, 140, 140)))[1:-1, 1:-1, 1:-1]).flatten()
    _ = p.add_mesh(point_cloud2)
    p.subplot(0, 2)
    point_cloud3 = pv.PolyData(points)
    point_cloud3['amp_inter'] = amp_inter
    _ = p.add_mesh(point_cloud3)
    p.show()
    # model_interpolation2('1.txt')