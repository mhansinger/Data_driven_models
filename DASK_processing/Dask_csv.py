#!/usr/bin/env python
# coding: utf-8

# Read in the parquet files and compute one large database for the given UPRIME

import pandas as pd
import numpy as np
import os
from os.path import join
import dask
import dask.dataframe as dd
import dask.array as da
import random 


path_to_data = '/media/max/HDD3/DNS_Data/Planar/NX512/UPRIME5/postProcess_DNN'

files_list = os.listdir(path_to_data)

files_list = [f for f in files_list if f.endswith('test.csv')]

# shuffle the files list
random.shuffle(files_list)

#files_list.remove('filter_width_8_DNN_train.parquet')

for f in files_list:

    print("Reading file: ",f)

    #data_pq = pd.read_parquet(join(path_to_data,'filter_width_*_DNN_train.parquet'),chunksize='2GB') #CHUNKSIZE matters??
    #data_pq = pd.read_parquet(join(path_to_data,'small_*.parquet'),chunksize='1MB')

    data_pq = pd.read_csv(join(path_to_data,f)) #CHUNKSIZE matters??

    # print('Convert to single precission and sample')
    # data_pq = data_pq.astype(np.float32).sample(frac=0.3)


    # ## compute the additional quantites (tensors)

    # compute tensors R, S mag(U) etc.

    mag_U = np.sqrt(data_pq['U_bar'].values**2 + data_pq['V_bar'].values**2 +data_pq['W_bar'].values**2)
    mag_grad_c = np.sqrt(data_pq['grad_c_x_LES'].values**2 + data_pq['grad_c_y_LES'].values**2 +data_pq['grad_c_z_LES'].values**2)

    sum_U = data_pq['U_bar'].values + data_pq['V_bar']+data_pq['W_bar'].values
    sum_c = np.absolute(data_pq['grad_c_x_LES'].values) + np.absolute(data_pq['grad_c_y_LES'].values) +np.absolute(data_pq['grad_c_z_LES'].values)

    grad_U = np.sqrt(data_pq['grad_U_x_LES'].values**2 + data_pq['grad_U_y_LES'].values**2 +data_pq['grad_U_z_LES'].values**2)
    grad_V = np.sqrt(data_pq['grad_V_x_LES'].values**2 + data_pq['grad_V_y_LES'].values**2 +data_pq['grad_V_z_LES'].values**2)
    grad_W = np.sqrt(data_pq['grad_W_x_LES'].values**2 + data_pq['grad_W_y_LES'].values**2 +data_pq['grad_W_z_LES'].values**2)

    mag_grad_U = np.sqrt(grad_U**2 + grad_V**2 +grad_W**2)
    sum_grad_U = np.absolute(grad_U) + np.absolute(grad_V) +np.absolute(grad_W)

    print('Computing gradient_tensor')

    gradient_tensor = np.array([
                        [data_pq['grad_U_x_LES'],data_pq['grad_V_x_LES'],data_pq['grad_W_x_LES']],
                        [data_pq['grad_U_y_LES'],data_pq['grad_V_y_LES'],data_pq['grad_W_y_LES']],
                        [data_pq['grad_U_z_LES'],data_pq['grad_V_z_LES'],data_pq['grad_W_z_LES']]
                        ])



    print('Computing S and R')
    # symetric strain
    Strain = 0.5*(gradient_tensor + np.transpose(gradient_tensor,(1,0,2)))
    #anti symetric strain
    Anti =  0.5*(gradient_tensor - np.transpose(gradient_tensor,(1,0,2)))


    print('Computing lambdas')

    lambda_1 = np.trace(Strain**2)
    lambda_2 = np.trace(Anti**2)
    lambda_3 = np.trace(Strain**3)
    lambda_4 = np.trace(Anti**2 * Strain)
    lambda_5 = np.trace(Anti**2 * Strain**2)


    # Add to the dask dataframe

    data_pq['mag_grad_c'] = mag_grad_c
    data_pq['mag_U'] = mag_U
    data_pq['sum_c'] = sum_c
    data_pq['sum_U'] = sum_U
    data_pq['sum_grad_U'] = sum_grad_U
    data_pq['mag_grad_U'] = mag_grad_U

    # REPARTITON
    #data_pq = data_pq.repartition(npartitions=lambda_1.npartitions)

    data_pq['lambda_1'] = lambda_1
    data_pq['lambda_2'] = lambda_2
    data_pq['lambda_3'] = lambda_3
    data_pq['lambda_4'] = lambda_4
    data_pq['lambda_5'] = lambda_5

    print('Done with feature computation')

    # if file == files_list[0]:
    #     data_all = data_pq.copy()
    # else:
    #     data_all = data_all.append(data_pq)
    #     #data_all = data_all.reset_index().drop('index', axis=1)


    # write data to file
    print("Writing: ",f)
    #data_pq.to_csv(join(path_to_data,'ALL',f),index=False)
    data_pq.to_parquet(join(path_to_data,'ALL',f.split('.')[0]+'.parquet'),index=False)


    print(" ")

    # print('Computing the statistics')
    # #
    # # reindex
    # data_pq = data_pq.reset_index().drop('index',axis=1)

    # data_mean, data_std = dask.compute(data_pq.mean(),data_pq.std())