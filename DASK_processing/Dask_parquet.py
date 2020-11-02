#!/usr/bin/env python
# coding: utf-8

# Read in the parquet files and compute one large database for the given UPRIME

import pandas as pd
import numpy as np
import tensorflow as tf
import os
from os.path import join
import dask
import dask.dataframe as dd
import dask.array as da
import random 
import argparse



def compose_parquet(case,test_train_set):

    path_to_data = '/media/max/HDD3/DNS_Data/Planar/NX512/'+case+'/postProcess_DNN'

    files_list = os.listdir(path_to_data)

    files_list = [f for f in files_list if f.endswith(test_train_set+'.parquet')]

    # shuffle the files list
    random.shuffle(files_list)


    for f in files_list:

        print("Reading file: ",f)

        #data_pq = dd.read_parquet(join(path_to_data,'filter_width_*_DNN_train.parquet'),chunksize='2GB') #CHUNKSIZE matters??
        #data_pq = dd.read_parquet(join(path_to_data,'small_*.parquet'),chunksize='1MB')

        data_pq = dd.read_parquet(join(path_to_data,f),chunksize='2GB') #CHUNKSIZE matters??

        # ## compute the additional quantites (tensors)

        # compute tensors R, S mag(U) etc.

        mag_U = da.sqrt(data_pq['U_bar'].values**2 + data_pq['V_bar'].values**2 +data_pq['W_bar'].values**2)
        mag_grad_c = da.sqrt(data_pq['grad_c_x_LES'].values**2 + data_pq['grad_c_y_LES'].values**2 +data_pq['grad_c_z_LES'].values**2)

        sum_U = data_pq['U_bar'].values + data_pq['V_bar']+data_pq['W_bar'].values
        sum_c = da.absolute(data_pq['grad_c_x_LES'].values) + da.absolute(data_pq['grad_c_y_LES'].values) +da.absolute(data_pq['grad_c_z_LES'].values)

        grad_U = da.sqrt(data_pq['grad_U_x_LES'].values**2 + data_pq['grad_U_y_LES'].values**2 +data_pq['grad_U_z_LES'].values**2)
        grad_V = da.sqrt(data_pq['grad_V_x_LES'].values**2 + data_pq['grad_V_y_LES'].values**2 +data_pq['grad_V_z_LES'].values**2)
        grad_W = da.sqrt(data_pq['grad_W_x_LES'].values**2 + data_pq['grad_W_y_LES'].values**2 +data_pq['grad_W_z_LES'].values**2)

        mag_grad_U = da.sqrt(grad_U**2 + grad_V**2 +grad_W**2)
        sum_grad_U = da.absolute(grad_U) + da.absolute(grad_V) +da.absolute(grad_W)

        print('Computing gradient_tensor')

        gradient_tensor = da.array([
                            [data_pq['grad_U_x_LES'],data_pq['grad_V_x_LES'],data_pq['grad_W_x_LES']],
                            [data_pq['grad_U_y_LES'],data_pq['grad_V_y_LES'],data_pq['grad_W_y_LES']],
                            [data_pq['grad_U_z_LES'],data_pq['grad_V_z_LES'],data_pq['grad_W_z_LES']]
                            ])



        print('Computing S and R')
        # symetric strain
        Strain = 0.5*(gradient_tensor + da.transpose(gradient_tensor,(1,0,2)))
        #anti symetric strain
        Anti =  0.5*(gradient_tensor - da.transpose(gradient_tensor,(1,0,2)))


        print('Computing lambdas')

        lambda_1 = da.trace(Strain**2)
        lambda_2 = da.trace(Anti**2)
        lambda_3 = da.trace(Strain**3)
        lambda_4 = da.trace(Anti**2 * Strain)
        lambda_5 = da.trace(Anti**2 * Strain**2)


        # Add to the dask dataframe

        data_pq['mag_grad_c'] = mag_grad_c
        data_pq['mag_U'] = mag_U
        data_pq['sum_c'] = sum_c
        data_pq['sum_U'] = sum_U
        data_pq['sum_grad_U'] = sum_grad_U
        data_pq['mag_grad_U'] = mag_grad_U

        # REPARTITON
        data_pq = data_pq.repartition(npartitions=lambda_1.npartitions)

        data_pq['lambda_1'] = lambda_1
        data_pq['lambda_2'] = lambda_2
        data_pq['lambda_3'] = lambda_3
        data_pq['lambda_4'] = lambda_4
        data_pq['lambda_5'] = lambda_5

        print('Done with feature computation')

        # write data to file
        print("Writing: ",f)
        dd.to_parquet(data_pq,join(path_to_data,'ALL',f),write_index=False)

        print(" ")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="case: UPRIME5, UPRIME75 .., test_train_set: train or test")
    parser.add_argument('--test_train_set',type=str)
    parser.add_argument('--case',type=str)
    args = parser.parse_args()

    # run the function
    compose_parquet(args.case,args.test_train_set)