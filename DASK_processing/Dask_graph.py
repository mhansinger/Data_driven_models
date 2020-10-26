#!/usr/bin/env python
# coding: utf-8

# Read in the parquet files and compute one large database for the given UPRIME

import pandas as pd
import numpy as np
import tensorflow as tf
import os
from os.path import join
import dask.dataframe as dd
import dask.array as da

# dask distributed
from distributed import Client, LocalCluster


# In[ ]:


path_to_data = '/media/max/HDD3/DNS_Data/Planar/NX512/UPRIME5/postProcess_DNN'


# In[ ]:


data_pq = dd.read_parquet(join(path_to_data,'filter_width_*_DNN_train.parquet'))

data_pq = data_pq.sample(frac=0.3)


# compute tensors R, S mag(U) etc.

def compute_mag_U(data):
    return da.sqrt(data_pq['U_bar'].values**2 + data_pq['V_bar'].values**2 +data_pq['W_bar'].values**2)

def compute_mag_grad_c(data):
    return da.sqrt(data_pq['grad_c_x_LES'].values**2 + data_pq['grad_c_y_LES'].values**2 +data_pq['grad_c_z_LES'].values**2)

def compute_sum_U(data):
    return data_pq['U_bar'].values + data_pq['V_bar']+data_pq['W_bar'].values

def compute_sum_c(data):
    return da.absolute(data_pq['grad_c_x_LES'].values) + da.absolute(data_pq['grad_c_y_LES'].values) +da.absolute(data_pq['grad_c_z_LES'].values)

def compute_grad_U(data):
    return da.sqrt(data_pq['grad_U_x_LES'].values**2 + data_pq['grad_U_y_LES'].values**2 +data_pq['grad_U_z_LES'].values**2)

def compute_grad_V(data):
    return da.sqrt(data_pq['grad_V_x_LES'].values**2 + data_pq['grad_V_y_LES'].values**2 +data_pq['grad_V_z_LES'].values**2)

def compute_grad_W(data):
    return da.sqrt(data_pq['grad_W_x_LES'].values**2 + data_pq['grad_W_y_LES'].values**2 +data_pq['grad_W_z_LES'].values**2)

def compute_mag_grad_U(grad_U,grad_V,grad_W):
    return da.sqrt(grad_U**2 + grad_V**2 +grad_W**2)

def compute_sum_grad_U(grad_U,grad_V,grad_W):
    return da.absolute(grad_U) + da.absolute(grad_V) +da.absolute(grad_W)


def compute_gradient_tensor(data):
    return da.array([
                    [data_pq['grad_U_x_LES'].compute(),data_pq['grad_V_x_LES'].compute(),data_pq['grad_W_x_LES'].compute()],
                    [data_pq['grad_U_y_LES'].compute(),data_pq['grad_V_y_LES'].compute(),data_pq['grad_W_y_LES'].compute()],
                    [data_pq['grad_U_z_LES'].compute(),data_pq['grad_V_z_LES'].compute(),data_pq['grad_W_z_LES'].compute()]
                    ],dtype=np.float32)

# In[ ]:

# symetric strain
def compute_Strain(tensor):
    return 0.5*(tensor + da.transpose(tensor,(1,0,2)))

#anti symetric strain
def compute_Anti(tensor):
    return  0.5*(tensor - da.transpose(tensor,(1,0,2)))


# In[ ]:

def compute_lambda_1(Strain):
    return da.trace(Strain**2)

def compute_lambda_2(Anti):
    return da.trace(Anti**2)

def compute_lambda_3(Strain):
    return da.trace(Strain**3)

def compute_lambda_4(Strain,Anti):
    return da.trace(Anti**2 * Strain)

def compute_lambda_5(Strain,Anti):
    return da.trace(Anti**2 * Strain**2)

gradient_tensor = compute_gradient_tensor(data_pq)


#%%
# print('Set up client and connect to scheduler\n')
#
# client = Client('tcp://137.193.236.40:8786')
#
# print('Set up the graph\n')
#
# # dask graph
# dsk_graph = {'mag_U':(compute_mag_U,data_pq),
#        'mag_grad_c':(compute_mag_grad_c,data_pq),
#        'sum_U': (compute_sum_U,data_pq),
#        'sum_c': (compute_sum_c,data_pq),
#        'grad_U':(compute_grad_U,data_pq),
#        'grad_V':(compute_grad_U,data_pq),
#        'grad_W':(compute_grad_U,data_pq),
#        'mag_grad_U':(compute_mag_grad_U,'grad_U','grad_V','grad_W'),
#        'sum_grad_U':(compute_sum_grad_U,'grad_U','grad_V','grad_W'),
#        'grad_tensor':(compute_gradient_tensor,data_pq),
#        'Strain':(compute_Strain,'grad_tensor'),
#        'Anti':(compute_Anti,'grad_tensor'),
#        'lambda_1':(compute_lambda_1,'Strain'),
#        'lambda_2':(compute_lambda_2,'Anti'),
#        'lambda_3':(compute_lambda_3,'Strain'),
#        'lambda_4': (compute_lambda_4, 'Strain','Anti'),
#        'lambda_5': (compute_lambda_4, 'Strain','Anti'),
#        }
#
# if __name__ =='__main__':
#     #client.persist()
#
#     print('get...')
#     lamda_1= client.get(dsk_graph,'lamda_1')
