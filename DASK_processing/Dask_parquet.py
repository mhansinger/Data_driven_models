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


# In[ ]:


path_to_data = '/media/max/HDD3/DNS_Data/Planar/NX512/UPRIME5/postProcess_DNN'


# In[ ]:


data_pq = dd.read_parquet(join(path_to_data,'filter_width_*_DNN_train.parquet'))


# In[ ]:
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

gradient_tensor = da.array([
                    [data_pq['grad_U_x_LES'],data_pq['grad_V_x_LES'],data_pq['grad_W_x_LES']],
                    [data_pq['grad_U_y_LES'],data_pq['grad_V_y_LES'],data_pq['grad_W_y_LES']],
                    [data_pq['grad_U_z_LES'],data_pq['grad_V_z_LES'],data_pq['grad_W_z_LES']]
                    ])

# In[ ]:
# symetric strain
Strain = 0.5*(gradient_tensor + da.transpose(gradient_tensor,(1,0,2)))
#anti symetric strain
Anti =  0.5*(gradient_tensor - da.transpose(gradient_tensor,(1,0,2)))


# In[ ]:

lambda_1 = da.trace(Strain**2)
lambda_2 = da.trace(Anti**2)
lambda_3 = da.trace(Strain**3)
lambda_4 = da.trace(Anti**2 * Strain)
lambda_5 = da.trace(Anti**2 * Strain**2)


# In[ ]:

data_pq['mag_grad_c'] = mag_grad_c
data_pq['mag_U'] = mag_U
data_pq['sum_c'] = sum_c
data_pq['sum_U'] = sum_U
data_pq['sum_grad_U'] = sum_grad_U
data_pq['mag_grad_U'] = mag_grad_U

data_pq['lambda_1'] = lambda_1
data_pq['lambda_2'] = lambda_2
data_pq['lambda_3'] = lambda_3
data_pq['lambda_4'] = lambda_4
data_pq['lambda_5'] = lambda_5


# In[ ]:



