
from typing import List

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import os
import xgboost as xgb
from os.path import join
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from tensorflow.keras.regularizers import l1, l2, l1_l2

# from utils.customObjects import coeff_r2, SGDRScheduler
# from utils.normalize_data import normalizeStandard, reTransformStandard, reTransformTarget
# from utils.resBlock import res_block_org

'''
This is to train a XGB classifier for various Delta_LES
'''

##################################
#CASE and parameters
##################################

CASE = 'UPRIME5'

LOSS='mse' #'mse'
lr = 1e-2


# path to the UPRIMEXY data set
path_to_data = '/media/max/HDD3/DNS_Data/Planar/NX512/'+CASE+'/postProcess_DNN'

# read in the moments (mean and std of the data set
moments = pd.read_csv(join(path_to_data,'moments_'+CASE+'_Log.csv'),index_col=0)

XGB_model_path = join('/home/max/Python/Data_driven_models/XGBoost/XGB_'+CASE)

###################################
# which are the features in the data to train for
###################################
FEATURES: List[str] = ['c_bar',  'omega_model_planar', 'UP_delta',
                        'SGS_flux', 'Delta_LES', 'mag_grad_c', 'mag_U', 'sum_c', 'sum_U',
                        'sum_grad_U', 'mag_grad_U', 'lambda_1',  'lambda_3','c_prime','mag_strain','mag_vorticity']

TARGET: List[str] = ['omega_DNS_filtered']

training_files = os.listdir(join(path_to_data,'TRAIN'))
training_files = [f for f in training_files if (f.startswith('train') and f.endswith('Log.parquet'))]
random.shuffle(training_files)

print('Training files: ',training_files)

#retain one set as validation set:
validation_file = training_files.pop(0)


# read in from file
this_train_df = pd.read_parquet(join(path_to_data,'TRAIN',training_files[0]))

# drop rows where NaN
this_train_df = this_train_df.dropna(axis=0)

# # normalize the data set
# normalized_train_df = normalizeStandard(this_train_df,moments)

# TRAIN THE XGB
print('Training the model...')
TREE_METHOD = 'hist'#'gpu_hist'    #'hist'

model = xgb.train({"learning_rate": lr,'tree_method': TREE_METHOD,'gpu_id':0,'max_depth':20},
                  xgb.DMatrix(this_train_df[FEATURES],
                  label=this_train_df[TARGET]),
                  num_boost_round=100)

print('\nDone!')

# TEST XBG
test_files = os.listdir(join(path_to_data,'TEST'))
test_df = pd.read_parquet(join(path_to_data,'TEST',test_files[0]))

DTest = xgb.DMatrix(test_df[FEATURES])

y_hat = model.pr
