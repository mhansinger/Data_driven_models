#join the single parquet files to one large DB for the training

# this works: Oct 2020

import numpy as np
import dask.dataframe as dd
import pandas as pd
from os.path import join
import dask
import dask.array as da
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse


def compose_db(case,train_test_set,scaler):
    '''

    :param case: UPRIMEXY case
    :param train_test_set: test or train set
    :param scaler: Standard, or Log and Standard scaler
    :return:
    '''

    path_to_data = '/media/max/HDD3/DNS_Data/Planar/NX512/'+case+'/postProcess_DNN/'

    # read in the data as dask dataframe
    data_pq = dd.read_parquet(join(path_to_data,'filter_width_*_DNN_'+case+'_'+train_test_set+'.parquet'),chunksize='2GB')
    #data_pq = dd.read_parquet(join(path_to_data,'train_example_*.parquet'))

    # no of output files

    if train_test_set=='test':
        files = 1
        FOLDER='TEST'
    else:
        files = 10
        FOLDER='TRAIN'

    # # remove columns which are not used for training as they have spatial information (direction)
    # columns_to_remove=['U_bar', 'V_bar', 'W_bar','grad_c_x_LES', 'grad_c_y_LES', 'grad_c_z_LES', 'grad_U_x_LES',
    #    'grad_V_x_LES', 'grad_W_x_LES', 'grad_U_y_LES', 'grad_V_y_LES',
    #    'grad_W_y_LES', 'grad_U_z_LES', 'grad_V_z_LES', 'grad_W_z_LES']

    # data_pq=data_pq.drop(columns_to_remove,axis=1).astype(np.float32)

    # check if Log scaler
    if scaler=='Log':
        print('\nApplying Log Transformation to the data set')
        data_pq = data_pq.apply(np.log,axis=1)

    # get the mean and STD of the dataset
    data_mean, data_std = dask.compute(data_pq.mean(),data_pq.std())

    #assign variables to dataFrame
    moments = pd.DataFrame(data_mean,columns=['mean'])
    moments['std'] = data_std

    # write down moments file
    if train_test_set=='train':
        if scaler=='Log':
            print('\nComputing moments for Log transformed...')
            moments.to_csv(join(path_to_data, 'moments_' + case + '_Log.csv'))
        elif scaler == 'Standard':
            print('\nComputing moments for Standard Scaler...')
            moments.to_csv(join(path_to_data, 'moments_' + case + '.csv'))
        print('Moments are written.')
    else:
        print('\nNo moments computed for test set ...')

    for i in tqdm(range(1,files+1),desc='Computing ...',ncols=75):

        data_pq = data_pq.repartition(npartitions=1)

        #sample and change format
        data_df = data_pq.sample(frac=1/files).compute()

        # sample again to loose dask indexing
        data_df = data_df.sample(frac=1.0).astype(np.float32)#.drop('Unnamed: 0',axis=1)

        # write the data base
        filename=join(path_to_data,FOLDER,train_test_set+'_'+case+'_'+str(i))

        ## PARQUET
        if scaler=='Log':
            data_df.to_parquet(filename+'_Log.parquet')
        elif scaler=='Standard':
            data_df.to_parquet(filename + '.parquet')

    #plt.show()

    print('\nAll done. Bye.\n')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="case: UPRIME5, UPRIME75 .., train_test_set: train or test")
    parser.add_argument('--train_test_set',type=str)
    parser.add_argument('--case',type=str)
    parser.add_argument('--scaler', type=str) #TODO: Standard, Log, maybe more...
    args = parser.parse_args()

    # run the function
    compose_db(args.case,args.train_test_set,args.scaler)
