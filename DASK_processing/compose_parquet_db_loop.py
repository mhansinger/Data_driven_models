#join the single parquet files to one large DB for the training

# this works: Oct 2020data_d    

import numpy as np
import dask.dataframe as dd
import pandas as pd
from os.path import join
import dask
import dask.array as da
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse


def compose_db(case,test_train_set):
    '''

    :param case: UPRIMEXY case
    :param test_train_set: test or train set
    :return:
    '''

    path_to_data = '/media/max/HDD3/DNS_Data/Planar/NX512/'+case+'/postProcess_DNN/ALL'

    # read in the data as dask dataframe
    data_pq = dd.read_parquet(join(path_to_data,'filter_width_*_DNN_'+case+'_'+test_train_set+'.parquet'),chunksize='2GB')
    #data_pq = dd.read_parquet(join(path_to_data,'train_example_*.parquet'))

    # now sample
    files = 50

    data_pq=data_pq.sample(frac=0.5).astype(np.float32)

    # get the mean and STD of the dataset
    data_mean, data_std = dask.compute(data_pq.mean(),data_pq.std())

    #assign variables to dataFrame
    moments = pd.DataFrame(data_mean,columns=['mean'])
    moments['std'] = data_std

    # write down moments file
    moments.to_csv(join(path_to_data,'moments_'+case+'.csv'))
    print('moments are written.')

    for i in tqdm(range(1,files+1),desc='Computing ...',ncols=75):

        data_pq = data_pq.repartition(npartitions=1)

        #sample and change format
        data_df = data_pq.sample(frac=1/files).compute()

        # sample again to loose dask indexing
        data_df = data_df.sample(frac=1.0).astype(np.float32).drop('Unnamed: 0',axis=1)

        # write the data base
        filename=join(path_to_data,test_train_set+'_'+case+'_'+str(i))

        ## PARQUET
        data_df.to_parquet(filename+'.parquet')

        # ## PICKLE
        # data_pq.to_pickle(filename + '.pickle')
        #
        # ## PICKLE
        # data_pq.to_csv(filename + '.csv')
        #
        # ## HDF
        # data_pq.to_hdf(filename + '.hdf',key='data')

        #plt.scatter(data_df.c_bar,data_df.omega_DNS_filtered,s=0.5)
        #print(data_df.Delta_LES)

    #plt.show()

    print('\nAll done. Bye.\n')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="case: UPRIME5, UPRIME75 .., test_train_set: train or test")
    parser.add_argument('--test_train_set',type=str)
    parser.add_argument('--case',type=str)
    args = parser.parse_args()

    # run the function
    compose_db(args.case,args.test_train_set)
