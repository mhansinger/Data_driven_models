
from typing import List

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn as sk
import os
from os.path import join
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from utils.customObjects import coeff_r2, SGDRScheduler
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.regularizers import l1, l2, l1_l2

from utils.normalize_data import normalizeStandard, reTransformStandard, reTransformTarget
from utils.resBlock import res_block_org

import argparse

import shap
# print the JS visualization code to the notebook

shap.initjs()

'''
This is to train the Network for various Delta_LES
'''

##################################
# LIMIT GPU MEMORY USAGE
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

# if gpus:
#     try:
#         tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
#     except RuntimeError as e:
#         print(e)

##################################
#CASE and parameters
##################################

# # switch off GPU
# os.environ['CUDA_VISIBLE_DEVICES']="-1"
#
# # DISTRIBUTED
# strategy = tf.distribute.MirroredStrategy()
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

###################################
# Args parser
parser = argparse.ArgumentParser(
    description="Define batch size, learing_rate, scaler...")
parser.add_argument('--BATCH_SIZE', type=int)
parser.add_argument('--LEARNING_RATE', type=float)
parser.add_argument('--SCALER', type=str)
parser.add_argument('--CASE', type=str)
parser.add_argument('--MODE', type=str)
args = parser.parse_args()
###################################


CASE = args.CASE #'UPRIME5'

BATCH_SIZE = args.BATCH_SIZE #3200 #000#64#128
NEURONS = 200
RES_BLOCKS = 10
EPOCHS = 30
LOSS = 'mse' #'mse'
lr = args.LEARNING_RATE #1e-5

# scaler
SCALER = args.SCALER

# print('\n############################')
# print('THIS RUNS WITH LOG TRANSFORMED DATA')
# print('############################\n')

##################################
# PATHS
##################################

# path to the UPRIMEXY data set
path_to_data = '/media/max/HDD3/DNS_Data/Planar/NX512/'+CASE+'/postProcess_DNN'

# read in the moments (mean and std of the data set
if SCALER == 'Log':
    moments = pd.read_csv(join(path_to_data,'moments_'+CASE+'_Log.csv'),index_col=0)
    DNN_model_path = join(
        '/home/max/Python/Data_driven_models/TF2/trained_models/DNN_' + CASE + '_nrns_' + str(NEURONS) + '_blks_' + str(
            RES_BLOCKS) + '_Log.h5')
elif SCALER == 'Standard':
    moments = pd.read_csv(join(path_to_data, 'moments_' + CASE + '.csv'), index_col=0)
    DNN_model_path = join(
        '/home/max/Python/Data_driven_models/TF2/trained_models/DNN_' + CASE + '_nrns_' + str(NEURONS) + '_blks_' +
        str(RES_BLOCKS) + '.h5')


#join(path_to_data,'DNN_'+CASE+'_nrns_'+str(NEURONS)+'_blks_'+str(RES_BLOCKS)+'.h5')

###################################
# which are the features in the data to train for
###################################
FEATURES: List[str] = ['c_bar',
                       'c_tilde',
                       #'omega_model_planar',
                       #'omega_oblique',
                       #'UP_delta',
                       #'SGS_flux',
                       'Delta_LES',
                       #'mag_grad_c_tilde',
                       #'mag_grad_U',
                        #'c_prime',

                        ]

TARGET: List[str] = ['omega_DNS_filtered']

training_files = os.listdir(join(path_to_data,'TRAIN'))

if SCALER=='Log':
    training_files = [f for f in training_files if (f.startswith('train') and f.endswith('Log.parquet'))]
elif SCALER=='Standard':
    training_files = [f for f in training_files if (f.startswith('train') and f.endswith('.parquet') and not f.endswith('Log.parquet'))]

random.shuffle(training_files)

print('Training files: ',training_files)

#retain one set as validation set:
validation_file = training_files.pop(0)

##################################
#validation data set
##################################
validation_df = pd.read_parquet(join(path_to_data,'TRAIN',validation_file))

validation_df = validation_df.dropna(axis=0)

validation_df = validation_df.sample(frac=0.5)

validation_norm_df = normalizeStandard(validation_df,moments)

# compose TF dataset
validation_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(validation_norm_df[FEATURES].values, tf.float32),
            tf.cast(validation_norm_df[TARGET].values, tf.float32)
        )
    )
)

# batch validation dataset
validation_batch=int(len(validation_df)/4)
validation_dataset = validation_dataset.batch(validation_batch)

validation_steps= len(validation_df) / validation_batch


#################################
# PREDICT TEST SET FUNCTION
#################################

# load test set
test_files = os.listdir(join(path_to_data, 'TEST'))

if SCALER=='Log':
    test_files = [f for f in test_files if (f.startswith('test') and f.endswith('Log.parquet'))]
elif SCALER=='Standard':
    test_files = [f for f in test_files if (f.startswith('test') and f.endswith('.parquet'))]

# shuffle the test_files list
random.shuffle(test_files)

test_df = pd.read_parquet(join(path_to_data, 'TEST', test_files[0])).sample(frac=0.3)

test_df_norm = normalizeStandard(test_df, moments)


def predict_test(keep_out=False,THIS_CASE=CASE,test_df = test_df, test_df_norm=test_df_norm):
    print('\n##############################')
    print('Predict on test set')
    print('##############################\n')

    if keep_out is True:
        print('Predicting keep out data')
        path_to_data = '/media/max/HDD3/DNS_Data/Planar/NX512/' + THIS_CASE + '/postProcess_DNN'
        test_df = pd.read_parquet(join(path_to_data, 'KEEP_OUT/filter_width_20_DNN_'+THIS_CASE+'_test.parquet')).sample(frac=0.5)
        # THE KEEP OUT DATA IS NOT YET TRANSFORMED
        test_transformed_df = test_df.apply(np.log)
        test_df_norm = normalizeStandard(test_transformed_df, moments)

    # predict y_hat
    y_hat = DNN.predict(test_df_norm[FEATURES])

    omega_DNS_predict = reTransformTarget(y_hat, moments)

    # Retransform
    if SCALER == 'Log':
        test_df_real = test_df.apply(np.exp)
        omega_DNS_predict_real = omega_DNS_predict.apply(np.exp)
    elif SCALER == 'Standard':
        test_df_real = test_df
        omega_DNS_predict_real = omega_DNS_predict

    # overwrite if keep_out==True
    if keep_out is True:
        test_df_real = test_df

    # plot the results
    plt.figure()
    plt.scatter(test_df_real['c_bar'],test_df_real['omega_DNS_filtered'],s=0.3,c='b')
    plt.scatter(test_df_real['c_bar'],omega_DNS_predict_real,s=0.3,c='r')
    plt.scatter(test_df_real['c_bar'],test_df_real['omega_oblique'],s=0.3,c='k')
    plt.xlabel('c_bar')
    plt.ylabel('omega')
    if keep_out is True:
        plt.title('Prediction on ' + THIS_CASE + ' with model model for ' + CASE + ' on fw=20')
    else:
        plt.title('Prediction on ' + THIS_CASE + ' with model model for ' + CASE)
    plt.savefig('/home/max/Python/Data_driven_models/TF2/trained_models/scatter_'+THIS_CASE+'.png')
    plt.show(block=False)


    R2_score = r2_score(omega_DNS_predict_real,test_df_real['omega_DNS_filtered'])
    print('##############################')
    print('R2 score on test set: ',R2_score)
    print('##############################')

    plt.figure()
    plt.scatter(test_df_real['omega_DNS_filtered'],omega_DNS_predict_real,s=0.3)
    plt.plot([0,1],[0,1],'k')
    plt.xlabel('true omega_DNS_filtered')
    plt.ylabel('predicted omega_DNS_filtered')
    plt.text(0.05, 0.8, 'R2_score: '+str(round(R2_score,5)))
    plt.savefig('/home/max/Python/Data_driven_models/TF2/trained_models/CORR_' + THIS_CASE + '.png')
    plt.show(block=False)



##################################
#Build and compile the ResNet
##################################

# estimate the length of the data set
number_datapoints = tf.data.experimental.cardinality(validation_dataset).numpy() * len(training_files)

#TODO: test with this optimizer
# custom optimizer with learning rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=lr,
    decay_steps=1000,
    decay_rate=0.9)
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# function to compile a model
def compiled_model(dim_input=len(FEATURES),dim_output=len(TARGET),neurons=NEURONS,blocks=RES_BLOCKS,loss=LOSS):

    # DISTRIBUTED
    #with strategy.scope():
    inputs = tf.keras.layers.Input(shape=(dim_input,),name='Input')
    x = tf.keras.layers.Dense(neurons, activation='relu')(inputs)
    for b in range(1,blocks+1):
        x = res_block_org(x,neurons,block=str(b))
        #x = tf.keras.layers.Dropout(rate=0.1)(x)

    # add a droput layer
    # x = tf.keras.layers.Dropout(rate=0.1)(x)
    # add another bypass layer
    x = tf.keras.layers.Dense(dim_input,activation='relu')(x)
    x = tf.keras.layers.add([x, inputs],name='add_layers')
    # x = tf.keras.Activation('relu')(x)
    output = tf.keras.layers.Dense(dim_output, activation='linear', name='prediction_layer', kernel_regularizer=l1_l2(0.01,0.01))(x)
    model = tf.keras.Model(inputs=inputs,outputs=output)

    #compile model
    model.compile(loss=loss, optimizer=adam_optimizer, metrics=[loss])

    return model

# call backs list for early stopping
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_loss')<1e-7):
            print("\nReached loss < 1e-7 so cancelling training!")
            self.model.stop_training = True

loss_callback = myCallback()

# This callback will stop the training when there is no improvement for mind_delta in
# the validation loss for 5 consecutive epochs.
earlyStop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

# checkpoint the model (save it)
checkpoint = ModelCheckpoint(DNN_model_path,
                             monitor='val_loss',        #TODO: 'val_loss should somehow also work??
                             verbose=1,
                             save_best_only=True,
                             mode='min',
                             save_freq='epoch')


#TODO: check this again
# create the model or read it from file
if os.path.isfile(DNN_model_path):
    print('\nTrained model is already available, reading it from disk:')
    print(DNN_model_path)
    DNN = tf.keras.models.load_model(DNN_model_path)
    DNN.compile(loss=LOSS, optimizer=adam_optimizer, metrics=[LOSS])
else:
    print('\nNo model available. Compiling new one')
    DNN = compiled_model(dim_input=len(FEATURES),
                         dim_output=len(TARGET),
                         neurons=NEURONS,
                         blocks=RES_BLOCKS,
                         loss=LOSS)


##################################
# LOOP OVER THE TRAININGS FILES
##################################
history_val_loss =[]
history_train_loss =[]

if args.MODE == 'TRAIN':
    print('\n*******************\nStarting Training Loop\n*******************\n')
    for file_name in training_files:
        print('\nTraining on file: %s\n' % file_name)

        # read in from file
        this_train_df = pd.read_parquet(join(path_to_data,'TRAIN',file_name)).sample(frac=0.75)

        # drop rows where NaN
        this_train_df = this_train_df.dropna(axis=0)

        # normalize the data set
        normalized_train_df = normalizeStandard(this_train_df,moments)

        # compose TF dataset
        training_dataset = (
            tf.data.Dataset.from_tensor_slices(
                (
                    tf.cast(normalized_train_df[FEATURES].values, tf.float32),
                    tf.cast(normalized_train_df[TARGET].values, tf.float32)
                )
            )
        )

        # Cache and prefetch
        #training_dataset = training_dataset.cache()
        training_dataset = training_dataset.batch(BATCH_SIZE )
        #training_dataset = training_dataset.prefetch(tf.data.experimental.AUTOTUNE) # AUTOTUNE?

        # update the call backs list
        callbacks = [loss_callback,checkpoint,earlyStop_callback] #[schedule,loss_callback,earlyStop_callback,checkpoint]

        #print(DNN.summary())

        print('Batch size: ', BATCH_SIZE )

        print('Learning rate: ',lr)

        # custom optimizer with learning rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=lr,
                decay_steps=1000,
                decay_rate=0.9)

        adam_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        DNN.compile(loss=LOSS, optimizer=adam_optimizer, metrics=[LOSS])

        # fit the model
        history = DNN.fit(
            training_dataset,
            #normalized_train_df[FEATURES].to_numpy(),normalized_train_df[TARGET].to_numpy(),       #TODO: What if I use X_train, y_train (np.array)?
            epochs=EPOCHS,  #epochs,
            #validation_split=0.1,
            validation_data=validation_dataset,
            validation_steps=validation_steps,
            #verbose=1,
            callbacks=callbacks,
        )

        # append loss (train/val) per epoch to list for evaluation
        history_train_loss.append(history.history['loss'])
        history_train_loss.append(history.history['val_loss'])

        # decrease learning rate
        lr = lr*0.9


elif args.MODE=='TEST':
    print('\n*******************\nTEST MODE\n*******************\n')
    #TEST_CASE=input('Test on case: ')
    TEST_CASES = ['UPRIME5','UPRIME75','UPRIME90','UPRIME15']
    for case in TEST_CASES:
        predict_test(keep_out=False,THIS_CASE=case)

else:
    print('Either TRAIN or TEST mode are allowed')



# # SHAP model explanation
# # use Kernel SHAP to explain test set predictions
# explainer = shap.DeepExplainer(DNN, normalized_train_df[FEATURES].sample(n=1000, random_state=0).to_numpy())
# shap_values = explainer.shap_values(X_test, nsamples=100)