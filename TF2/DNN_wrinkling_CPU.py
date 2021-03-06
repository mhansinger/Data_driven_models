
from typing import List

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn as sk
import os
from os.path import join
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from utils.customObjects import coeff_r2, SGDRScheduler
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.regularizers import l1, l2, l1_l2

from utils.normalize_data import normalizeStandard, reTransformStandard, reTransformTarget
from utils.resBlock import res_block_org


'''
This is to train the Network for various Delta_LES
'''


# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#   except RuntimeError as e:
#     print(e)

#%%

##################################
#CASE and parameters
##################################

# switch off GPU
os.environ['CUDA_VISIBLE_DEVICES']="-1"

# DISTRIBUTED
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


CASE = 'UPRIME5'

BATCH_SIZE = 64#64#128
NEURONS = 200
RES_BLOCKS =10
EPOCHS=10
LOSS='mse' #'mse'

##################################
# PATHS
##################################

# path to the UPRIMEXY data set
path_to_data = '/home/hansinger/hansinger_share/'+CASE

# read in the moments (mean and std of the data set
moments = pd.read_csv(join(path_to_data,'moments_'+CASE+'.csv'),index_col=0)

DNN_model_path = join('/home/hansinger/hansinger_share/ANNs/DNN_'+CASE+'_nrns_'+str(NEURONS)+'_blks_'+str(RES_BLOCKS)+'.h5')
#join(path_to_data,'DNN_'+CASE+'_nrns_'+str(NEURONS)+'_blks_'+str(RES_BLOCKS)+'.h5')

###################################
# which are the features in the data to train for
###################################
FEATURES: List[str] = ['c_bar',  'omega_model_planar', 'UP_delta',
                        'SGS_flux', 'Delta_LES', 'mag_grad_c', 'mag_U', 'sum_c', 'sum_U',
                        'sum_grad_U', 'mag_grad_U', 'lambda_1',  'lambda_3',]

TARGET: List[str] = ['omega_DNS_filtered']

training_files = os.listdir(join(path_to_data,'TRAIN'))
training_files = [f for f in training_files if (f.startswith('train') and f.endswith('parquet'))]
random.shuffle(training_files)

print('Training files: ',training_files)

#retain one set as validation set:
validation_file = training_files.pop(0)

##################################
#validation data set
##################################
validation_df = pd.read_parquet(join(path_to_data,'TRAIN',validation_file))

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
test_files = [f for f in test_files if (f.startswith('test') and f.endswith('parquet'))]
# shuffle the test_files list
random.shuffle(test_files)

test_df = pd.read_parquet(join(path_to_data, 'TEST', test_files[0]))

test_df_norm = normalizeStandard(test_df, moments)


def predict_test():
    print('\n##############################')
    print('Predict on test set')
    print('##############################\n')

    # predict y_hat
    y_hat = DNN.predict(test_df_norm[FEATURES])

    omega_DNS_predict = reTransformTarget(y_hat, moments)

    # plot the results
    plt.figure()
    plt.scatter(test_df['c_bar'],test_df['omega_DNS_filtered'],s=0.3,c='b')
    plt.scatter(test_df['c_bar'],omega_DNS_predict,s=0.3,c='r')
    plt.scatter(test_df['c_bar'],test_df['omega_model_planar'],s=0.3,c='k')
    plt.xlabel('c_bar')
    plt.ylabel('omega')
    plt.show(block=False)

    plt.figure()
    plt.scatter(test_df['omega_DNS_filtered'],omega_DNS_predict,s=0.3)
    plt.plot([0,1],[0,1],'k')
    plt.xlabel('true omega_DNS_filtered')
    plt.ylabel('predicted omega_DNS_filtered')
    plt.show(block=False)

##################################
#Build and compile the ResNet
##################################

# estimate the length of the data set
number_datapoints = tf.data.experimental.cardinality(validation_dataset).numpy() * len(training_files)

#TODO: test with this optimizer
# custom optimizer with learning rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=1000,
    decay_rate=0.8)
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# function to compile a model
def compiled_model(dim_input=len(FEATURES),dim_output=len(TARGET),neurons=NEURONS,blocks=RES_BLOCKS,loss=LOSS):

    # DISTRIBUTED
    with strategy.scope():
        inputs = tf.keras.layers.Input(shape=(dim_input,),name='Input')
        x = tf.keras.layers.Dense(neurons, activation='relu')(inputs)
        for b in range(1,blocks+1):
            x = res_block_org(x,neurons,block=str(b))
           # x = tf.keras.layers.Dropout(rate=0.1)(x)

        # add a droput layer
        x = tf.keras.layers.Dropout(rate=0.1)(x)
        # add another bypass layer
        x = tf.keras.layers.Dense(dim_input,activation='relu',kernel_regularizer=l2(0.01))(x)
        x = tf.keras.layers.add([x, inputs],name='add_layers')
        # x = tf.keras.Activation('relu')(x)
        output = tf.keras.layers.Dense(dim_output,activation='linear', name='prediction_layer',kernel_regularizer=l2(0.01))(x)
        model = tf.keras.Model(inputs=inputs,outputs=output)

        #compile model
        model.compile(loss=loss, optimizer=adam_optimizer, metrics=[loss])

    return model

# call backs list for early stopping
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_loss')<1e-5):
            print("\nReached loss < 1e-7 so cancelling training!")
            self.model.stop_training = True

loss_callback = myCallback()
earlyStop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, min_delta=1e-6)

# checkpoint the model (save it)
checkpoint = ModelCheckpoint(DNN_model_path,
                             monitor='val_loss',        #TODO: 'val_loss should somehow also work??
                             verbose=1,
                             save_best_only=True,
                             mode='max',
                             save_freq='epoch')


#TODO: check this ain
# create the model or read it from file
if os.path.isfile(DNN_model_path):
    print('\nTrained model is already available, reading it from disk')
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

# counter training_files
no_files = len(training_files)+1

print('\n*******************\nStarting Training Loop\n*******************\n')
for file_name in training_files:
    print('Training on file: %s\n' % file_name)

    # read in from file
    this_train_df = pd.read_parquet(join(path_to_data,'TRAIN',file_name))

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

    # Scheduler
    #TODO: seems to work
    a = 0
    base = 2
    clc = 2
    for i in range(6):
        a += base * clc ** (i)
    print(a)
    epochs, c_len = a, base
    schedule = SGDRScheduler(min_lr=1e-6, max_lr=1e-4,
                             steps_per_epoch=np.ceil(epochs / (BATCH_SIZE )),
                             cycle_length=c_len, lr_decay=0.6, mult_factor=clc)

    # update the call backs list
    callbacks = [loss_callback,checkpoint, earlyStop_callback] #[schedule,loss_callback,earlyStop_callback,checkpoint]

    #print(DNN.summary())

    print('Batch size: ',BATCH_SIZE )

    # fit the model
    history = DNN.fit(
        training_dataset,
        #normalized_train_df[FEATURES].to_numpy(),normalized_train_df[TARGET].to_numpy(),                       #TODO: What if I use X_train, y_train (np.array)?
        epochs=EPOCHS,#epochs,
        #validation_split=0.1,
        validation_data=validation_dataset,     #TODO: use crossvalidation??
        validation_steps=validation_steps,
        verbose=1,
        callbacks=callbacks,
        )

    no_files = no_files -1

    # append loss (train/val) per epoch to list for evaluation
    history_train_loss.append(history.history['loss'])
    history_train_loss.append(history.history['val_loss'])

    # SAVE model
    DNN.save(DNN_model_path,save_format='h5')



    # plt.figure()
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.show(block=False)
