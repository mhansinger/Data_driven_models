#%%
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

print('TF version: ',tf.__version__)


#%%
#########################
# BATCH SIZE
BATCH_SIZE=100
########################

# create training data
X_train_set = np.random.random(size=(10000,10))
y_train_set = np.random.random(size=(10000))

# create validation data
X_val_set = np.random.random(size=(100,10))
y_val_set = np.random.random(size=(100))

# convert np.array to dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_set, y_train_set))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val_set, y_val_set))

# batching
train_dataset=train_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)

# set up the model
my_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])

#%%
# custom optimizer with learning rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


# compile the model
my_model.compile(optimizer=optimizer,loss='mse')

# define a checkpoint
checkpoint = ModelCheckpoint('./tf.keras_test',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',
                             save_freq='epoch')

callbacks = [checkpoint]

#%%
# train with datasets
history= my_model.fit(train_dataset,
             validation_data=val_dataset,
             #validation_steps=100,
             #callbacks=callbacks,
             epochs=2)

# save as .h5
my_model.save("./my_model",save_format='h5')

#%%
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show(blocking=False)