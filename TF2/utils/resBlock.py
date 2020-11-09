from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Input, BatchNormalization, Dropout, concatenate
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint


def res_block_org(input_tensor, n_neuron,  block):
    name_base = 'res' + block + '_branch'
    bn_name_base = 'bn' + block + '_branch'

    x = Dense(n_neuron, name=name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=-1, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Dense(n_neuron, name=name_base + '2b')(x)
    x = BatchNormalization(axis=-1, name=bn_name_base + '2b')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)

    return x


# Deactivate the skip connection
# 1 Block is two dense layers
def NoRes_block(input_tensor, n_neuron, stage, block, bn=False):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Dense(n_neuron, name=conv_name_base + '2a')(input_tensor)
    x = Activation('relu')(x)
    x = Dropout(0.)(x)

    x = Dense(n_neuron, name=conv_name_base + '2b')(x)
    # x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    x = Dropout(0.)(x)

    return x



def res_branch(bi, conv_name_base, bn_name_base, scale, input_tensor, n_neuron, stage, block, dp1, bn=False):
    x_1 = Dense(scale * n_neuron, name=conv_name_base + '2a_' + str(bi))(input_tensor)
    if bn:
        x_1 = BatchNormalization(axis=-1, name=bn_name_base + '2a_' + str(bi))(x_1)
    x_1 = Activation('relu')(x_1)
    if dp1 > 0:
        x_1 = Dropout(dp1)(x_1)
    return x_1

# new resnet block implementation with bottle neck
def res_block(input_tensor, scale, n_neuron, stage, block, bn=False, branches=0):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # scale = 2
    x = Dense(scale * n_neuron, name=conv_name_base + '2a')(input_tensor)
    if bn:
        x = BatchNormalization(axis=-1, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)


    dp1 = 0


    if dp1 > 0:
        x = Dropout(dp1)(x)

    branch_list = [x]
    for i in range(branches - 1):
        branch_list.append(
            res_branch(i, conv_name_base, bn_name_base, scale, input_tensor, n_neuron, stage, block, dp1, bn))
    if branches - 1 > 0:
        x = Dense(n_neuron, name=conv_name_base + '2b')(concatenate(branch_list, axis=-1))
    #         x = Dense(n_neuron, name=conv_name_base + '2b')(layers.add(branch_list))
    else:
        x = Dense(n_neuron, name=conv_name_base + '2b')(x)

    if bn:
        x = BatchNormalization(axis=-1, name=bn_name_base + '2b')(x)
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    if dp1 > 0:
        x = Dropout(dp1)(x)

    return x


# from Junsu
def identity_block(input_tensor, n_neuron, stage, step):

    res_name_base = 'res' + str(stage) + str(step)
    bn_name_base = 'bn' + str(stage) + str(step)

    x = Dense(n_neuron, name=res_name_base+'1')(input_tensor)
    x = BatchNormalization(axis=-1, name=bn_name_base+'1')(x)
    x = Activation('relu')(x)

    x = Dense(n_neuron, name=res_name_base+'2')(x)
    x = BatchNormalization(axis=-1, name=bn_name_base+'2')(x)
    x = Activation('relu')(x)

    x = Dense(n_neuron, name=res_name_base+'3')(x)
    x = BatchNormalization(axis=-1, name=bn_name_base+'3')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)

    return x

def dens_block(input_tensor, n_neuron, stage):

    res_name_base = 'res' + str(stage)
    bn_name_base = 'bn' + str(stage)

    x = Dense(n_neuron, name=res_name_base+'1')(input_tensor)
    x = BatchNormalization(axis=-1, name=bn_name_base+'1')(x)
    x = Activation('relu')(x)

    x = Dense(n_neuron, name=res_name_base+'2')(x)
    x = BatchNormalization(axis=-1, name=bn_name_base+'2')(x)
    x = Activation('relu')(x)

    x = Dense(n_neuron, name=res_name_base+'3')(x)
    x = BatchNormalization(axis=-1, name=bn_name_base+'3')(x)

    shortcut = Dense(n_neuron, name='res_shortcut'+str(stage))(input_tensor)
    shortcut = BatchNormalization(axis=-1, name='bn_shortcut'+str(stage))(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)

    return x

