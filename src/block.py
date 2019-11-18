from config import *
from tensorflow.keras.layers import Conv3DTranspose, BatchNormalization, Activation


def generator_block(input_tensor, filters, strides=2, padding='same', activation='relu'):
    '''Generator Block
    Params:
        input_tensor -> Input tensor
        filters      -> Number of filters in the Conv3DTranspose layer
        strides      -> Strides in the Conv3DTranspose layer
        padding      -> Padding in the Conv3DTranspose layer (same/valid)
        activation   -> Activation Function
    '''
    x = Conv3DTranspose(
        filters=filters, kernel_size=4,
        strides=strides, padding=padding,
        kernel_initializer='glorot_normal',
        bias_initializer='zeros'
    )(input_tensor)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x