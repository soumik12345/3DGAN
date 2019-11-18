from .block import *
from config import *
from tensorflow.keras.layer import Input
from tensorflow.keras.models import Model


def Generator():
	'''Generator Model'''
	input_layer = Input(shape=(1, 1, 1, LATENT_DIMENSION))
	x = generator_block(input_layer, 512, (1, 1, 1), 'valid')
	x = generator_block(x, 256)
	x = generator_block(x, 128)
	x = generator_block(x, 64)
	output_layer = generator_block(x, 1, 'sigmoid')
	model = Model(input_layer, output_layer, name='Generator')
	return model