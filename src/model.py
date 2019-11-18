from .block import *
from config import *
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam


def Generator():
	'''Generator Model'''
	input_layer = Input(shape=(1, 1, 1, LATENT_DIMENSION))
	x = generator_block(input_layer, 512, (1, 1, 1), 'valid')
	x = generator_block(x, 256)
	x = generator_block(x, 128)
	x = generator_block(x, 64)
	output_layer = generator_block(x, 1, activation='sigmoid')
	model = Model(input_layer, output_layer, name='Generator')
	return model


def Discriminator():
	'''Discriminator Model'''
	image_input = Input(shape=(IMAGE_DIMENSION, IMAGE_DIMENSION, IMAGE_DIMENSION, 1))
	blocks = []
	blocks += discriminator_block(64)
	blocks += discriminator_block(128)
	blocks += discriminator_block(256)
	blocks += discriminator_block(512)
	blocks += discriminator_block(1, (1, 1, 1), 'valid', 'sigmoid')
	model = Sequential(blocks)
	validity_output = model(image_input)
	discriminator_model = Model(image_input, validity_output, name='Discriminator')
	return discriminator_model



def GAN(generator, discriminator):
	'''Combined GAN model'''
	discrimintor_optimizer = Adam(lr=GENERATOR_LEARNING_RATE, beta_1=0.5)
	generator_optimizer = Adam(lr=DISCRIMINATOR_LEARNING_RATE, beta_1=0.5)
	
	discriminator = Discriminator()
	discriminator.compile(
		loss='binary_crossentropy',
		optimizer=discrimintor_optimizer
	)

	generator = Generator()
	adversarial_noise = Input(shape=(1, 1, 1, LATENT_DIMENSION))
	image = generator(adversarial_noise)

	discriminator.trainable = False
	validity_output = discriminator(image)

	gan = Model(adversarial_noise, validity_output, name='3DGAN')
	gan.compile(
		loss='binary_crossentropy',
		optimizer=generator_optimizer
	)

	return generator, discriminator, gan