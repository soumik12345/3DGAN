from .dataset import *
from config import *
import numpy as np


def train(generator, discriminator, gan, x_train):
    '''Train GAN
    Params:
        generator       -> Generator Model
        discriminator   -> Discriminator Model
        gan             -> Combined GAN Model
        x_train         -> Training Data
    '''
    generator_loss_history, discriminator_loss_history = [], []
    for epoch in tqdm(range(1, EPOCHS + 1)):
        index = np.random.randint(len(x_train), size=BATCH_SIZE)
        real = x_train[index]
        z = np.random.normal(
            0, 0.33,
            size=[BATCH_SIZE, 1, 1, 1, LATENT_DIMENSION]
        ).astype(np.float32)
        fake = generator.predict(z)
        real = np.expand_dims(real, axis=4)
        ground_real = np.reshape([1] * BATCH_SIZE, (-1, 1, 1, 1, 1))
        ground_fake = np.reshape([0] * BATCH_SIZE, (-1, 1, 1, 1, 1))
        discriminator_loss_real = discriminator.train_on_batch(real, ground_real)
        discriminator_loss_fake = discriminator.train_on_batch(fake, ground_fake)
        discriminator_loss_history.append(
            0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)
        )
        z = np.random.normal(0, 0.33, size=[BATCH_SIZE, 1, 1, 1, LATENT_DIMENSION]).astype(np.float32)
        generator_loss = gan.train_on_batch(z, np.reshape([1] * BATCH_SIZE, (-1, 1, 1, 1, 1))).astype(np.float64)
        generator_loss_history.append(generator_loss)
    return generator_loss_history, discriminator_loss_history