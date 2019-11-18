from torch.nn import (
    ReLU, BatchNorm3d, Sigmoid,
    Conv3d, BatchNorm3d, LeakyReLU,
    Module, Sequential, ConvTranspose3d
)


class Generator(Module):
	
	def __init__(self):
		super(Generator, self).__init__()
		self.block_1 = Sequential(
			ConvTranspose3d(200, 512, 4, 2, 0 ),
			BatchNorm3d(512),
			ReLU()
		)
		self.block_2 = Sequential(
			ConvTranspose3d(512, 256, 4, 2, 1 ),
			BatchNorm3d(256),
			ReLU()
		)
		self.block_3 = Sequential(
			ConvTranspose3d(256, 128, 4, 2, 1 ),
			BatchNorm3d(128),
			ReLU()
		)
		self.block_4 = Sequential(
			ConvTranspose3d(128, 1, 4, 2, 1 ),
			Sigmoid()
		)
		
	def forward(self, x):
		x = x.view(-1, 200, 1, 1, 1)
		x = self.block_1(x)
		x = self.block_2(x)
		x = self.block_3(x)
		x = self.block_4(x)



class Discrimintor(Module):

	def __init__(self):
		super(Discrimintor, self).__init__()
		self.block_1 = Sequential(
			Conv3d(1, 64, 4, 2, 1),
			BatchNorm3d(64),
			LeakyReLU()
		)
		self.block_2 = Sequential(
			Conv3d(64, 128, 4, 2, 1),
			BatchNorm3d(128),
			LeakyReLU()
		)
		self.block_3 = Sequential(
			Conv3d(128, 256, 4, 2, 1),
			BatchNorm3d(256),
			LeakyReLU()
		)
		self.block_4 = Sequential(
			Conv3d(256, 512, 4, 2, 1),
			BatchNorm3d(512),
			LeakyReLU()
		)
		self.block_5 = Sequential(
			Conv3d(512, 1, 2, 2, 0),
			Sigmoid()
		)
	
	def forward(self, x):
		x = x.view(-1, 1, 32, 32, 32)
		x = self.block_1(x)
		x = self.block_2(x)
		x = self.block_3(x)
		x = self.block_4(x)
		x = self.block_5(x)
		return x