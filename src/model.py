from torch.nn import Module, Sequential, ConvTranspose3d, ReLU, BatchNorm3d, Sigmoid


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