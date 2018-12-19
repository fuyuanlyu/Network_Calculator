##############################################################################################
#	Created by Fuyuan Lyu Tommy at 2018-12/18
#	MIT License
#   Contain definition of different layer
##############################################################################################


import math
import numpy as np
import util.util as utils

# Basic Layer Class
# All Layer inherit from this
class Layer:
	def __init__(self, name='layer', input_k=0, input_channel=0):
		self.input_k = utils.make_double(input_k)
		self.input_channel = input_channel
		self.name = name

	# Pass the output structure of former layer to the latter one as input structure
	def inherit(self, prev):
		self.input_k = prev.output_k
		self.input_channel = prev.output_channel

	# Re-initialize input parameters
	def reinit(self, input_k, input_channel):
		self.input_k = utils.make_double(input_k)
		self.input_channel = input_channel

# Pooling layer
class PoolLayer(Layer):
	def __init__(self, name='pooling', input_k=0, input_channel=0, \
					kernel=1, stride=1, padding=0, dilation=1):
		super(PoolLayer, self).__init__(name, input_k, input_channel)
		self.output_channel = self.input_channel
		self.kernel = utils.make_double(kernel)
		self.stride = utils.make_double(stride)
		self.padding = utils.make_double(padding)
		self.dilation = utils.make_double(dilation)
		self.params = 0

	# Calculation Formula comes from PyTorch documentation
	# https://pytorch.org/docs/stable/nn.html?highlight=pool#torch.nn.MaxPool2d
	def calculate_output(self):
		self.output_k = [1,1]
		self.output_k[0] = int(math.floor((self.input_k[0]-self.dilation[0]*(self.kernel[0]-1)\
						+2*self.padding[0]-1)/float(self.stride[0]) + 1.0))
		self.output_k[1] = int(math.floor((self.input_k[1]-self.dilation[1]*(self.kernel[1]-1)\
						+2*self.padding[1]-1)/float(self.stride[1]) + 1.0))
		self.output_channel = self.input_channel

	# Compare to convolution and fully-connected layers, pooling layer can be ignored
	def calculate_FLOPs(self):
		self.add_ops = 0
		self.times_ops = 0
		self.FLOPs = 0

	def calculate_all(self):
		self.calculate_output()
		self.calculate_FLOPs()




# Convolutional Layer, current version only contain 2d
# 1d and 3d will be added later
class ConvLayer(Layer):
	def __init__(self, name='conv', input_k=0, input_channel=0, kernel=1,\
	 				output_channel=0, stride=1, padding=0, dilation=1, bias=True):
		super(ConvLayer, self).__init__(name, input_k, input_channel)
		self.kernel = utils.make_double(kernel)
		self.output_channel = output_channel
		self.stride = utils.make_double(stride)
		self.padding = utils.make_double(padding)
		self.dilation = utils.make_double(dilation)
		self.bias = bias

	# Calculation Formula comes from PyTorch documentation
	# https://pytorch.org/docs/stable/nn.html?highlight=conv#torch.nn.Conv2d
	def calculate_output(self):
		self.output_k = [1,1]
		self.output_k[0] = int(math.floor((self.input_k[0]-self.dilation[0]*(self.kernel[0]-1)\
						+2*self.padding[0]-1)/float(self.stride[0]) + 1.0))
		self.output_k[1] = int(math.floor((self.input_k[1]-self.dilation[1]*(self.kernel[1]-1)\
						+2*self.padding[1]-1)/float(self.stride[1]) + 1.0))

	def calculate_parameters(self):
		self.params = float(np.multiply.reduce(self.kernel)) * self.input_channel * self.output_channel
		if self.bias == True:
			self.params += self.output_channel

	def calculate_FLOPs(self):
		self.times_ops = float(np.multiply.reduce(self.kernel)) * self.input_channel *\
						np.multiply.reduce(self.output_k) * self.output_channel
		if self.bias == True:
			self.add_ops = self.times_ops
		else:
			self.add_ops = (float(np.multiply.reduce(self.kernel)) * self.input_channel - 1) *\
							np.multiply.reduce(self.output_k) * self.output_channel
		self.FLOPs = self.times_ops + self.add_ops


	def calculate_all(self):
		self.calculate_output()
		self.calculate_parameters()
		self.calculate_FLOPs()


# Fully-Connected Layer
class FCLayer(Layer):
	def __init__(self, name='FC', input_k=0, input_channel=0, output_k = 1):
		super(FCLayer, self).__init__(name, input_k, input_channel)
		self.output_k = output_k

	def calculate_parameters(self):
		self.params = 2 * np.multiply.reduce(self.input_k) * self.input_channel * self.output_k

	def calculate_FLOPs(self):
		self.times_ops = self.params
		self.add_ops = self.params
		self.FLOPs = self.times_ops + self.add_ops

	def calculate_all(self):
		self.calculate_parameters()
		self.calculate_FLOPs()


# Concatenation, defined as layer for utility
class ConcateLayer(Layer):
	def __init__(self, name='ConcateLayer', layer_list=[]):
		self.output_k = layer_list[0].output_k
		self.output_channel = 0
		for i,l in enumerate(layer_list):
			self.output_channel += l.output_channel			
		#Qprint(self.output_channel)