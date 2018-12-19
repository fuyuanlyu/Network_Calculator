##############################################################################################
#	Created by Fuyuan Lyu Tommy at 2018-12/18
#	MIT License
#   Contain GoogLeNet Structure
##############################################################################################

import util.layer as layer
import util.util as utils
import csv



# GoogLeNet Block
class InceptionLayer(layer.Layer):
	def __init__(self, name, input_k=0, input_channel=1, n1x1=1, n3x3red=1, n3x3=1, \
					n5x5red=1, n5x5=1, pool=1):
		self.input_k = utils.make_double(input_k)
		self.input_channel = input_channel
		self.name = name

		# 1x1 channel
		self.layer_1x1 = layer.ConvLayer(name='1x1',input_k=self.input_k,\
					input_channel=self.input_channel,kernel=1,output_channel=n1x1)
		
		# 3x3 channel
		self.layer_3x3red = layer.ConvLayer(name='3x3red',input_k=self.input_k,\
					input_channel=self.input_channel,kernel=1,output_channel=n3x3red)
		self.layer_3x3 = layer.ConvLayer(name='3x3',kernel=3,output_channel=n3x3,padding=1)

		# 5x5 channel
		self.layer_5x5red = layer.ConvLayer(name='5x5red',input_k=self.input_k,\
					input_channel=self.input_channel,kernel=1,output_channel=n5x5red)
		self.layer_5x5 = layer.ConvLayer(name='5x5',kernel=3,output_channel=n5x5,padding=1)
		self.layer_5x5_2 = layer.ConvLayer(name='5x5_2',kernel=3,output_channel=n5x5,padding=1)

		# Pooling channel
		self.layer_pool = layer.PoolLayer(name='pool',input_k=self.input_k,\
					input_channel=self.input_channel,kernel=3,padding=1)
		self.layer_pool_1x1 = layer.ConvLayer(name='pool_1x1',kernel=1,output_channel=pool)


	def calculate_FLOPs_and_params(self):
		for i,l in enumerate(self.all_layer):
			l.input_k = self.input_k
			l.input_channel = self.input_channel
			l.calculate_all()
			self.params += float(l.params)
			self.times_ops += float(l.times_ops)
			self.add_ops += float(l.add_ops)
			self.FLOPs += float(l.FLOPs)


	def calculate_output(self):
		# 1x1 channel
		self.layer_1x1.reinit(self.input_k, self.input_channel)
		self.layer_1x1.calculate_all()

		# 3x3 channel
		self.layer_3x3red.reinit(self.input_k, self.input_channel)
		self.layer_3x3red.calculate_all()
		self.layer_3x3.inherit(self.layer_3x3red)
		self.layer_3x3.calculate_all()

		# 5x5 channel
		self.layer_5x5red.reinit(self.input_k, self.input_channel)
		self.layer_5x5red.calculate_all()
		self.layer_5x5.inherit(self.layer_5x5red)
		self.layer_5x5.calculate_all()
		self.layer_5x5_2.inherit(self.layer_5x5)
		self.layer_5x5_2.calculate_all()

		# Pooling channel
		self.layer_pool.reinit(self.input_k, self.input_channel)
		self.layer_pool.calculate_all()
		self.layer_pool_1x1.inherit(self.layer_pool)
		self.layer_pool_1x1.calculate_all()

		# Create layer lists
		self.second_layer = []
		self.second_layer.append(self.layer_1x1)
		self.second_layer.append(self.layer_3x3)
		self.second_layer.append(self.layer_5x5_2)
		self.second_layer.append(self.layer_pool_1x1)
		self.all_layer = self.second_layer.copy()
		self.all_layer.append(self.layer_3x3red)
		self.all_layer.append(self.layer_5x5red)
		self.all_layer.append(self.layer_5x5)

		# Concatenation Layer
		self.concate_layer = layer.ConcateLayer(name='concate',layer_list=self.second_layer)

		# For latter usage
		self.output_channel = self.concate_layer.output_channel
		self.output_k = utils.make_double(self.concate_layer.output_k)


	def calculate_all(self):
		self.calculate_output()
		self.calculate_FLOPs_and_params()



class GoogLeNet(object):
	def __init__(self,input_k,input_channel):

		input_k = utils.make_double(input_k)
		self.conv1 = layer.ConvLayer(name='conv1', input_k=input_k, input_channel=input_channel, kernel=3,\
						output_channel=192, padding=1)
		self.a3 = InceptionLayer(name='a3', n1x1=64, n3x3red=96, \
						n3x3=128, n5x5red=16, n5x5=32, pool=32)
		self.b3 = InceptionLayer(name='b3', n1x1=128, n3x3red=128, \
						n3x3=192, n5x5red=32, n5x5=96, pool=64)

		self.maxpool = layer.PoolLayer(name='maxpool', kernel=3, stride=2, padding=1)

		self.a4 = InceptionLayer(name='a4', n1x1=192, n3x3red=96, \
						n3x3=208, n5x5red=16, n5x5=48, pool=64)
		self.b4 = InceptionLayer(name='b4', n1x1=160, n3x3red=112, \
						n3x3=224, n5x5red=24, n5x5=64, pool=64)
		self.c4 = InceptionLayer(name='c4', n1x1=128, n3x3red=128, \
						n3x3=256, n5x5red=24, n5x5=64, pool=64)
		self.d4 = InceptionLayer(name='d4', n1x1=112, n3x3red=144, \
						n3x3=288, n5x5red=32, n5x5=64, pool=64)
		self.e4 = InceptionLayer(name='e4', n1x1=256, n3x3red=160, \
						n3x3=320, n5x5red=32, n5x5=128, pool=128)

		self.a5 = InceptionLayer(name='a3', n1x1=256, n3x3red=160, \
						n3x3=320, n5x5red=32, n5x5=128, pool=128)
		self.b5 = InceptionLayer(name='b3', n1x1=384, n3x3red=192, \
						n3x3=384, n5x5red=48, n5x5=128, pool=128)
		

		self.avgpool = layer.PoolLayer(name='avgpool', kernel=8, stride=1)
		self.FC = layer.FCLayer(name='FC', output_k=10)

		
	def calculate_output(self):
		self.conv1.calculate_all()
		self.a3.inherit(self.conv1)
		self.a3.calculate_all()
		self.b3.inherit(self.a3)
		self.b3.calculate_all()

		self.maxpool.inherit(self.b3)
		self.maxpool.calculate_all()

		self.a4.inherit(self.maxpool)
		self.a4.calculate_all()
		self.b4.inherit(self.a4)
		self.b4.calculate_all()
		self.c4.inherit(self.b4)
		self.c4.calculate_all()
		self.d4.inherit(self.c4)
		self.d4.calculate_all()
		self.e4.inherit(self.d4)
		self.e4.calculate_all()

		self.a5.inherit(self.e4)
		self.a5.calculate_all()
		self.b5.inherit(self.a5)
		self.b5.calculate_all()

		self.avgpool.inherit(self.b5)
		self.avgpool.calculate_all()
		self.FC.inherit(self.avgpool)
		self.FC.calculate_all()


		self.all_layer = [self.conv1, self.a3, self.b3, self.maxpool, self.a4, self.b4, self.c4,\
						self.d4, self.e4, self.a5, self.b5, self.avgpool, self.FC]

	def calculate_FLOPs_and_params(self):
		self.times_ops = 0.0
		self.add_ops = 0.0
		self.FLOPs = 0.0
		self.params = 0.0
		for i,l in enumerate(self.all_layer):
			self.params += float(l.params)
			self.times_ops += float(l.times_ops)
			self.add_ops += float(l.add_ops)
			self.FLOPs += float(l.FLOPs)


	# Write to CSV File
	def write_csv(self,filename="GoogLeNet.csv"):
		with open(filename,'w') as csvFile:
			writer = csv.writer(csvFile,lineterminator='\n')
			writer.writerow(["name","input","output","filter","1x1","3x3red","3x3","5x5red","5x5","5x5_2",\
							"pool","pool_1x1","params","add_ops","times_ops","FLOPs"])
			for i, l in enumerate(self.all_layer):
				temp = []
				temp.append(l.name)
				temp.append("{}x{}x{}".format(l.input_k[0], l.input_k[1], l.input_channel))

				if isinstance(l, layer.FCLayer):
					temp.append("{}".format(l.output_k))
				else:
					temp.append("{}x{}x{}".format(l.output_k[0], l.output_k[1], l.output_channel))

				if isinstance(l, InceptionLayer):
					temp.append(" ")
					temp.append("{}".format(l.layer_1x1.output_channel))
					temp.append("{}".format(l.layer_3x3red.output_channel))
					temp.append("{}".format(l.layer_3x3.output_channel))
					temp.append("{}".format(l.layer_5x5red.output_channel))
					temp.append("{}".format(l.layer_5x5.output_channel))
					temp.append("{}".format(l.layer_5x5_2.output_channel))
					temp.append("{}".format(l.layer_pool.output_channel))
					temp.append("{}".format(l.layer_pool_1x1.output_channel))
				elif isinstance(l, layer.FCLayer):
					for j in range(9):
						temp.append(" ")
				else:
					temp.append("{}x{}/{}".format(l.kernel[0], l.kernel[1], l.stride[0]))
					for j in range(8):
						temp.append(" ")

				temp.append("{0:e}".format(l.params))
				temp.append("{0:e}".format(l.add_ops))
				temp.append("{0:e}".format(l.times_ops))
				temp.append("{0:e}".format(l.FLOPs))
				writer.writerow(temp)

	def calculate_all(self):
		self.calculate_output()
		self.calculate_FLOPs_and_params()



