##############################################################################################
#	Created by Fuyuan Lyu Tommy at 2018-12/18
#	MIT License
#   Contain Network in Network Structure
##############################################################################################


import util.layer as layer
import csv


class NIN(object):
	def __init__(self):
		self.conv1 = layer.ConvLayer('conv1',input_k=32,input_channel=3,kernel=5,\
										output_channel=192,padding=2)
		self.conv1.calculate_all()	
		self.conv2 = layer.ConvLayer('conv2',kernel=1,output_channel=160)
		self.conv2.inherit(self.conv1)
		self.conv2.calculate_all()
		self.conv3 = layer.ConvLayer('conv3',kernel=1,output_channel=96)
		self.conv3.inherit(self.conv2)
		self.conv3.calculate_all()
		self.pool1 = layer.PoolLayer('pool1',kernel=3,stride=2,padding=1)
		self.pool1.inherit(self.conv3)
		self.pool1.calculate_all()


		self.conv4 = layer.ConvLayer('conv4',kernel=5,output_channel=192,padding=2)
		self.conv4.inherit(self.pool1)
		self.conv4.calculate_all()
		self.conv5 = layer.ConvLayer('conv5',kernel=1,output_channel=192)
		self.conv5.inherit(self.conv4)
		self.conv5.calculate_all()
		self.conv6 = layer.ConvLayer('conv6',kernel=1,output_channel=192)
		self.conv6.inherit(self.conv5)
		self.conv6.calculate_all()
		self.pool2 = layer.PoolLayer('pool2',kernel=3,stride=2,padding=1)
		self.pool2.inherit(self.conv6)
		self.pool2.calculate_all()


		self.conv7 = layer.ConvLayer('conv7',kernel=3,output_channel=192,padding=1)
		self.conv7.inherit(self.pool2)
		self.conv7.calculate_all()
		self.conv8 = layer.ConvLayer('conv8',kernel=1,output_channel=192)
		self.conv8.inherit(self.conv7)
		self.conv8.calculate_all()
		self.conv9 = layer.ConvLayer('conv9',kernel=1,output_channel=10)
		self.conv9.inherit(self.conv8)
		self.conv9.calculate_all()
		self.pool3 = layer.PoolLayer('pool3',kernel=8)
		self.pool3.inherit(self.conv9)
		self.pool3.calculate_all()

		self.all_layer = []
		self.all_layer.append(self.conv1)
		self.all_layer.append(self.conv2)
		self.all_layer.append(self.conv3)
		self.all_layer.append(self.pool1)

		self.all_layer.append(self.conv4)
		self.all_layer.append(self.conv5)
		self.all_layer.append(self.conv6)
		self.all_layer.append(self.pool2)

		self.all_layer.append(self.conv7)
		self.all_layer.append(self.conv8)
		self.all_layer.append(self.conv9)
		self.all_layer.append(self.pool3)


	def write_csv(self,filename="nin.csv"):
		with open(filename,'w') as csvFile:
			writer = csv.writer(csvFile,lineterminator='\n')
			writer.writerow(["name","input","filter","stride","padding","dilation",\
							"output","params","add_ops","times_ops","FLOPs"])
			for i, l in enumerate(self.all_layer):
				temp = []
				temp.append(l.name)
				temp.append("{}x{}x{}".format(l.input_k[0],l.input_k[1],l.input_channel))
				temp.append("{}x{}x{}x{}".format(l.kernel[0],l.kernel[1],l.input_channel,l.output_channel))
				temp.append("{}x{}".format(l.stride[0],l.stride[1]))
				temp.append("{}x{}".format(l.padding[0],l.padding[1]))
				temp.append("{}x{}".format(l.dilation[0],l.dilation[1]))
				temp.append("{}x{}x{}".format(l.output_k[0],l.output_k[1],l.output_channel))
				temp.append("{0:e}".format(l.params))
				temp.append("{0:e}".format(l.add_ops))
				temp.append("{0:e}".format(l.times_ops))
				temp.append("{0:e}".format(l.FLOPs))
				writer.writerow(temp)