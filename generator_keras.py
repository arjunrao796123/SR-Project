import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense,Input,Conv2D,Reshape,BatchNormalization,PReLU,Add,Lambda
from pixel_shuffler import SubpixelConv2D
from tensorflow.python.keras.optimizers import *
import numpy as np

from keras.models import load_model

class generator:
	def __init__(self, type_no, filter_array, input_shape,optimizer=None, loss=None,path=None, inflow_layer=None,flag=False):
		print('Generator'+str(type_no))
		self.type_no = type_no
		self.filter_array = filter_array
		self.input_shape = input_shape
		self.path = path
		self.optimizer = optimizer
		self.loss = loss
		#self.path = path
		self.inflow_layer = inflow_layer
		self.flag = flag

# 1 + 2 +3 +4(+3)+ check
#gen1
#+4(+2) +5 check
#gen2
#5(+4)+6(+5)(+2)+7  
#gen3
#5(+4)+6(+5)+7(+6)+8(+2)+9
	def generator_model(self):
		
		input_image = Input(shape=self.input_shape)
		#32
		layer_1 = Conv2D(self.filter_array[0],kernel_size=(3,3), padding="SAME")(input_image)
		layer_1 = BatchNormalization()(layer_1)
		layer_1 = PReLU(shared_axes=[1,2])(layer_1)
		
		#128
		layer_2 = Conv2D(self.filter_array[1],kernel_size=(3,3), padding="SAME")(layer_1)
		layer_2 = BatchNormalization()(layer_2)
		layer_2 = PReLU(shared_axes=[1,2])(layer_2)

		#128
		layer_3 = Conv2D(self.filter_array[2],kernel_size=(3,3), padding="SAME")(layer_2)
		layer_3 = BatchNormalization()(layer_3)
		layer_3 = PReLU(shared_axes=[1,2])(layer_3)

		#128
		layer_4 = Conv2D(self.filter_array[3],kernel_size=(3,3), padding="SAME")(layer_3)
		layer_4 = BatchNormalization()(layer_4)
		layer_4 = Add()([PReLU(shared_axes=[1,2])(layer_4),layer_3])


		

		if self.type_no==3:
			#128
			layer_5 = Conv2D(self.filter_array[4], kernel_size=(3,3), padding="SAME")(layer_4)
			layer_5 = BatchNormalization()(layer_5)
			layer_5 = Add()([PReLU(shared_axes=[1,2])(layer_5),layer_4])		
			
			#128
			layer_6 = Conv2D(self.filter_array[5], kernel_size=(3,3), padding="SAME")(layer_5)
			layer_6 = BatchNormalization()(layer_6)
			layer_6 = Add()([PReLU(shared_axes=[1,2])(layer_6),layer_5])	
			layer_7 = Add()([layer_6,layer_2])

			#128
			layer_7 = Conv2D(self.filter_array[6], kernel_size=(5,5), padding="SAME")(layer_6)
			layer_7 = BatchNormalization()(layer_7)
			#layer_7 = Add()([PReLU(shared_axes=[1,2])(layer_7),layer_6])
			
			#512
			layer_8 = Conv2D(self.filter_array[7], kernel_size=(5,5), padding="SAME")(layer_7)
			layer_8 = BatchNormalization()(layer_8)
			layer_8 = PReLU(shared_axes=[1,2])(layer_8)
			layer_8 = Add()([PReLU(shared_axes=[1,2])(layer_8),layer_7])	


			#if self.inflow_layer!=None:
			#	layer_8 = Add()([layer_8,self.inflow_layer])

			#512
			layer_9 = Conv2D(self.filter_array[8], kernel_size=(7,7), padding="SAME")(layer_8)

			layer_9 = SubpixelConv2D(layer_9.shape,scale=2)(layer_9)
			layer_9 = PReLU(shared_axes=[1,2])(layer_9)
			

		
			layer = layer_9


		elif self.type_no==1:
			#128
			layer_4 = Add()([PReLU(shared_axes=[1,2])(layer_4),layer_2])
			layer_5 = Conv2D(self.filter_array[4], kernel_size=(3,3), padding="SAME")(layer_4)
			layer_5 = BatchNormalization()(layer_5)

			layer_5 = PReLU(shared_axes=[1,2])(layer_5)

			#	layer_5 = Add()([PReLU(shared_axes=[1,2])(layer_5),layer_4]		
			
			#256
			layer_5 = SubpixelConv2D(self.input_shape,scale=2)(layer_5) 

			layer = layer_5

		elif self.type_no==2:


			layer_4 = Add()([PReLU(shared_axes=[1,2])(layer_4),layer_2])
			#256
			layer_5 = Conv2D(self.filter_array[4], kernel_size=(5,5), padding="SAME")(layer_4)
			layer_5 = BatchNormalization()(layer_5)
			layer_5 = PReLU(shared_axes=[1,2])(layer_5)
			layer_6 = Conv2D(self.filter_array[4], kernel_size=(5,5), padding="SAME")(layer_5)
			layer_6 = BatchNormalization()(layer_6)
			layer_6 = PReLU(shared_axes=[1,2])(layer_6)
			layer_6 = Add()([PReLU(shared_axes=[1,2])(layer_6),layer_5])	



			
			#if self.inflow_layer!=None:
				#layer_5 = Add()([layer_5, self.inflow_layer]) 	
			
			#512
			layer_7 = Conv2D(self.filter_array[6], kernel_size=(7,7), padding="SAME")(layer_6)
			layer_7 = SubpixelConv2D(layer_7.shape,scale=2)(layer_7)
			layer_7 = PReLU(shared_axes=[1,2])(layer_7)

			layer = layer_7
		
		
		out_layer = Conv2D(3, kernel_size=(1,1), activation='tanh')(layer)
		out_layer = Lambda(lambda x:(x+1)*127.5)(out_layer)
		model = Model(inputs=input_image,outputs=out_layer)
		outflow_layer = layer
		if self.path!=None:
				model.load_weights(self.path)
				print('Loaded g_%s weights',str(self.type_no))
				
				return model
		if self.flag:
			model.summary()
			return model 
		
		else:
			model.compile(loss = self.loss, optimizer = self.optimizer)
			model.summary()
			#model = load_model('/home/mdo2/sid_codes/new_codes/model.h5')
			#if self.path!=None:
				#model.load_weights(self.path)
				#print('Loaded g_%s weights',str(self.type_no))
		
			return model
		

if __name__ == '__main__':

	
	x_1 = generator(1,[32,128,128,128,256],[64,64,3]).generator_model()

