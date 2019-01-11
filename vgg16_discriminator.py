import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Input
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.optimizers import Adam

class discriminator:
	def __init__(self,shape):
		print('Discriminator')
		self.shape = shape
		

	def modified_vgg(self):
		vgg_in = Input(shape=self.shape)
		vgg_model = VGG16(include_top=False,input_tensor=vgg_in)	

		model = Sequential()
		for layer in vgg_model.layers:
			model.add(layer)
		
		for layer in model.layers:	
			layer.trainable = False	
		
		model.add(Dense(2048,activation='relu'))
		model.add(Dense(1024,activation='relu'))
		model.add(Dense(1024,activation='relu'))
		model.add(Dense(1,activation='sigmoid'))			
		model.compile(loss=binary_crossentropy,optimizer='Adam', metrics=['accuracy'])
		model.summary()
		return model


if __name__=="__main__":
	D = discriminator([64,64,3]).modified_vgg()
	D.summary()