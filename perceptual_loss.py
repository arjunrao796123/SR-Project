from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
	
class vgg:
	def __init__(self,shape):
		self.shape = shape	

	def vgg_loss_model(self):
		vgg_in = Input(shape=self.shape)
		vgg_model = VGG19(include_top=False, input_tensor=vgg_in)
		vgg_out = vgg_model.get_layer('block3_conv1').output

		vgg = Model(vgg_in,vgg_out)
		for layer in vgg.layers:
			layer.trainable=False

		return vgg

