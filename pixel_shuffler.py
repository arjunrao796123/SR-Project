from tensorflow.python.keras.layers import Lambda
import tensorflow as tf

def SubpixelConv2D(input_shape, scale=2):
	def subpixel_shape(input_shape):
		dims = [input_shape[0], input_shape[1]*scale, input_shape[2]*scale, int(input_shape[3]/(scale**2))]
		output_shape = tuple(dims)
		return output_shape

	def subpixel(x):
		return tf.depth_to_space(x,scale)	

	return Lambda(subpixel, output_shape=subpixel_shape)

