from generator_keras import generator
from vgg16_discriminator import discriminator
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator as IDG
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input,Conv2D,BatchNormalization,PReLU,Reshape,Lambda
from pixel_shuffler import SubpixelConv2D
from tensorflow.python.keras.losses import mean_squared_error, mean_squared_logarithmic_error,logcosh
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.backend import tanh
from tensorflow.python.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
import os
from keras.models import load_model
from perceptual_loss import vgg
import numpy as np
import glob
import math
import random
from PIL import Image
from scipy.optimize import fsolve
from tensorflow import keras
import cv2
from keras.models import load_model
# Loading multiple instances of dataset
# Combining Loss/Create new model

# Dataset directory parameters


# Training 3rd one
# change Ckpt file
# Change batch to not less than 20
# change filter_kernal_no
# change size of y in load_data to 512
# Comment ckpt in generator.py


# Testing
# comment load_data
# comment .fit
# load ckpt file in gen
# Uncomment
# output to 2048

WORKSPACE_DIR = '/home/mdo2/sid_codes/new_codes'
dataset_dir =  '/home/mdo2/Downloads/Codes/init_data/mirflickr/'
ckpt_path = WORKSPACE_DIR + '/ckpt_1.h5'

in_1_size = [24,24,3]
in_2_size = [48,48,3]
in_3_size = [96,96,3]
out = [192,192,3]
n = np.arange(10000,100000,10000)
# Parameter Class
class parameters:
    def __init__(self):
        self.batch_size = 250
        self.epochs = 10000
        self.filter_kernal_no = 0
        self.gen_type = self.filter_kernal_no + 1
        self.input_shape = [64,64,3]
        self.inflow_shape = [128,128,3]
        self.filter_kernal = [[32,128,128,128,256],[32,128,128,128,128,256,512],[32,128,128,128,128,128,128,128,512]]
        

def load_data(path,flag=1,start=0,batch_size=25000):
    x1_train=[]
    y1_train=[]
    y2_train=[]
    y3_train=[]

    print("point 1")
    dirs = os.listdir( path )
    random.shuffle(dirs)
    count = start
    for item in dirs[start:start+batch_size]:
        if count<start+batch_size:
        	im = Image.open(path+item)
        	f, e = os.path.splitext(path+item)
        	imResize = im.resize((24,24), Image.ANTIALIAS)
        	imResize = np.array(imResize)
        	x1_train.append((imResize))
        	imResize = im.resize((48,48), Image.ANTIALIAS)
        	imResize = np.array(imResize)
        	y1_train.append(imResize)
        	imResize = im.resize((96,96), Image.ANTIALIAS)
        	imResize = np.array(imResize)
        	y2_train.append(imResize)
        	imResize = im.resize((192,192), Image.ANTIALIAS)
        	imResize = np.array(imResize)
        	y3_train.append(imResize)
        	print('Img Done',count+1) 
        count+=1
    if flag==0:
    	
        return np.array(x1_train).astype('uint8'),np.array(y1_train).astype('uint8')
    else:
    	return np.array(x1_train).astype('uint8'),np.array(y1_train).astype('uint8'),np.array(y2_train).astype('uint8'),np.array(y3_train).astype('uint8')


def gen_model(param_object):
    x_train , y_train = load_data(dataset_dir,flag=0) 
    _G = generator(param_object.gen_type, param_object.filter_kernal[param_object.filter_kernal_no], param_object.input_shape,optimizer='Adam',loss=mean_squared_error,path=None, inflow_layer = tf.zeros(shape=param_object.inflow_shape, dtype=tf.float32)).generator_model()
    opti_trick = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=100, verbose=1)
    ckpt =  ModelCheckpoint(ckpt_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, period=1)    
    _G.fit(x_train, y_train, epochs=param_object.epochs, batch_size=param_object.batch_size, callbacks=[opti_trick,ckpt], validation_split=0.2, shuffle=True)
    return _G

#param_obj = parameters()
#G = gen_model(param_obj)


class merge_models:   
   
    def __init__(self,epochs,batch_size):
        self.epochs = epochs
        self.batch_size = batch_size
        
            # Generator
        filter_kernal = [[32,128,128,128,256],[32,128,128,128,128,256,512],[32,128,128,128,128,128,128,128,512]]
       
        self.g_1 = generator(3,filter_kernal[2],in_1_size,'Adam','mse',path = '/home/mdo2/sid_codes/new_codes/gen_1.h5').generator_model()   
        self.g_2 = generator(2,filter_kernal[1],in_2_size,'Adam','mse',path = '/home/mdo2/sid_codes/new_codes/gen_2.h5').generator_model()
        self.g_3 = generator(1,filter_kernal[0],in_3_size,'Adam','mse',path = '/home/mdo2/sid_codes/new_codes/gen_3.h5').generator_model()

        # Discriminator
        #self.D1 = discriminator(in_2_size).modified_vgg()
        #self.D1.compile(loss='mse',optimizer='Adam',metrics=['accuracy'])
        #self.D2 = discriminator(in_3_size).modified_vgg()
        #self.D2.compile(loss='mse',optimizer='Adam',metrics=['accuracy'])
        self.D = discriminator(out).modified_vgg()
        self.D.compile(loss='mse',optimizer='Adam',metrics=['accuracy'])
        
        self.g = 0
        # Images
        lr = Input(shape=in_1_size)
        hr_1 = Input(shape=in_2_size)
        hr_2 = Input(shape=in_3_size)
        hr_3 = Input(shape=out)
        fake_hr_image_1 = self.g_1(lr)
        fake_hr_image_2 = self.g_2(fake_hr_image_1)
        fake_hr_image_3 = self.g_3(fake_hr_image_2)

        self.vgg_1 = vgg(in_2_size).vgg_loss_model()
        self.vgg_2 = vgg(in_3_size).vgg_loss_model()
        self.vgg_3 = vgg(out).vgg_loss_model()
       
        fake_hr_feat_1 = self.vgg_1(fake_hr_image_1)
        fake_hr_feat_2 = self.vgg_2(fake_hr_image_2)
        fake_hr_feat_3 = self.vgg_3(fake_hr_image_3)

        self.D.trainable = False
        #self.D1.trainable = False
        #self.D2.trainable = False
        validity = self.D(fake_hr_image_3)
        #self.D.load_weights('/media/arjun/119GB/attachments/D_best.h5')

        self.merged_net = Model(inputs=[lr,hr_1,hr_2,hr_3],outputs=[validity,fake_hr_feat_1,fake_hr_feat_2,fake_hr_feat_3])
        self.merged_net.compile(loss=['binary_crossentropy','mse','mse','mse'],loss_weights=[1e-3,.3,.3,.3],optimizer='Adam')
        self.merged_net.summary()
        #self.merged_net.load_weights('/home/mdo2/sid_codes/newew/my_model.h5')
        print('loaded')
        
    def save(self):
        #self.D.save_weights('D_best.h5')
        #print('Discriminator saved')
        
      	#self.merged_net.save('my_model.h5')
      	
      	self.merged_net.save('/home/mdo2/sid_codes/new_codes/model.h5')
      	self.g_1.save('/home/mdo2/sid_codes/new_codes/gen_1.h5')
      	self.g_2.save('/home/mdo2/sid_codes/new_codes/gen_2.h5')
      	self.g_3.save('/home/mdo2/sid_codes/new_codes/gen_3.h5')
      	print('saved')
	

    """

        
        
    def test(self):
        test_path, test_path = ''
        img_test_path = glob.glob(test_path)
        random.shuffle(img_test_path)
        d_sum = 0.
        g_sum = np.array([0.,0.,0.,0.,0.,0.,0.,0.])
       
        step_size = int(800/self.batch_size)
        for steps in range(step_size):
            # Load data
            print('Loading Test Batch')
            x_test, y_test_1, y_test_2, y_test_3 = load_data(img_test_path, start=steps*self.batch_size, batch_size=self.batch_size)
           
            fake_hr_1 = self.g_1.predict(x_test)
            fake_hr_2 = self.g_2.predict(fake_hr_1)
            fake_hr_3 = self.g_3.predict(fake_hr_2)
            valid = np.ones([self.batch_size,12,12,16])
            fake = np.zeros([self.batch_size,12,12,16])
           
            d_test_loss_real = self.D.test_on_batch(y_test_3, valid)
            d_test_loss_fake = self.D.test_on_batch(fake_hr_3, fake)
            d_test_loss = 0.5*np.add(d_test_loss_fake,d_test_loss_real)
       8
            real_feat_1 = self.vgg_1.predict(y_test_1)
            real_feat_2 = self.vgg_2.predict(y_test_2)
            real_feat_3 = self.vgg_3.predict(y_test_3)
            g_test_loss = self.merged_net.test_on_batch([x_test,y_test_3], [valid,real_feat_3])

            d_sum += d_test_loss[0]
            g_sum = np.add(g_sum, g_test_loss)

        save(d_sum/float(step_size),g_sum/float(step_size))
    """

    def train(self):
        try:
            #self.D.load_weights('/media/arjun/119GB/attachments/D_best.h5')
            #self.g_1.load_weights('/media/arjun/119GB/attachments/g_1_best.h5')

            #self.g_2.load_weights('/media/arjun/119GB/attachments/g_2_best.h5')
            #self.g_3.load_weights('/media/arjun/119GB/attachments/g_3_best.h5')
	
            print("Discriminator loaded")
            for epoch in range(self.epochs):
            	   
            	print("loading data")
            	train_path = dataset_dir
            	img_train_path = glob.glob(train_path)
            	i=0
            	if i==0:
            		avg_loss=0
            		i+=1
            	avg_loss = avg_loss/int(25000/self.batch_size)
            	
            	for steps in range(int(25000/self.batch_size)):
                    # Load data
                    print(epoch)
                    print('Loading Train Batch:',steps+1)
                    x_train, y_train_1, y_train_2, y_train_3 = load_data(train_path, start=steps*self.batch_size, batch_size=self.batch_size)
                    tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)


                    # Discriminator

                    
                    print('Training Discriminator')
                    '''
                    valid1 = np.ones([self.batch_size,1,1,1])
                    fake1 = np.zeros([self.batch_size,1,1,1])
                    valid2 = np.ones([self.batch_size,3,3,1])
                    fake2 = np.zeros([self.batch_size,3,3,1])
                   
                    
                    fake_hr_1 = self.g_1.predict(x_train)
                    fake_hr_feat_1 = self.vgg_1(fake_hr_1)
                    
                    #d_train_loss_real = self.D1.train_on_batch(y_train_1, valid1)
                    #d_train_loss_fake = self.D1.train_on_batch(fake_hr_1, fake1)
                    
                    fake_hr_2 = self.g_2.predict(fake_hr_1)
                    fake_hr_feat_2 = self.vgg_2(fake_hr_2)
                    
                    #d_train_loss_real = self.D2.train_on_batch(y_train_2, valid2)
                    #d_train_loss_fake = self.D2.train_on_batch(fake_hr_2, fake2)
                    
                    
                   
                    fake_hr_3 = self.g_3.predict(fake_hr_2)
                    fake_hr_feat_3 = self.vgg_3(fake_hr_3)

                    d_train_loss_fake = self.D.train_on_batch(fake_hr_3, fake)
                    d_train_loss_real = self.D.train_on_batch(y_train_3, valid)
		    

                    
                    #d_train_loss_real = self.D.train_on_batch(y_train_3, valid)
                    #d_train_loss_fake = self.D.train_on_batch(fake_hr_3, fake)
                    #print(d_train_loss_fake,' fake loss ')
                    #print(d_train_loss_real,' real loss ')
                    d_train_loss = 0.5*np.add(d_train_loss_fake,d_train_loss_real)
                    # Generator   
                    #valid = np.ones([self.batch_size,192,192,3])
                    '''
                    print('Training Generator')
                    valid = np.ones([self.batch_size,6,6,1])
                    fake = np.zeros([self.batch_size,6,6,1])
                    real_feat_1 = self.vgg_1.predict(y_train_1)
                    real_feat_2 = self.vgg_2.predict(y_train_2)
                    real_feat_3 = self.vgg_3.predict(y_train_3)
                    d_train_loss=0
                    #cv2.imshow('image',real_feat_3)
                    
                    g_loss = self.merged_net.train_on_batch([x_train,y_train_1,y_train_2,y_train_3], [valid,real_feat_1,real_feat_2,real_feat_3])
                    
                    #g_loss=0
                    #d_train_loss = 0
                    
                    print(g_loss,d_train_loss)
                    avg_loss+=g_loss[0]
                    #self.low = g_loss
                    
                    #print("aaaaaa",K.eval(K.sum(log_loss_1)))
                    print('*******************************************************\n')
                    if self.g == 0:
                    	
                    	try:
                    		self.old_loss = np.load('/home/mdo2/sid_codes/newew/g_loss.npy')
                    		print('latest gen')
                    		self.g+=1
                    	except:
                    		old_loss = float('inf')
                    		self.g+=1
                    if g_loss[0]< self.old_loss:
                    	self.old_loss = g_loss[0]
                    	print('saving best')
                    	np.save	('g_loss',g_loss[0])
                    	self.g+=1
                    	self.save()
                    	
                    #if (steps+1)%100==0:
                    #    self.save()   
                    #self.merged_net.inputs = [x_train,y_train_1,y_train_2,y_train_3]
                    #self.merged_net.outputs = [valid,fake_hr_feat_1,fake_hr_feat_2,fake_hr_feat_3]
                    #c_loss = K.sqrt(K.mean((fake_hr_feat_1 - real_feat_1)**2))
                    #log_loss = logcosh(fake_hr_feat_1,real_feat_1)
                    #grads = K.gradients([log_loss,c_loss],[fake_hr_feat_1,real_feat_1])[0]
                    #print(grads)
                    #iterate = K.function([(y_train_1)], [c_loss, grads])
                    #print(iterate,'c_loss')
                    #iterate = K.function([tf.convert_to_tensor(y_train_1)], [log_loss, grads])
                    #print(iterate,'log_loss_1')
                    #self.D.fit(y_train_3,valid, epochs=epoch, batch_size=self.batch_size, callbacks=[tbCallBack])
                   
                    
        except KeyboardInterrupt:
            print('Saving Weights')
            #self.save()           
    def feed_forward(self):
    	in_1 = [64,64,3]
    	in_2 = [64,64,3]
    	in_3 = [64*4,64*4,3]
    	out_ = [128,128,3]
    	img = cv2.resize(cv2.imread('/home/mdo2/sid_codes/datasets_big/qhd_no_pad/train/1.png',1),(64,64))
    	img = img.reshape(1,64,64,3)
    	filter_kernal = [[32,128,128,128,256],[32,128,128,128,128,256,512],[32,128,128,128,128,128,128,128,512]]
    	g3 = generator(3,filter_kernal[2],in_1,'Adam',path = '/home/mdo2/sid_codes/new_codes/gen_1.h5').generator_model()
    	#g2 = generator(2,filter_kernal[1],in_2,'Adam',path = '/home/mdo2/sid_codes/new_codes/gen_2.h5').generator_model()
    	#g1 = generator(1,filter_kernal[0],in_1,'Adam',path = '/home/mdo2/sid_codes/new_codes/gen_3.h5').generator_model()
    	img = g3.predict(img)
    	#img = g2.predict(img)
    	img = np.array(img).astype('uint8')
    	img = img.reshape(128,128,3)
    	cv2.imshow('img',img)
    	cv2.waitKey(0)
    	cv2.imwrite('ee.jpg',img)
	#img = g2.predict(img)
	#img = g3.predict(img)
	
	

merge_models(100,100).feed_forward()


def dice_coef(y_true, y_pred, smooth, thresh):
    y_pred = y_pred > thresh
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    #c_loss_1 = self.merged_net.compile(loss=['binary_crossentropy','mse','mse','mse'],loss_weights=[1e-3,.3,.3,.3],optimizer='Adam')
    #grads_1 = K.gradients(c_loss_1,self.merged_net.outputs)[0]
    #iterate = K.function([tf.convert_to_tensor(self.merged_net.inputs)], [c_loss_1, grads_1])
    #print(iterate,'c_loss_1')
    #print('new grads',grads_1)
    





