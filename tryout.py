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
import tensorflow.keras.backend as K
from perceptual_loss import vgg
import numpy as np
import glob
import cv2
import random
from scipy.optimize import fsolve

# Loading multiple instances of dataset
# Combining Loss/Create new model

# Dataset directory parameters
WORKSPACE_DIR = '/media/arjun/119GB/Proj/Thesis_master/'
dataset_dir =  '/media/arjun/119GB/Proj/vallarge/val_large'
ckpt_path = WORKSPACE_DIR + '/ckpt_1.h5'

in_1_size = [24,24,3]
in_2_size = [48,48,3]
in_3_size = [96,96,3]
out = [192,192,3]
n = np.arange(10000,100000,10000)
# Parameter Class
class parameters:
    def __init__(self):
        self.batch_size = 8
        self.epochs = 10000
        self.filter_kernal_no = 0
        self.gen_type = self.filter_kernal_no + 1
        self.input_shape = [64,64,3]
        self.inflow_shape = [128,128,3]
        self.filter_kernal = [[32,128,128,128,256],[32,128,128,128,128,256,512],[32,128,128,128,128,128,128,128,512]]
        

def load_data(path,start=0,batch_size=1,flag=1):
    x1_train=[]
    y1_train=[]
    y2_train=[]
    y3_train=[]

    images = sorted(glob.glob(path))
    random.shuffle(images)
    count = start
    for image in images:
        if count<start+batch_size:
            img = cv2.imread(image,1)
            x1_train.append(cv2.resize(img,(24,24)))
            y1_train.append(cv2.resize(img,(48,48)))
            y2_train.append(cv2.resize(img,(96,96)))
            y3_train.append(cv2.resize(img,(192,192)))
            print('Batch Done') 
        count+=1
    if flag==0:
        return np.array(x1_train).astype('uint8'),np.array(y1_train).astype('uint8')
    else:   
        return np.array(x1_train).astype('uint8'),np.array(y1_train).astype('uint8'),np.array(y2_train).astype('uint8'),np.array(y3_train).astype('uint8')


def gen_model(param_object):
    x_train , y_train = load_data(flag=0) 
    _G = generator(param_object.gen_type, param_object.filter_kernal[param_object.filter_kernal_no], param_object.input_shape, 'Adam', mean_squared_error, tf.zeros(shape=param_object.inflow_shape, dtype=tf.float32)).generator_model()
    opti_trick = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=100, verbose=1)
    ckpt =  ModelCheckpoint(ckpt_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, period=1)    
    _G.fit(x_train, y_train, epochs=param_object.epochs, batch_size=param_object.batch_size, callbacks=[opti_trick,ckpt], validation_split=0.2, shuffle=True)
    return _

#param_obj = parameters()
#G = gen_model(param_obj)


class merge_models:   
   
    def __init__(self,epochs,batch_size):
        self.epochs = epochs
        self.batch_size = batch_size
       
        # Generator
        filter_kernal = [[32,128,128,128,256],[32,128,128,128,128,256,512],[32,128,128,128,128,128,128,512,512]]
       
        self.g_1 = generator(3,filter_kernal[2],in_1_size,'Adam','mse').generator_model()   
        self.g_2 = generator(2,filter_kernal[1],in_2_size,'Adam','mse').generator_model()
        self.g_3 = generator(1,filter_kernal[0],in_3_size,'Adam','mse').generator_model()

        # Discriminator
        self.D = discriminator(out).modified_vgg()
        self.D.compile(loss='mse',optimizer='Adam',metrics=['accuracy'])

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
        valid = self.D(fake_hr_image_3)

        self.merged_net = Model(inputs=[lr,hr_1,hr_2,hr_3],outputs=[valid,fake_hr_feat_1,fake_hr_feat_2,fake_hr_feat_3])
        self.merged_net.compile(loss=['binary_crossentropy','mse','mse','mse'],loss_weights=[1e-3,.3,.3,.3],optimizer='Adam')
        self.merged_net.summary()

   
    def save(self):
        self.D.save_weights('D_best.h5')
        print('Discriminator saved')
        self.g_1.save_weights('g_1_best.h5')
        self.g_2.save_weights('g_2_best.h5')
        self.g_3.save_weights('g_3_best.h5')
        print('All generators saved')


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
            for epoch in range(self.epochs):

                train_path = dataset_dir +'/*.jpg'
                img_train_path = glob.glob(train_path)
                random.shuffle(img_train_path)

                for steps in range(int(25000/self.batch_size)):
                    # Load data
                    print('Loading Train Batch:',steps+1)
                    x_train, y_train_1, y_train_2, y_train_3 = load_data(train_path, start=steps*self.batch_size, batch_size=self.batch_size)


                    # Discriminator
                    print('Training Discriminator')
                   
                    fake_hr_1 = self.g_1.predict(x_train)
                    fake_hr_2 = self.g_2.predict(fake_hr_1)
                    fake_hr_3 = self.g_3.predict(fake_hr_2)
                    fake_hr_feat_1 = self.vgg_1(fake_hr_1)
                    fake_hr_feat_2 = self.vgg_1(fake_hr_2)
                    fake_hr_feat_3 = self.vgg_1(fake_hr_3)


                    valid = np.ones([self.batch_size,6,6,1])
                    fake = np.zeros([self.batch_size,6,6,1])
                   
                    d_train_loss_real = self.D.train_on_batch(y_train_3, valid)
                    d_train_loss_fake = self.D.train_on_batch(fake_hr_3, fake)
                    d_train_loss = 0.5*np.add(d_train_loss_fake,d_train_loss_real)
                    
                    # Generator   
                    print('Training Generator')
                    real_feat_1 = self.vgg_1.predict(y_train_1)
                    real_feat_2 = self.vgg_2.predict(y_train_2)
                    real_feat_3 = self.vgg_3.predict(y_train_3)
                    g_loss = self.merged_net.train_on_batch([x_train,y_train_1,y_train_2,y_train_3], [valid,real_feat_1,real_feat_2,real_feat_3])
                    print(g_loss,d_train_loss)
                    self.low = g_loss
                    #print("aaaaaa",K.eval(K.sum(log_loss_1)))
                    print('*******************************************************\n')
                    
                    if (steps+1)%100==0:
                        self.save()   
                    self.merged_net.inputs = [x_train,y_train_1,y_train_2,y_train_3]
                    self.merged_net.outputs = [valid,fake_hr_feat_1,fake_hr_feat_2,fake_hr_feat_3]
                    c_loss = K.sqrt(K.mean((fake_hr_feat_1 - real_feat_1)**2))
                    log_loss = logcosh(fake_hr_feat_1,real_feat_1)
                    grads = K.gradients([log_loss,c_loss],[fake_hr_feat_1,real_feat_1])[0]
                    print(grads)
                    iterate = K.function([tf.convert_to_tensor(y_train_1)], [c_loss, grads])
                    print(iterate,'c_loss')
                    iterate = K.function([tf.convert_to_tensor(y_train_1)], [log_loss, grads])
                    print(iterate,'log_loss_1')
                    
                    
        except KeyboardInterrupt:
            print('Saving Weights')
            #self.save()           
    


merge_models(100,1).train()


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
    





