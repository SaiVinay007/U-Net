import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow.keras.layers import (Activation, AveragePooling2D,
                                            BatchNormalization, Conv2D, Conv3D,
                                            Dense, Flatten,
                                            GlobalAveragePooling2D,
                                            GlobalMaxPooling2D, Input,
                                            MaxPooling2D, MaxPooling3D,
                                            Reshape, Dropout, concatenate,
											UpSampling2D,Input)

from tensorflow.keras import applications, regularizers, optimizers
from tensorflow.keras.models import Model, Sequential


def Unet(input_shape = (512,512,1)):

    inputs = Input(input_shape)

    conv1 = Conv2D(64,kernel_size=(3,3),activation='relu')(inputs)
    conv1 = Conv2D(64,kernel_size=(3,3),activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(128,kernel_size=(3,3),activation='relu')(pool1)
    conv2 = Conv2D(128,kernel_size=(3,3),activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = Conv2D(256,kernel_size=(3,3),activation='relu')(pool2)
    conv3 = Conv2D(256,kernel_size=(3,3),activation='relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

    conv4 = Conv2D(512,kernel_size=(3,3),activation='relu')(pool3)
    conv4 = Conv2D(512,kernel_size=(3,3),activation='relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2,2))(conv4)

    conv5 = Conv2D(1024,kernel_size=(3,3),activation='relu')(pool4)
    conv5 = Conv2D(1024,kernel_size=(3,3),activation='relu')(conv5)
    pool5 = MaxPooling2D(pool_size=(2,2))(conv5)

    up1 = Conv2D(512,kernel_size=(3,3),activation='relu')(UpSampling2D(size=(2,2))(pool5))
    merge1 = concatenate([pool4,up1])       # toral layer size = 512+512
    conv6 = Conv2D(512,kernel_size=(3,3),activation='relu')(merge1)
    conv6 = Conv2D(512,kernel_size=(3,3),activation='relu')(conv6)    

    up2 = Conv2D(256,kernel_size=(3,3),activation='relu')(UpSampling2D(size=(2,2))(conv6))
    merge2 = concatenate([pool3,up2])       # toral layer size = 256+256
    conv7 = Conv2D(256,kernel_size=(3,3),activation='relu')(merge2)
    conv7 = Conv2D(256,kernel_size=(3,3),activation='relu')(conv7)

    up3 = Conv2D(128,kernel_size=(3,3),activation='relu')(UpSampling2D(size=(2,2))(conv7))
    merge3 = concatenate([pool2,up3])       # toral layer size = 128+128
    conv8 = Conv2D(128,kernel_size=(3,3),activation='relu')(merge3)
    conv8 = Conv2D(128,kernel_size=(3,3),activation='relu')(conv8)

    up4 = Conv2D(64,kernel_size=(3,3),activation='relu')(UpSampling2D(size=(2,2))(conv8))
    merge4 = concatenate([pool1,up4])       # toral layer size = 64+64
    conv9 = Conv2D(64,kernel_size=(3,3),activation='relu')(merge4)
    conv9 = Conv2D(64,kernel_size=(3,3),activation='relu')(conv9)
    conv10 = Conv2D(2,kernel_size=(1,1))(conv9)

    model = Model(input = inputs,output = conv10)

    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss = 'mean_squared_error', optimizer = 'sgd')


    return model







