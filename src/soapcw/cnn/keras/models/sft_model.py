#!/home/joseph.bayley/.virtualenvs/soap27/bin/python
from keras.models import Sequential, load_model, Input, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, concatenate, BatchNormalization, LeakyReLU, Conv3D, MaxPooling3D
from keras.utils import Sequence
from keras.constraints import nonneg
from keras.optimizers import Adam
from keras import backend as K
import keras
import plot_and_sigfits as psf



def model_sig(image_shape=(156,89)):
    
    inputHL = Input(shape=(image_shape[0], image_shape[1],2),name="HL_input")

    
    ##############
    # convolutional network for detector data
    ############### 

    imgHL = Conv2D(8,(5,5), name= "conv0")(inputHL)
    imgHL = LeakyReLU(alpha=0.1,name="conv0_leakyrelu")(imgHL)
    imgHL = MaxPooling2D(pool_size=(8, 8),name="maxpool_0")(imgHL)

    imgHL = Conv2D(8,(3,3), name= "conv1")(imgHL)
    imgHL = LeakyReLU(alpha=0.1,name="conv1_leakyrelu")(imgHL)
    imgHL = MaxPooling2D(pool_size=(8, 8),name="maxpool_1")(imgHL)
    
    #imgHL = Conv2D(32,(3,3),name="imghl_conv2")(imgHL)
    #imgHL = LeakyReLU(alpha=0.1)(imgHL)
    #imgHL = MaxPooling2D(pool_size=(2, 2))(imgHL)
    
    #img = Conv2D(8,(3,3),name="img_conv3")(img)
    #img = LeakyReLU(alpha=0.1)(img)
    #img = MaxPooling2D(pool_size=(2, 2))(img)
    
    imgHL = Flatten(name="flatten")(imgHL)   

    #imgHL = Dense(256)(imgHL)
    #imgHL = LeakyReLU(alpha=0.1)(imgHL)
    #imgHL = Dropout(0.1)(imgHL)
    
    imgHL = Dense(8,name="dense1")(imgHL)
    imgHL = LeakyReLU(alpha=0.1,name="dense1_leakyrelu")(imgHL)
    imgHL = Dropout(0.5)(imgHL)

    #imgHL = Dense(64,name="imghl_dense1r")(imgHL)
    #imgHL = LeakyReLU(alpha=0.1,name="imghl_dense1ract")(imgHL)
    #imgHL = Dropout(0.5)(imgHL)
    
    imgHL = Dense(8,name="dense2")(imgHL)
    imgHL = LeakyReLU(alpha=0.1,name="dense2_leakyrelu")(imgHL)
    imgHL = Dropout(0.5)(imgHL)
    
    #imgHL = Dense(32,name="imghl_dense3")(imgHL)
    #imgHL = LeakyReLU(alpha=0.1,name="imghl_dense3act")(imgHL)
    #imgHL = Dropout(0.5)(imgHL)
    
    imgHL = Dense(8,name="dense3")(imgHL)
    imgHL = LeakyReLU(alpha=0.1,name="dense3_leakyrelu")(imgHL)
    imgHL = Dropout(0.5)(imgHL)
    
    imgHL = Dense(1,name="out",activation="sigmoid")(imgHL)
    
    model = Model(inputs=inputHL, output=imgHL)
    
    adam = keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999)
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    
    return model

def model_oneim():
    
    inputHL = Input(shape=(image_shape[0], image_shape[1],1),name="HL_input")

    
    ##############
    # convolutional network for detector data
    ############### 

    imgHL = Conv2D(16,(5,5), name= "sft_imghl_conv0")(inputHL)
    imgHL = LeakyReLU(alpha=0.1,name="sft_imghl_convac0")(imgHL)
    imgHL = MaxPooling2D(pool_size=(2, 2),name="sft_maxpoolh0")(imgHL)
   
    imgHL = Conv2D(16,(3,3), name= "sft_imghl_conv1")(imgHL)
    imgHL = LeakyReLU(alpha=0.1,name="imghl_convac1")(imgHL)
    imgHL = MaxPooling2D(pool_size=(2, 2),name="maxpoolh1")(imgHL)
    
    #imgHL = Conv2D(32,(3,3),name="imghl_conv2")(imgHL)
    #imgHL = LeakyReLU(alpha=0.1)(imgHL)
    #imgHL = MaxPooling2D(pool_size=(2, 2))(imgHL)
    
    #img = Conv2D(8,(3,3),name="img_conv3")(img)
    #img = LeakyReLU(alpha=0.1)(img)
    #img = MaxPooling2D(pool_size=(2, 2))(img)
    
    imgHL = Flatten(name="flattenhl")(imgHL)   

    #imgHL = Dense(256)(imgHL)
    #imgHL = LeakyReLU(alpha=0.1)(imgHL)
    #imgHL = Dropout(0.1)(imgHL)
    
    #imgHL = Dense(128,name="imghl_dense1")(imgHL)
    #imgHL = LeakyReLU(alpha=0.1,name="imghl_dense1act")(imgHL)
    #imgHL = Dropout(0.5)(imgHL)

    #imgHL = Dense(64,name="imghl_dense1r")(imgHL)
    #imgHL = LeakyReLU(alpha=0.1,name="imghl_dense1ract")(imgHL)
    #imgHL = Dropout(0.5)(imgHL)
    
    imgHL = Dense(32,name="imghl_dense2")(imgHL)
    imgHL = LeakyReLU(alpha=0.1,name="imghl_dense2act")(imgHL)
    imgHL = Dropout(0.5)(imgHL)
    
    imgHL = Dense(8,name="imghl_dense3")(imgHL)
    imgHL = LeakyReLU(alpha=0.1,name="imghl_dense3act")(imgHL)
    imgHL = Dropout(0.5)(imgHL)
    
    imgHL = Dense(1,name="imghl_out",activation="sigmoid")(imgHL)
    
    model = Model(inputs=inputHL, output=imgHL)
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


def model_lin():
    
    inputHL = Input(shape=(image_shape[0], image_shape[1],2),name="HL_input")

    
    ##############
    # convolutional network for detector data
    ############### 

    imgHL = Conv2D(64,(15,15), name= "imghl_conv0")(inputHL)
    imgHL = LeakyReLU(alpha=0.1,name="imghl_convac0")(imgHL)
    imgHL = MaxPooling2D(pool_size=(2, 2),name="maxpoolh0")(imgHL)
   
    imgHL = Conv2D(32,(3,3), name= "imghl_conv1")(imgHL)
    imgHL = LeakyReLU(alpha=0.1,name="imghl_convac1")(imgHL)
    imgHL = MaxPooling2D(pool_size=(2, 2),name="maxpoolh1")(imgHL)
    
    #imgHL = Conv2D(32,(3,3),name="imghl_conv2")(imgHL)
    #imgHL = LeakyReLU(alpha=0.1)(imgHL)
    #imgHL = MaxPooling2D(pool_size=(2, 2))(imgHL)
    
    #img = Conv2D(8,(3,3),name="img_conv3")(img)
    #img = LeakyReLU(alpha=0.1)(img)
    #img = MaxPooling2D(pool_size=(2, 2))(img)
    
    imgHL = Flatten(name="flattenhl")(imgHL)   

    imgHL = Dense(256)(imgHL)
    imgHL = LeakyReLU(alpha=0.1)(imgHL)
    imgHL = Dropout(0.1)(imgHL)
    
    imgHL = Dense(128,name="imghl_dense1")(imgHL)
    imgHL = LeakyReLU(alpha=0.1,name="imghl_dense1act")(imgHL)
    imgHL = Dropout(0.5)(imgHL)

    imgHL = Dense(64,name="imghl_dense1r")(imgHL)
    imgHL = LeakyReLU(alpha=0.1,name="imghl_dense1ract")(imgHL)
    imgHL = Dropout(0.5)(imgHL)
    
    imgHL = Dense(32,name="imghl_dense2")(imgHL)
    imgHL = LeakyReLU(alpha=0.1,name="imghl_dense2act")(imgHL)
    imgHL = Dropout(0.5)(imgHL)
    
    imgHL = Dense(8,name="imghl_dense3")(imgHL)
    imgHL = LeakyReLU(alpha=0.1,name="imghl_dense3act")(imgHL)
    imgHL = Dropout(0.5)(imgHL)
    
    imgHL = Dense(1,name="imghl_out",activation="relu")(imgHL)
    
    model = Model(inputs=inputHL, output=imgHL)
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    
    return model


def model_3d():
    
    inputHL = Input(shape=(image_shape[0], image_shape[1],2),name="HL_input")

    
    ##############
    # convolutional network for detector data
    ############### 

    imgHL = Conv3D(16,(15,15,2), name= "imghl_conv0")(inputHL)
    imgHL = LeakyReLU(alpha=0.1,name="imghl_convac0")(imgHL)
    imgHL = MaxPooling3D(pool_size=(2, 2,2),name="maxpoolh0")(imgHL)
   
    #imgHL = Conv2D(64,(5,5), name= "imghl_conv1")(imgHL)
    #imgHL = LeakyReLU(alpha=0.1,name="imghl_convac1")(imgHL)
    #imgHL = MaxPooling2D(pool_size=(2, 2),name="maxpoolh1")(imgHL)
    
    imgHL = Conv3D(32,(3,3,2),name="imghl_conv2")(imgHL)
    imgHL = LeakyReLU(alpha=0.1)(imgHL)
    imgHL = MaxPooling3D(pool_size=(2, 2,2))(imgHL)
    
    #img = Conv2D(8,(3,3),name="img_conv3")(img)
    #img = LeakyReLU(alpha=0.1)(img)
    #img = MaxPooling2D(pool_size=(2, 2))(img)
    
    imgHL = Flatten(name="flattenhl")(imgHL)   

    imgHL = Dense(256)(imgHL)
    imgHL = LeakyReLU(alpha=0.1)(imgHL)
    imgHL = Dropout(0.1)(imgHL)
    
    imgHL = Dense(128,name="imghl_dense1")(imgHL)
    imgHL = LeakyReLU(alpha=0.1,name="imghl_dense1act")(imgHL)
    imgHL = Dropout(0.5)(imgHL)

    imgHL = Dense(64,name="imghl_dense1r")(imgHL)
    imgHL = LeakyReLU(alpha=0.1,name="imghl_dense1ract")(imgHL)
    imgHL = Dropout(0.5)(imgHL)
    
    imgHL = Dense(32,name="imghl_dense2")(imgHL)
    imgHL = LeakyReLU(alpha=0.1,name="imghl_dense2act")(imgHL)
    imgHL = Dropout(0.5)(imgHL)
    
    imgHL = Dense(8,name="imghl_dense3")(imgHL)
    imgHL = LeakyReLU(alpha=0.1,name="imghl_dense3act")(imgHL)
    imgHL = Dropout(0.5)(imgHL)
    
    imgHL = Dense(1,name="imghl_out",activation="sigmoid")(imgHL)
    
    model = Model(inputs=inputHL, output=imgHL)
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model



