#!/home/joseph.bayley/.virtualenvs/soap27/bin/python
from keras.models import Sequential, load_model, Input, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, concatenate, BatchNormalization, LeakyReLU
from keras.utils import Sequence
from keras.constraints import nonneg
from keras import backend as K
import keras
import argparse
import plot_and_sigfits as psf
import os

#os.environ['LD_LIBRARY_PATH'] = "/usr/local/cuda-10.0/lib64/"


def model_sig():
    
    inputim = Input(shape=(psf.image_shape[0], psf.image_shape[1],1),name="image_input")
    inputHL = Input(shape=(psf.image_shape[0], psf.image_shape[1],2),name="HL_input")
    
    
    #################
    # convolutional network for image
    #################
    
    img = Conv2D(8,(5,5), name= "img_conv0")(inputim)
    img = LeakyReLU(alpha=0.1,name="img_convact0")(img)
    img = MaxPooling2D(pool_size=(4, 4),name="maxpool0")(img)
    
    #img = Conv2D(16,(5,5), name= "img_conv")(inputim)
    #img = LeakyReLU(alpha=0.1,name="img_convact")(img)
    #img = MaxPooling2D(pool_size=(2, 2),name="maxpool")(img)
    
    img = Conv2D(8,(3,3),name="img_conv2")(img)
    img = LeakyReLU(alpha=0.1)(img)
    img = MaxPooling2D(pool_size=(4, 4))(img)
    
    #img = Conv2D(8,(3,3),name="img_conv3")(img)
    #img = LeakyReLU(alpha=0.1)(img)
    #img = MaxPooling2D(pool_size=(2, 2))(img)
    
    img = Flatten(name="flatten")(img)   

    #img = Dense(256)(img)
    #img = LeakyReLU(alpha=0.1)(img)
    #img = Dropout(0.1)(img)
    
    img = Dense(8,name="img_dense1")(img)
    img = LeakyReLU(alpha=0.1,name="img_dense1act")(img)
    img = Dropout(0.5)(img)
    
    img = Dense(8,name="img_dense2")(img)
    img = LeakyReLU(alpha=0.1,name="img_dense3act")(img)
    img = Dropout(0.5)(img)
    
    img = Dense(8,name="img_out",activation="sigmoid")(img)
    #img = LeakyReLU(alpha=0.1,name="img_dense3act")(img)
    
    img = Model(inputs=inputim, output=img)
    
    ##############
    # convolutional network for detector data
    ###############
    
    imgHL = Conv2D(8,(5,5), name= "imghl_conv0")(inputHL)
    imgHL = LeakyReLU(alpha=0.1,name="imghl_convac0")(imgHL)
    imgHL = MaxPooling2D(pool_size=(4, 4),name="maxpoolh0")(imgHL)
    
    #imgHL = Conv2D(16,(5,5), name= "imghl_conv")(inputHL)
    #imgHL = LeakyReLU(alpha=0.1,name="imghl_convact")(imgHL)
    #imgHL = MaxPooling2D(pool_size=(2, 2),name="maxpoolhl")(imgHL)
    
    imgHL = Conv2D(8,(3,3),name="imghl_conv2")(imgHL)
    imgHL = LeakyReLU(alpha=0.1)(imgHL)
    imgHL = MaxPooling2D(pool_size=(4, 4))(imgHL)
    
    #img = Conv2D(8,(3,3),name="img_conv3")(img)
    #img = LeakyReLU(alpha=0.1)(img)
    #img = MaxPooling2D(pool_size=(2, 2))(img)
    
    imgHL = Flatten(name="flattenhl")(imgHL)   

    #imgHL = Dense(256)(imgHL)
    #imgHL = LeakyReLU(alpha=0.1)(imgHL)
    #imgHL = Dropout(0.1)(imgHL)
    
    imgHL = Dense(8,name="imghl_dense1")(imgHL)
    imgHL = LeakyReLU(alpha=0.1,name="imghl_dense1act")(imgHL)
    imgHL = Dropout(0.5)(imgHL)
    
    imgHL = Dense(8,name="imghl_dense2")(imgHL)
    imgHL = LeakyReLU(alpha=0.1,name="imghl_dense3act")(imgHL)
    imgHL = Dropout(0.5)(imgHL)
    
    imgHL = Dense(8,name="imghl_out",activation="sigmoid")(imgHL)
    
    imgHL = Model(inputs=inputHL, output=imgHL)
    
    
    #################
    # combine the outputs from all models
    ################
    
    combined = concatenate([img.output, imgHL.output])
    
    z = Dense(4)(combined)
    z = Dropout(0.5)(z)
    z = LeakyReLU(alpha=0.1)(z)
    
    z = Dense(4)(z)
    z = Dropout(0.5)(z)
    z = LeakyReLU(alpha=0.1)(z)
    z = Dense(1, activation = "sigmoid",name="final_out")(combined)
    
    # setup final model which takes in inputs for each model
    
    model = Model(inputs=[img.input, imgHL.input], output=[z])
    
    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy',psf.sensitivity_at_specificity(0.99)])
    
    return model

def model_lin():
    
    inputim = Input(shape=(psf.image_shape[0], psf.image_shape[1],1),name="image_input")
    inputHL = Input(shape=(psf.image_shape[0], psf.image_shape[1],2),name="HL_input")
    
    
    #################
    # convolutional network for image
    #################
    
    img = Conv2D(8,(5,5), name= "img_conv0")(inputim)
    img = LeakyReLU(alpha=0.1,name="img_convact0")(img)
    img = MaxPooling2D(pool_size=(2, 2),name="maxpool0")(img)
    
    #img = Conv2D(16,(5,5), name= "img_conv")(inputim)
    #img = LeakyReLU(alpha=0.1,name="img_convact")(img)
    #img = MaxPooling2D(pool_size=(2, 2),name="maxpool")(img)
    
    img = Conv2D(8,(3,3),name="img_conv2")(img)
    img = LeakyReLU(alpha=0.1)(img)
    img = MaxPooling2D(pool_size=(2, 2))(img)
    
    #img = Conv2D(8,(3,3),name="img_conv3")(img)
    #img = LeakyReLU(alpha=0.1)(img)
    #img = MaxPooling2D(pool_size=(2, 2))(img)
    
    img = Flatten(name="flatten")(img)   

    #img = Dense(256)(img)
    #img = LeakyReLU(alpha=0.1)(img)
    #img = Dropout(0.1)(img)
    
    img = Dense(16,name="img_dense1")(img)
    img = LeakyReLU(alpha=0.1,name="img_dense1act")(img)
    img = Dropout(0.5)(img)
    
    img = Dense(16,name="img_dense2")(img)
    img = LeakyReLU(alpha=0.1,name="img_dense3act")(img)
    img = Dropout(0.5)(img)
    
    img = Dense(8,name="img_out",activation="sigmoid")(img)
    #img = LeakyReLU(alpha=0.1,name="img_dense3act")(img)
    
    img = Model(inputs=inputim, output=img)
    
    ##############
    # convolutional network for detector data
    ###############
    
    imgHL = Conv2D(8,(5,5), name= "imghl_conv0")(inputHL)
    imgHL = LeakyReLU(alpha=0.1,name="imghl_convac0")(imgHL)
    imgHL = MaxPooling2D(pool_size=(2, 2),name="maxpoolh0")(imgHL)
    
    #imgHL = Conv2D(16,(5,5), name= "imghl_conv")(inputHL)
    #imgHL = LeakyReLU(alpha=0.1,name="imghl_convact")(imgHL)
    #imgHL = MaxPooling2D(pool_size=(2, 2),name="maxpoolhl")(imgHL)
    
    imgHL = Conv2D(8,(3,3),name="imghl_conv2")(imgHL)
    imgHL = LeakyReLU(alpha=0.1)(imgHL)
    imgHL = MaxPooling2D(pool_size=(2, 2))(imgHL)
    
    #img = Conv2D(8,(3,3),name="img_conv3")(img)
    #img = LeakyReLU(alpha=0.1)(img)
    #img = MaxPooling2D(pool_size=(2, 2))(img)
    
    imgHL = Flatten(name="flattenhl")(imgHL)   

    #imgHL = Dense(256)(imgHL)
    #imgHL = LeakyReLU(alpha=0.1)(imgHL)
    #imgHL = Dropout(0.1)(imgHL)
    
    imgHL = Dense(16,name="imghl_dense1")(imgHL)
    imgHL = LeakyReLU(alpha=0.1,name="imghl_dense1act")(imgHL)
    imgHL = Dropout(0.5)(imgHL)
    
    imgHL = Dense(16,name="imghl_dense2")(imgHL)
    imgHL = LeakyReLU(alpha=0.1,name="imghl_dense3act")(imgHL)
    imgHL = Dropout(0.5)(imgHL)
    
    imgHL = Dense(8,name="imghl_out",activation="relu")(imgHL)
    
    imgHL = Model(inputs=inputHL, output=imgHL)
    
    
    #################
    # combine the outputs from all models
    ################
    
    combined = concatenate([img.output, imgHL.output])
    
    z = Dense(4)(combined)
    z = Dropout(0.5)(z)
    z = LeakyReLU(alpha=0.1)(z)
    
    z = Dense(4)(z)
    z = Dropout(0.5)(z)
    z = LeakyReLU(alpha=0.1)(z)
    z = Dense(1, activation = "relu",name="final_out")(combined)
    
    # setup final model which takes in inputs for each model
    
    model = Model(inputs=[img.input, imgHL.input], output=[z])
    
    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
    
    return model
 

def model_init(vitmap_model,img_model):
    
    #inputstat = Input(shape=(1,),name = "stat_input")
    #inputim = Input(shape=(psf.image_shape[0], psf.image_shape[1],1),name="image_input")
    #inputHL = Input(shape=(psf.image_shape[0], psf.image_shape[1],2),name="HL_input")
    trainable = False
    
    img = load_model(os.path.join(vitmap_model,"training_model.h5"))
    for layer in img.layers:
        layer.name += "_vitmap"
        layer.trainable = trainable
    img.layers.pop()
    
    #img.summary()
    imgHL = load_model(os.path.join(img_model,"training_model.h5"))
    for layer in imgHL.layers:
        layer.name += "_sft"
        layer.trainable = trainable
    imgHL.layers.pop()
    #imgHL.summary()

        
    #################
    # combine the outputs from all models
    ################
    
    combined = concatenate([img.output, imgHL.output])
    
    #z = Dense(4)(combined)
    #z = Dropout(0.5)(z)
    #z = LeakyReLU(alpha=0.1)(z)
    
    #z = Dense(8)(combined)
    #z = Dropout(0.5)(z)
    #z = LeakyReLU(alpha=0.1)(z)
    z = Dense(1, activation = "sigmoid",name="all_out")(combined)
    
    # setup final model which takes in inputs for each model
    
    model = Model(inputs=[img.input, imgHL.input], output=[z])
    
    model.summary()
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    return model

