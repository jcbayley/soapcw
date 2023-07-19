#!/home/joseph.bayley/.virtualenvs/soap27/bin/python
from keras.models import Sequential, load_model, Input, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, concatenate, BatchNormalization, LeakyReLU
from keras.utils import Sequence
from keras.constraints import nonneg
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import keras
import plot_and_sigfits as psf

#os.environ['LD_LIBRARY_PATH'] = "/usr/local/cuda-10.0/lib64/"

def model_sig(image_shape = (156,89)):
    
    inputim = Input(shape=(image_shape[0], image_shape[1],1),name="image_input")

    ############
    # convolutional network for image
    ###############
    
    #img = Conv2D(32,(11,11), name= "img_conv")(inputim)
    #img = LeakyReLU(alpha=0.1,name="img_convact")(img)
    #img = MaxPooling2D(pool_size=(2, 2),name="maxpool")(img)
    
    img = Conv2D(8,(5,5),name="conv0")(inputim)
    img = LeakyReLU(alpha=0.1,name="conv0_leakyrelu")(img)
    img = MaxPooling2D(pool_size=(8, 8),name="maxpool_0")(img)
    
    img = Conv2D(8,(3,3),name="conv1")(img)
    img = LeakyReLU(alpha=0.1,name="conv1_leakyrelu")(img)
    img = MaxPooling2D(pool_size=(8, 8),name="maxpool_1")(img)
    
    img = Flatten(name="flatten")(img) 
    
    
    img = Dense(8,name="dense1")(img)
    img = LeakyReLU(alpha=0.1,name="dense1_leakyrelu")(img)
    img = Dropout(0.5)(img)
    
    img = Dense(8,name="dense2")(img)
    img = LeakyReLU(alpha=0.1,name="dense2_leakyrelu")(img)
    img = Dropout(0.5)(img)
    
    img = Dense(8,name="dense3")(img)
    img = LeakyReLU(alpha=0.1,name="dense3_leakyrelu")(img)
    img = Dropout(0.5)(img)
    
    img = Dense(1,name="out",activation="sigmoid")(img)
    #img = LeakyReLU(alpha=0.1,name="img_dense3act")(img)
    
    model = Model(inputs=inputim, output=img)
    
    #model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['accuracy'],loss_weights=[1., 0.0,0.0])
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    return model

def model_lin():
    
    inputim = Input(shape=(psf.image_shape[0], psf.image_shape[1],1),name="image_input")

    ############
    # convolutional network for image
    ###############
    
    #img = Conv2D(32,(11,11), name= "img_conv")(inputim)
    #img = LeakyReLU(alpha=0.1,name="img_convact")(img)
    #img = MaxPooling2D(pool_size=(2, 2),name="maxpool")(img)
    
    img = Conv2D(8,(5,5),name="img_conv2")(inputim)
    img = LeakyReLU(alpha=0.1)(img)
    img = MaxPooling2D(pool_size=(2, 2))(img)
    
    img = Conv2D(8,(3,3),name="img_conv3")(img)
    img = LeakyReLU(alpha=0.1)(img)
    img = MaxPooling2D(pool_size=(2, 2))(img)
    
    img = Flatten(name="flatten")(img) 
    
    
    img = Dense(16,name="img_dense1")(img)
    img = LeakyReLU(alpha=0.1,name="img_dense1act")(img)
    img = Dropout(0.5)(img)
    
    img = Dense(16,name="img_dense2")(img)
    img = LeakyReLU(alpha=0.1,name="img_dense2act")(img)
    img = Dropout(0.5)(img)
    
    img = Dense(8,name="img_dense3")(img)
    img = LeakyReLU(alpha=0.1,name="img_dense3act")(img)
    img = Dropout(0.5)(img)
    
    img = Dense(1,name="img_out",activation="relu")(img)
    #img = LeakyReLU(alpha=0.1,name="img_dense3act")(img)
    
    model = Model(inputs=inputim, output=img)
    
    #model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['accuracy'],loss_weights=[1., 0.0,0.0])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    
    return model

