#!/home/joseph.bayley/.virtualenvs/soap27/bin/python
from keras.models import Sequential, load_model, Input, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, concatenate, BatchNormalization, LeakyReLU
from keras.utils import Sequence
from keras.constraints import nonneg
from keras import backend as K
import keras
import plot_and_sigfits as psf

def model():
    
    inputstat = Input(shape=(1,),name = "stat_input")
    
    #############
    # simple network for statistic
    ###############
    
    #stat = Dense(16)(inputstat)
    #stat = LeakyReLU(alpha=0.1)(stat)

    stat = Dense(1,name="stat_out",activation="sigmoid")(inputstat)
    #stat = LeakyReLU(alpha=0.1,name = "stat_act")(stat)

    model = Model(inputs=inputstat, output=stat)
    
    
    #model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['accuracy'],loss_weights=[1., 0.0,0.0])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model




