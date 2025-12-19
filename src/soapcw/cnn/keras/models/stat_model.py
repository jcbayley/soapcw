#!/home/joseph.bayley/.virtualenvs/soap27/bin/python
import keras
import plot_and_sigfits as psf
from keras import backend as K
from keras.constraints import nonneg
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    LeakyReLU,
    MaxPooling2D,
    concatenate,
)
from keras.models import Input, Model, Sequential, load_model
from keras.utils import Sequence


def model():
    inputstat = Input(shape=(1,), name="stat_input")

    #############
    # simple network for statistic
    ###############

    # stat = Dense(16)(inputstat)
    # stat = LeakyReLU(alpha=0.1)(stat)

    stat = Dense(1, name="stat_out", activation="sigmoid")(inputstat)
    # stat = LeakyReLU(alpha=0.1,name = "stat_act")(stat)

    model = Model(inputs=inputstat, output=stat)

    # model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['accuracy'],loss_weights=[1., 0.0,0.0])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model
