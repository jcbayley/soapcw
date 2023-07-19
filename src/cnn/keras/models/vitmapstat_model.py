#!/home/joseph.bayley/.virtualenvs/soap27/bin/python
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
from keras.models import Sequential, load_model, Input, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, concatenate, BatchNormalization, LeakyReLU
from keras.utils import Sequence
from keras.constraints import nonneg
from keras import backend as K
import keras
import argparse
import os
import sys
import matplotlib.pyplot as plt
import pickle
import json
from random import sample
import corner
import emcee
from scipy.optimize import minimize
import plot_and_sigfits as psf

#os.environ['LD_LIBRARY_PATH'] = "/usr/local/cuda-10.0/lib64/"

cuda_dev = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_dev

import tensorflow as tf
from tensorflow.python.client import device_lib

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)
print("----------------------------",device_lib.list_local_devices(),"------------------------------------")

print("---------------------",K.tensorflow_backend._get_available_gpus(),"---------------------")


def model_sig():
    
    inputstat = Input(shape=(1,),name = "stat_input")
    inputim = Input(shape=(psf.image_shape[0], psf.image_shape[1],1),name="image_input")
    
    #############
    # simple network for statistic
    ###############
    
    #stat = Dense(16)(inputstat)
    #stat = LeakyReLU(alpha=0.1)(stat)
    stat = Dense(1,name="stat_out",activation="sigmoid")(inputstat)
    #stat = LeakyReLU(alpha=0.1,name = "stat_act")(stat)
    stat = Model(inputs=inputstat, output=stat)
    
    ############
    # convolutional network for image
    ###############
    
    #img = Conv2D(32,(11,11), name= "img_conv")(inputim)
    #img = LeakyReLU(alpha=0.1,name="img_convact")(img)
    #img = MaxPooling2D(pool_size=(2, 2),name="maxpool")(img)
    
    img = Conv2D(8,(5,5),name="img_conv2")(inputim)
    img = LeakyReLU(alpha=0.1)(img)
    img = MaxPooling2D(pool_size=(4, 4))(img)
    
    img = Conv2D(8,(3,3),name="img_conv3")(img)
    img = LeakyReLU(alpha=0.1)(img)
    img = MaxPooling2D(pool_size=(4, 4))(img)
    
    img = Flatten(name="flatten")(img) 
    
    #img = Dense(256)(img)
    #img = LeakyReLU(alpha=0.1)(img)
    #img = Dropout(0.1)(img)

    #img = Dense(256)(img)
    #img = LeakyReLU(alpha=0.1)(img)
    #img = Dropout(0.1)(img)
    
    img = Dense(8,name="img_dense1")(img)
    img = LeakyReLU(alpha=0.1,name="img_dense1act")(img)
    img = Dropout(0.5)(img)
    
    img = Dense(8,name="img_dense2")(img)
    img = LeakyReLU(alpha=0.1,name="img_dense2act")(img)
    img = Dropout(0.5)(img)
    
    img = Dense(8,name="img_dense3",activation="relu")(img)
    #img = LeakyReLU(alpha=0.1,name="img_dense3act")(img)
    img = Dropout(0.5)(img)
    
    #img = Dense(1,name="img_out",activation="sigmoid")(img)
    #img = LeakyReLU(alpha=0.1,name="img_dense3act")(img)
    
    img = Model(inputs=inputim, output=img)
    
    ################
    # combine the outputs from both models
    ################
    
    combined = concatenate([stat.output, img.output])
    
    z = Dense(4)(combined)
    z = LeakyReLU(alpha=0.1)(z)
    z = Dropout(0.5)(z)
    z = Dense(4)(z)
    z = LeakyReLU(alpha=0.1)(z)
    z = Dropout(0.5)(z)
    z = Dense(1, activation = "sigmoid",name="final_out")(z)
    
    # setup final model which takes in inputs for each model
    
    model = Model(inputs=[stat.input, img.input], output=[z])
    
    #model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['accuracy'],loss_weights=[1., 0.0,0.0])
    adam = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    return model

def model_both_lin():
    
    inputim = Input(shape=(image_shape[0], image_shape[1],1),name="image_input")
    inputstat = Input(shape=(1,),name = "stat_input")
    
    #############
    # simple network for statistic
    ###############
    
    #stat = Dense(16)(inputstat)
    #stat = LeakyReLU(alpha=0.1)(stat)
    stat = Dense(1,name="stat_out",activation="relu")(inputstat)
    #stat = LeakyReLU(alpha=0.1,name = "stat_act")(stat)
    stat = Model(inputs=inputstat, output=stat)
    
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
    
    #img = Dense(256)(img)
    #img = LeakyReLU(alpha=0.1)(img)
    #img = Dropout(0.1)(img)

    #img = Dense(256)(img)
    #img = LeakyReLU(alpha=0.1)(img)
    #img = Dropout(0.1)(img)
    
    img = Dense(16,name="img_dense1")(img)
    img = LeakyReLU(alpha=0.1,name="img_dense1act")(img)
    img = Dropout(0.5)(img)
    
    img = Dense(16,name="img_dense2")(img)
    img = LeakyReLU(alpha=0.1,name="img_dense2act")(img)
    img = Dropout(0.5)(img)
    
    img = Dense(8,name="img_dense3",activation="relu")(img)
    #img = LeakyReLU(alpha=0.1,name="img_dense3act")(img)
    img = Dropout(0.5)(img)
    
    #img = Dense(1,name="img_out",activation="sigmoid")(img)
    #img = LeakyReLU(alpha=0.1,name="img_dense3act")(img)
    
    img = Model(inputs=inputim, output=img)
    
    ################
    # combine the outputs from both models
    ################
    
    combined = concatenate([stat.output, img.output])
    
    z = Dense(8)(combined)
    z = LeakyReLU(alpha=0.1)(z)
    z = Dropout(0.5)(z)
    z = Dense(1, activation = "relu",name="final_out")(z)
    
    # setup final model which takes in inputs for each model
    
    model = Model(inputs=[stat.input, img.input], output=[z])
    
    #model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['accuracy'],loss_weights=[1., 0.0,0.0])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    
    return model

def model_init(vitmap_model,stat_model):
    
    #inputstat = Input(shape=(1,),name = "stat_input")
    #inputim = Input(shape=(psf.image_shape[0], psf.image_shape[1],1),name="image_input")
    #inputHL = Input(shape=(psf.image_shape[0], psf.image_shape[1],2),name="HL_input")
    trainable = False
    
    img = load_model(os.path.join(vitmap_model,"training_model.h5"))
    img.layers.pop()
    for layer in img.layers:
        layer.name += "_vitmap"
        layer.trainable = trainable
    #img.summary()
    
    stat = load_model(os.path.join(stat_model,"training_model.h5"))
    for layer in stat.layers:
        layer.name += "_stat"
        layer.trainable = trainable
     

    #imgHL.summary()
        
    #################
    # combine the outputs from all models
    ################
    
    combined = concatenate([stat.output, img.output])
    
    #z = Dense(4)(combined)
    #z = Dropout(0.5)(z)
    #z = LeakyReLU(alpha=0.1)(z)
    
    #z = Dense(8)(combined)
    #z = Dropout(0.5)(z)
    #z = LeakyReLU(alpha=0.1)(z)
    z = Dense(1, activation = "sigmoid",name="final_out_all")(combined)
    
    # setup final model which takes in inputs for each model
    
    model = Model(inputs=[stat.input, img.input], output=[z])
    
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    return model



def train_cnn(traindir,model_dir,snr,norm_stat=False,norm_sft=False,vitmap_dir=None,stat_dir=None):

    # set directories for statistics and images training data
    
    numdata = 9000
    train = psf.get_train_filenames(traindir,numdata=numdata,lsnr=50,hsnr=150)

    # shuffle the filenames and take out a percentage as validation
    np.random.shuffle(train)
    train_set = train[:int(0.9*2*numdata)]
    validation_set = train[int(0.9*2*numdata):]
    np.random.shuffle(train_set)
    np.random.shuffle(validation_set)

    
    # load the date for validation and training
    training_set = psf.get_data(train_set,dtype="vitmapvit")
    validation_set = psf.get_data(validation_set,dtype="vitmapvit")
    
    if norm_sft:
        print("-------------------------------",np.shape(training_set[0][0]))
        absmaximg = np.abs(training_set[0][1]).max()
        divabsmaximg = 1./absmaximg
        for ind,dat in enumerate(training_set[0][1]):
            training_set[0][1][ind] = training_set[0][1][ind]*divabsmaximg
        for ind,dat in enumerate(validation_set[0][1]):
            validation_set[0][1][ind] = validation_set[0][1][ind]*divabsmaximg
        with open(os.path.join(model_dir,"maximg.txt"),"w+") as f:
            np.savetxt(f,[absmaximg])

    if norm_stat:
        absmaxstat = np.abs(training_set[0][0]).max()
        divabsmaxstat = 1./absmaxstat
        for ind,dat in enumerate(training_set[0][0]):
            training_set[0][0][ind] = training_set[0][0][ind]*divabsmaxstat
        for ind,dat in enumerate(validation_set[0][0]):
            validation_set[0][0][ind] = validation_set[0][0][ind]*divabsmaxstat
            
        with open(os.path.join(model_dir,"maxstat.txt"),"w+") as f:
            np.savetxt(f,[absmaxstat])
            
    loaded = True

    if loaded:
        model = model_both_init(vitmap_dir,stat_dir)
        with open(os.path.join(model_dir,"load_models.txt"),"w") as f:
            f.write("viterbi map model")
            f.write(vitmap_dir)
            f.write("viterbi statistic model")
            f.write(stat_dir)

    # fit model
    history = model.fit(training_set[0],training_set[1], epochs=600, batch_size=500, validation_data = validation_set,shuffle=True)

    # save outputs of model
    model.save(os.path.join(model_dir, "training_model.h5"))

    with open(os.path.join(model_dir,"training_data.json"),"w+") as f:
        json.dump(history.history,f)

def plot_hists(zero_data,snr_data,snr):

    fig, ax = plt.subplots()
    
    ax.hist(zero_data,histtype="stepfilled",alpha=0.6,label = "No inj",bins = 20)
    ax.hist(snr_sata,histtype="stepfilled",alpha=0.6, label = "SNR {}".format(snr),bins = 20)
    ax.set_xlabel("CNN \"Probability\" ",fontsize=20)
    ax.set_ylabel("Count",fontsize=20)
    ax.legend()
    
    return fig

def test_cnn(test_dir, output_dir,snr,norm_stat=False,norm_img=False):

    
    stat_test_sigpath = os.path.join(test_dir,"stats/snr_{}_{}".format(80,150))
    stat_test_noisepath = os.path.join(test_dir,"stats/snr_0.0_0.0")
    
    img_test_sigpath = os.path.join(test_dir,"vit_imgs/snr_{}_{}".format(80,150))
    img_test_noisepath = os.path.join(test_dir,"vit_imgs/snr_0.0_0.0")
    num_imgs = 4000

    test_noise = get_filenames(stat_test_noisepath,img_test_noisepath,num_imgs,1)
    #test_set_noise = mygenerator(test_noise[:,1],test_noise[:,2],test_noise[:,0],900)
    test_set_noise = get_data(test_noise)

    test_sig = get_filenames(stat_test_sigpath,img_test_sigpath,num_imgs,1)
    #test_set_sig = mygenerator(test_sig[:,1],test_sig[:,2],test_sig[:,0],900)
    test_set_sig = get_data(test_sig)
    
    
    if norm_img:
        
        with open(os.path.join(output_dir,"maximg.txt"),"r") as f:
            absmaximg = np.loadtxt(f)
        divabsmaximg = 1./absmaximg
        
        for ind,dat in enumerate(test_set_sig[0][1]):
            test_set_sig[0][1][ind] = test_set_sig[0][1][ind]*divabsmaximg
        for ind,dat in enumerate(test_set_noise[0][1]):
            test_set_noise[0][1][ind] = test_set_noise[0][1][ind]*divabsmaximg

    if norm_stat:

        with open(os.path.join(output_dir,"maxstat.txt"),"r") as f:
            absmaxstat = np.loadtxt(f)
        divabsmaxstat = 1./absmaxstat

        for ind,dat in enumerate(test_set_sig[0][0]):
            test_set_sig[0][0][ind] = test_set_sig[0][0][ind]*divabsmaxstat
        for ind,dat in enumerate(test_set_noise[0][0]):
            test_set_noise[0][0][ind] = test_set_noise[0][0][ind]*divabsmaxstat


    if not os.path.isfile(os.path.join(output_dir,"training_model.h5")):
        with open(os.path.join(output_dir,"training_model.h5"),"w+") as f:
            pass
        
    model = load_model(os.path.join(output_dir,"training_model.h5"))

    #prob_noise = model.predict_generator(test_set_noise,steps=len(test_set_noise))
    #prob_sig = model.predict_generator(test_set_sig,steps=len(test_set_noise))
    
    prob_noise = model.predict(test_set_noise[0])
    prob_sig = model.predict(test_set_sig[0])
    
    print(np.shape(prob_noise))
    
    save_data = []
    for arr in range(len(prob_noise)):
        snrsn = float(test_noise[arr][1].split("_")[-1].strip(".txt"))
        save_data.append((prob_noise[arr][0],0,snrsn))
        snrss = float(test_sig[arr][1].split("_")[-1].strip(".txt"))
        save_data.append((prob_sig[arr][0],1,snrss))
    """
    save_data = []
    for arr in range(len(prob_noise)):
        temp_list = []
        for ind,val in enumerate(prob_noise[arr]):
            snrs = float(test_noise[ind][1].split("_")[-1].strip(".txt"))
            temp_list.append((val[0],0,snrs))
        for ind,val in enumerate(prob_sig[arr]):
            snrs = float(test_sig[ind][1].split("_")[-1].strip(".txt"))
            temp_list.append((val[0],1,snrs))
            
        save_data.append(temp_list)
        del temp_list
    """
    #save_data = [(i,0) for i in prob_noise] + [(i,1) for i in prob_sig]

    outfile = os.path.join(output_dir,"hist_vals.pkl")

    with open(outfile,"w+") as f:
        pickle.dump(save_data,f)
        

if __name__ == "__main__":
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-vr', '--verbose', help='display status', action='store_true')
    parser.add_argument('-t', '--train-dir', help='training data directory', type=str, required=False)
    parser.add_argument('-v', '--val-dir', help='validations data directory', type=str, required=False)
    parser.add_argument('-e', '--test-dir', help='test data directory', type=str, required=False)
    parser.add_argument('-o', '--output-dir', help='test data directory', type=str, required=False)
    parser.add_argument('-T', '--type', help='set training or testing', type=str, required=True)
    parser.add_argument('-s', '--snr', help='signal snrs', type=float, required=True)
    parser.add_argument('-N', '--norm-sft', help='normalise data to maximum', type=str2bool, required=False, default=False)
    parser.add_argument('-vitmap', '--vitmap-dir', help='directory to vitmap model', type=str, required=False)
    parser.add_argument('-stat', '--stat-dir', help='directory to stat model', type=str, required=False)
    parser.add_argument('-tr', '--test-run', help='which data run the test was on', type=str, required=False, default="s6mdc")

    try:
        args = parser.parse_args()
    except:
        sys.exit(1)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        
    if args.type == "efficiency":
        psf.eff_curve(args.output_dir)
        
    elif args.type == "effmdc":
        psf.eff_curve(args.output_dir,mdc=True,run=args.test_run)
        
    elif args.type == "train":
        train_cnn(args.train_dir,args.output_dir,args.snr,norm_sft=args.norm_sft)
        psf.train_plots(args.output_dir)

    elif args.type == "test":
        
        test_cnn(args.test_dir, args.output_dir,args.snr,norm_sft=args.norm_sft)
        psf.test_plots(args.output_dir)
        
    elif args.type == "testeff":
        
        test_cnn(args.test_dir, args.output_dir,args.snr,norm_sft=args.norm_sft)
        psf.test_plots(args.output_dir)
        psf.eff_curve(args.output_dir)

    elif args.type == "testmdc":
        
        psf.test_mdc(args.test_dir, args.output_dir,args.snr,dtype="vitmapvit",run=args.test_run)
        psf.test_plots(args.output_dir,mdc=True,xlim=[-1,250],run=args.test_run)
    
    elif args.type == "testmdceff":
        
        psf.test_mdc(args.test_dir, args.output_dir,args.snr,dtype="vitmapvit",run=args.test_run)
        psf.test_plots(args.output_dir,mdc=True,xlim=[-1,250],run=args.test_run)
        psf.eff_curve(args.output_dir,mdc=True,run=args.test_run)

    elif args.type == "traintest":

        train_cnn(args.train_dir,args.output_dir,args.snr,norm_sft=args.norm_sft)
        psf.train_plots(args.output_dir)
        test_cnn(args.test_dir, args.output_dir,args.snr,norm_sft=args.norm_sft)
        psf.test_plots(args.output_dir,xlim=[-1,250])

    elif args.type == "traintesteff":

        train_cnn(args.train_dir,args.output_dir,args.snr,norm_sft=args.norm_sft)
        psf.train_plots(args.output_dir)
        test_cnn(args.test_dir, args.output_dir,args.snr,norm_sft=args.norm_sft)
        psf.test_plots(args.output_dir,xlim=[-1,250])
        psf.eff_curve(args.output_dir)

    elif args.type == "traintestmdceff":

        train_cnn(args.train_dir,args.output_dir,args.snr,norm_sft=args.norm_sft,vitmap_dir=args.vitmap_dir,stat_dir=args.stat_dir)
        psf.train_plots(args.output_dir)
        psf.test_mdc(args.test_dir, args.output_dir,args.snr,dtype="vitmapvit",run=args.test_run)
        psf.test_plots(args.output_dir,mdc=True,xlim=[-1,250],run=args.test_run)
        psf.eff_curve(args.output_dir,mdc=True,run=args.test_run)
