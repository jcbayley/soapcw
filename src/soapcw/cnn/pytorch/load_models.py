import pkg_resources
import torch
import numpy as np
import configparser
import soapcw

def load_model_from_config(config, weights, device="cpu"):
    """
    Load in model weights from config and weights
    args
    ---------
    config: string
        path to the config ini file
    weights: string
        path to for the checkpointed weights of network

    returns
    ---------
    model_soap: torch.Module
        returns a torch model with trained weights
    config: SoapConfigParser
        returns the parsed config file for the model
    """
    config = soapcw.soap_config_parser.SOAPConfig(config)

    pre_model_weights = torch.load(weights, map_location=device)


    model_soap = soapcw.cnn.pytorch.models.CNN(
        input_dim=config["model"]["img_dim"],               # the size of the input image
        fc_layers=config["model"]["fc_layers"],               # the size of the fully connected mlp layers 
        conv_layers=config["model"]["conv_layers"],           # the convolutional layers (num_filters, filter_size, maxpool_size, stride)
        inchannels=config["model"]["n_channels"], 
        avg_pool_size=config["model"]["avg_pool_size"],                            # the size of the average pooling layer    
        device=device).to(device)


    model_soap.load_state_dict(pre_model_weights["model_state_dict"])

    return model_soap, config
    
def load_vitmap_model(fmin, obs_run = "O3", checkpoint_file = None):
    """
    Load the Viterbimap model for testing
    """
    if 40 < fmin < 500:
        brange = "40_500"
        stride = 1
    elif 500 < fmin < 1000:
        brange = "500_1000"
        stride = 2
    elif 1000 < fmin < 1500:
        brange = "1000_1500"
        stride = 3
    elif 1500 < fmin < 2000:
        brange = "1500_2000"
        stride = 4
        
    if np.round(fmin*10).astype(int) % stride*2 == 0:
        oddeven = "even"
    else:
        oddeven = "odd"
    
    if checkpoint_file is None:
        model_state = pkg_resources.resource_stream(__name__, "trained_models/{}/vitmap_F{}_{}.ckpt".format(obs_run,brange, oddeven))
    model_init = pkg_resources.resource_stream(__name__, "trained_models/{}/vitmap.ini".format(obs_run))
    
    config = read_config(model_init)

    # hardcoded input size at the moment
    model = cnn.CNN((156,89), 1, config["lin_layers"], config["conv_layers"] , device ="cpu", num_channels = 0)
    model.load_state_dict(torch.load(model_state))
    
    return model


def load_spect_model(fmin, obs_run = "O3"):

    if 40 < fmin < 500:
        brange = "40_500"
        stride = 1
    elif 500 < fmin < 1000:
        brange = "500_1000"
        stride = 2
    elif 1000 < fmin < 1500:
        brange = "1000_1500"
        stride = 3
    elif 1500 < fmin < 2000:
        brange = "1500_2000"
        stride = 4
        
    if np.round(fmin*10).astype(int) % stride*2 == 0:
        oddeven = "even"
    else:
        oddeven = "odd"

    model_state = pkg_resources.resource_stream(__name__, "trained_models/{}/vitmap_F{}_{}.ckpt".format(obs_run,brange, oddeven))
    model_init = pkg_resources.resource_stream(__name__, "trained_models/{}/vitmap.ini".format(obs_run))
    
    config = read_config(model_init)

    model = cnn.CNN((156,89), 1, config["lin_layers"], config["conv_layers"] , device ="cpu", num_channels = 2)
    model.load_state_dict(torch.load(model_state))
    
    return model


    
def read_config(config_file):
    cp = configparser.ConfigParser()
    cp.read(config_file.name)

    p = {}
    params = {"model": ["conv_num_filt", "conv_filt_size", "conv_max_pool", "lin_num_neuron", "lin_dropout", "learning_rate", "num_channels"],}

    for key,vals in params.items():
        for val in vals:
            try:
                p[val] = cp.get(key,val)
            except:
                p[val] = None
                print("No key: {}, {}".format(key,val))

    p["lin_layers"] = [int(n) for n in p["lin_num_neuron"].split(" ")]
    conv_num_filt = [int(n) for n in p["conv_num_filt"].split(" ")]
    conv_filt_size = [int(n) for n in p["conv_filt_size"].split(" ")]
    conv_max_pool = [int(n) for n in p["conv_max_pool"].split(" ")]
    p["conv_layers"] = [(conv_num_filt[i], conv_filt_size[i], conv_max_pool[i], 1) for i in range(len(conv_num_filt))]

    return p

