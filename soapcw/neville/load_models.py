import pkg_resources
import torch
import numpy as np
import configparser
import soapcw

def load_model(modelfile, device="cpu"):
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

    pre_model = torch.load(modelfile, map_location=device).to(device)
    pre_model.device = device

    return model_soap

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

    model_soap = soapcw.neville.models.CVAE(
        input_dim=config["neville"]["input_dim"],
        par_dim=config["neville"]["num_predict_params"],
        latent_dim=config["neville"]["latent_dim"],
        fc_layers=config["neville"]["fc_layers"],
        conv_layers=config["neville"]["conv_layers"],
        inchannels=config["neville"]["n_channels"],
        fdim=config["neville"]["fdim"],
        dropout=config["neville"]["dropout"],
        dist_type=config["neville"]["dist_type"],
        device=device
        ).to(device)


    model_soap.load_state_dict(pre_model_weights["model_state_dict"])

    return model_soap, config
