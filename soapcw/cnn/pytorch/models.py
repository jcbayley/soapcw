import torch
import torch.nn as nn
import numpy as np

class CNN(nn.Module):

    def __init__(self, input_dim, fc_layers = [], conv_layers = [], stride = 1, device="cpu", dropout=0.1, inchannels = 1, avg_pool_size=None):
        """
        args
        ----------
        input_dim: int
            length of input time series
        latent_dim: int
            number of variables in latent space
        fc_layers: list
            list of the fully connected layers [8, 4, 2] (8 neurons, 4 neurons, 2 neurons)
        conv_layers: list
            list of the convolutional layers [(8, 5, 2, 1), (4, 3, 2, 1)] (8 filters of size 5, 4 filters of size 3)
        """
        super().__init__()
        # define useful variables of network
        self.device = device
        self.input_dim = input_dim
        self.inchannels = inchannels
        
        # convolutional parts
        self.fc_layers = fc_layers
        self.conv_layers = conv_layers
        self.num_conv = len(self.conv_layers)
        self.stride = stride
        self.avg_pool_size = tuple(avg_pool_size)

        self.activation = nn.LeakyReLU()
        self.out_activation = nn.Sigmoid()#nn.Softmax()
        self.drop = nn.Dropout(p=dropout)

        self.small_const = 1e-6
        
        self.conv_network, self.lin_network = self.create_network("o", self.input_dim, fc_layers = self.fc_layers, conv_layers = self.conv_layers)


    def create_network(self, name, input_dim,  append_dim=0, fc_layers=[], conv_layers=[]):
        """ Generate arbritrary network, with convolutional layers or not
        args
        ------
        name: str
            name of network
        input_dim: int
            size of input to network
        append_dim: int (optional)
            number of neurons to append to fully connected layers after convolutional layers
        mean: bool (optional) 
            if True adds extra output layer for means 
        variance : bool (optional)
            if True adds extra layer of outputs for variances
        fc_layers: list
            list of fully connected layers, format: [64,64,...] = [num_neurons_layer1, num_neurons_layer2, .....]
        conv_layers: list
            list of convolutional layers, format:[(8,8,2,1), (8,8,2,1)] = [(num_filt1, conv_size1, max_pool_size1, dilation1), (num_filt2, conv_size2, max_pool_size2, dilation2)] 
        """
        conv_network = nn.Sequential() # initialise networks
        lin_network = nn.Sequential()
        layer_out_sizes = []
        num_conv = len(conv_layers)
        num_fc = len(fc_layers)
        insize = self.input_dim
        inchannels = self.inchannels
        # add convolutional layers
        for i in range(num_conv):
            # set layer parameters
            padding = (int(conv_layers[i][1]/2.),int(conv_layers[i][1]/2.)) # padding half width
            maxpool_size = (conv_layers[i][2],conv_layers[i][2]) #define max pooling for this layer
            filt_size = (conv_layers[i][1],conv_layers[i][1])
            stride = (self.stride, self.stride)
            dilation = (conv_layers[i][3],conv_layers[i][3])
            maxpool = nn.MaxPool2d(maxpool_size) #define max pooling for this layer
            # add convolutional/activation/maxpooling layers
            #print(inchannels, conv_layers[i][0], filt_size, stride, padding, dilation)
            conv_network.add_module("r_conv{}".format(i),module = nn.Conv2d(inchannels, conv_layers[i][0], filt_size, stride = stride, padding=padding, dilation = dilation))
            conv_network.add_module("act_r_conv{}".format(i), module = self.activation)
            conv_network.add_module("pool_r_conv{}".format(i),module = maxpool)
            # define the output size of the layer
            outsize = self.conv_out_size(insize[0], insize[1], padding, dilation, filt_size, stride, maxpool_size) # output of one filter
            layer_out_sizes.append((conv_layers[i][0],outsize[0],outsize[1]))
            insize = outsize
            inchannels = conv_layers[i][0]

        if self.avg_pool_size is not None and num_conv > 0:
            conv_network.add_module("avgpool", nn.AdaptiveAvgPool2d(self.avg_pool_size))
            lin_input_size = np.prod(self.avg_pool_size)*layer_out_sizes[-1][0]
        else:
            # define the input size to fully connected layer
            lin_input_size = np.prod(layer_out_sizes[-1]) if num_conv > 0 else np.prod(self.input_dim)

        if append_dim:
            lin_input_size += append_dim
        layer_size = int(lin_input_size)
        # hidden layers
        for i in range(num_fc):
            lin_network.add_module("r_lin{}".format(i),module=nn.Linear(layer_size, fc_layers[i]))
            if i == num_fc - 1:
                pass
                #lin_network.add_module("act_r_lin{}".format(i),module=self.out_activation)
            else:
                lin_network.add_module("r_drop{}".format(i),module=self.drop)
                lin_network.add_module("act_r_lin{}".format(i),module=self.activation)
            layer_size = fc_layers[i]

        return conv_network, lin_network


    def conv_out_size(self, height_in, width_in, padding, dilation, kernel, stride, maxpool_size):
        height_out = int(((height_in + 2*padding[0] - dilation[0]*(kernel[0]-1)-1)/stride[0] + 1)/maxpool_size[0])
        width_out = int(((width_in + 2*padding[1] - dilation[1]*(kernel[1]-1)-1)/stride[1] + 1)/maxpool_size[1])
        return height_out, width_out
        
    def forward(self, y):
        """forward pass for training"""
        #y = y, (-1, self.inchannels, self.input_dim[0], self.input_dim[1])) if self.num_conv > 0 else y
        conv = self.conv_network(y) if self.num_conv > 0 else y
        lin_in = torch.flatten(conv,start_dim=1) if self.num_conv > 0 else y   # flatten convolutional layers
        out = self.lin_network(lin_in) # run fully connected network
        return out
    
    def test(model, y):
        """generating samples when testing the network """
        num_data = y.size(0)                                                                                                                                                     
        x_out = []
        # encode the data into latent space with r1(z,y)          
        for i in range(num_data):
            x_out.append(model.forward(y[i]).cpu().numpy())
        return np.array(x_out)






class VitmapSpectStatCNN(nn.Module):

    def __init__(self, input_dim, output_dim, fc_layers = [], conv_layers = [], stride = 1, device="cpu", dropout=0.0, inchannels = 1, stat_network = None, vitmap_network = None, spect_network = None, train_loaded_model = False):
        """
        args
        ----------
        input_dim: int
            length of input time series
        latent_dim: int
            number of variables in latent space
        output_dim: int
            number of output neurons
        fc_layers: list
            list of the fully connected layers [8, 4, 2] (8 neurons, 4 neurons, 2 neurons)
        conv_layers: list
            list of the convolutional layers [(8, 5, 2, 1), (4, 3, 2, 1)] (8 filters of size 5, 4 filters of size 3)
        """
        super().__init__()
        # define useful variables of network
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inchannels = inchannels

        # Define networks to load in
        self.stat_network = stat_network
        self.vitmap_network = vitmap_network
        self.spect_network = spect_network
        
        # convolutional parts
        self.fc_layers = fc_layers
        self.conv_layers = conv_layers
        self.num_conv = len(self.conv_layers)
        self.stride = stride

        self.activation = nn.LeakyReLU()
        self.out_activation = nn.Sigmoid()#nn.Softmax()
        self.drop = nn.Dropout(p=dropout)

        self.small_const = 1e-6

        # load or generate the spectrogram network
        if self.spect_network is not None:
            self.spect_network = torch.load(self.spect_network, map_location=self.device)
        else:
            self.spect_network = CNN(self.input_dim, self.output_dim, fc_layers = self.fc_layers, conv_layers =self.conv_layers, stride = self.stride, dropout = self.dropout, inchannels = 2)

        #load or generate the vitmap network
        if self.vitmap_network is not None:
            self.vitmap_network = torch.load(self.vitmap_network, map_location=self.device)
        else:
            self.vitmap_network = CNN(self.input_dim, self.output_dim, fc_layers = self.fc_layers, conv_layers =self.conv_layers, stride = self.stride, dropout = self.dropout, inchannels = 1)

        # load or generate the line aware stat network
        if self.stat_network is not None:
            self.stat_network = torch.load(self.stat_network, map_location = self.device)
        else:
            self.stat_network = nn.Sequential()
            self.stat_network.add_module("lin1",module=nn.Linear(1, 1))
            self.stat_network.add_module("act_lin1",module=self.out_activation)

        self.spect_network.lin_network = nn.Sequential(*list(self.spect_network.lin_network.children())[:-1])
        self.vitmap_network.lin_network = nn.Sequential(*list(self.vitmap_network.lin_network.children())[:-1])
        
        for net in [self.spect_network.lin_network, self.spect_network.conv_network, self.vitmap_network.lin_network, self.vitmap_network.conv_network]:
            for param in list(net.parameters()):
                param.requires_grad = train_loaded_model

        self.final_classifier = nn.Sequential()
        # we remove last layer from spect and vitmap networks so their outputs are of size fc_layers[-1]
        #we then add this to the output of stat network
        self.final_classifier.add_module("linout",module=nn.Linear(2*self.fc_layers[-1] + 1, self.output_dim))
        self.final_classifier.add_module("act_linout",module=self.out_activation)

        
    def forward(self, y_vitmap, y_spect, y_stat):
        """forward pass for training"""
        # run the vitmap network
        y_vitmap = torch.reshape(y_vitmap, (-1, self.vitmap_network.inchannels, self.input_dim[0], self.input_dim[1])) 
        vitmap_conv = self.vitmap_network.conv_network(y_vitmap)
        vitmap_lin = torch.flatten(vitmap_conv,start_dim=1)
        vitmap_out = self.vitmap_network.lin_network(vitmap_lin) # run fully connected network

        # run the spect network
        y_spect = torch.reshape(y_spect, (-1, self.spect_network.inchannels, self.input_dim[0], self.input_dim[1])) 
        spect_conv = self.spect_network.conv_network(y_spect)
        spect_lin = torch.flatten(spect_conv,start_dim=1)
        spect_out = self.spect_network.lin_network(spect_lin) # run fully connected network

        # run the stat network
        y_stat = torch.reshape(y_stat, (-1, 1)) 
        stat_out = self.stat_network(y_stat) # run fully connected network
        
        # combine all three outputs
        out = self.final_classifier(torch.cat([vitmap_out, spect_out, stat_out], dim = 1))

        return out
    
    def test(model, y_vitmap, y_spect, y_stat):
        """generating samples when testing the network """
        num_data = y_vitmap.size(0)                                                                                                                                                     
        x_out = []
        # encode the data into latent space with r1(z,y)          
        for i in range(num_data):
            x_out.append(model.forward(y_vitmap[i], y_spect[i], y_stat[i]).cpu().numpy())
        return np.array(x_out)




class VitmapStatCNN(nn.Module):

    def __init__(self, input_dim, output_dim, fc_layers = [], conv_layers = [], stride = 1, device="cpu", dropout=0.0, inchannels = 1, stat_network = None, vitmap_network = None, train_loaded_model = False):
        """
        args
        ----------
        input_dim: int
            length of input time series
        latent_dim: int
            number of variables in latent space
        output_dim: int
            number of output neurons
        fc_layers: list
            list of the fully connected layers [8, 4, 2] (8 neurons, 4 neurons, 2 neurons)
        conv_layers: list
            list of the convolutional layers [(8, 5, 2, 1), (4, 3, 2, 1)] (8 filters of size 5, 4 filters of size 3)
        """
        super().__init__()
        # define useful variables of network
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inchannels = inchannels

        # Define networks to load in
        self.stat_network = stat_network
        self.vitmap_network = vitmap_network
        
        # convolutional parts
        self.fc_layers = fc_layers
        self.conv_layers = conv_layers
        self.num_conv = len(self.conv_layers)
        self.stride = stride

        self.activation = nn.LeakyReLU()
        self.out_activation = nn.Sigmoid()#nn.Softmax()
        self.drop = nn.Dropout(p=dropout)

        self.small_const = 1e-6

        #load or generate the vitmap network
        if self.vitmap_network is not None:
            self.vitmap_network = torch.load(self.vitmap_network, map_location=self.device)
        else:
            self.vitmap_network = CNN(self.input_dim, self.output_dim, fc_layers = self.fc_layers, conv_layers =self.conv_layers, stride = self.stride, dropout = self.dropout, inchannels = 1)

        # load or generate the line aware stat network
        if self.stat_network is not None:
            self.stat_network = torch.load(self.stat_network, map_location = self.device)
        else:
            self.stat_network = nn.Sequential()
            self.stat_network.add_module("lin1",module=nn.Linear(1, 1))
            self.stat_network.add_module("act_lin1",module=self.out_activation)

        # remove last layer of vitmap network
        self.vitmap_network.lin_network = nn.Sequential(*list(self.vitmap_network.lin_network.children())[:-1])
        
        for net in [self.vitmap_network.lin_network, self.vitmap_network.conv_network]:
            for param in list(net.parameters()):
                param.requires_grad = train_loaded_model

        self.final_classifier = nn.Sequential()
        # we remove last layer from vitmap networks so their outputs are of size fc_layers[-1]
        #we then add this to the output of stat network
        self.final_classifier.add_module("linout",module=nn.Linear(self.fc_layers[-1] + 1, self.output_dim))
        self.final_classifier.add_module("act_linout",module=self.out_activation)

        
    def forward(self, y_vitmap, y_stat):
        """forward pass for training"""
        # run the vitmap network
        y_vitmap = torch.reshape(y_vitmap, (-1, self.vitmap_network.inchannels, self.input_dim[0], self.input_dim[1])) 
        vitmap_conv = self.vitmap_network.conv_network(y_vitmap)
        vitmap_lin = torch.flatten(vitmap_conv,start_dim=1)
        vitmap_out = self.vitmap_network.lin_network(vitmap_lin) # run fully connected network

        # run the stat network
        y_stat = torch.reshape(y_stat, (-1, 1)) 
        stat_out = self.stat_network(y_stat) # run fully connected network
        
        # combine all three outputs
        out = self.final_classifier(torch.cat([vitmap_out,  stat_out], dim = 1))

        return out
    
    def test(model, y_vitmap, y_stat):
        """generating samples when testing the network """
        num_data = y_vitmap.size(0)                                                                                                                                                     
        x_out = []
        # encode the data into latent space with r1(z,y)          
        for i in range(num_data):
            x_out.append(model.forward(y_vitmap[i], y_stat[i]).cpu().numpy())
        return np.array(x_out)

