import torch
import torch.nn as nn
import torchsummary
import os
import h5py
import numpy as np
from soapcw.cnn.pytorch import models
import argparse
import time
import matplotlib.pyplot as plt
import pandas as pd
import logging

class LoadDataOld(torch.utils.data.Dataset):

    def __init__(self, noise_load_directory, signal_load_directory, load_types = ["stats", "vit_imgs", "H_imgs", "L_imgs"], 
                shuffle=True, nfile_load="all", snr_min=None, snr_max=None, return_parameters=False, n_load_files=1, batch_size=128):
        self.load_types = load_types
        self.noise_load_directory = noise_load_directory
        self.signal_load_directory = signal_load_directory
        self.shuffle = shuffle
        self.nfile_load = nfile_load
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.return_parameters = return_parameters
        self.n_load_files = n_load_files
        self.batch_size = 128
        self.all_data = []
        self.all_truths = []
        self.n_data_in_load = 0
        self.total_n_data = 0
        


        if "H_imgs" in self.load_types and "L_imgs" in self.load_types and "vit_imgs" in self.load_types:
            self.n_load_types = len(self.load_types) - 2
        elif "H_imgs" in self.load_types and "L_imgs" in self.load_types and "vit_imgs" not in self.load_types:
            self.n_load_types = len(self.load_types) - 1
        else:
            self.n_load_types = len(self.load_types)

        self.get_filenames()

        if self.n_load_files == "all":
            self.start_file_ind = -len(self.noise_filenames)
        else:
            self.start_file_ind = -self.n_load_files

        self.load_files_to_data()
        if self.n_load_files != "all":
            self.reset_file_indices()

        self.n_noise = int(self.noise_filenames[0].split("_")[-1].split(".")[0])
        self.n_signal = int(self.signal_filenames[0].split("_")[-1].split(".")[0])

    def old___len__(self,):
        return min(len(self.noise_filenames), len(self.signal_filenames))

    def __len__(self,):
        return (2*len(self.noise_filenames)*self.n_data_in_load)//self.batch_size

    def get_image_size(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        with h5py.File(self.noise_filenames[0], "r") as f:
            img = f["H_imgs"]
            img_shape = np.shape(img)
        return img_shape

    def old___getitem__(self, idx):
        """_summary_

        Args:
            idx (int): index

        Returns:
            _type_: data, truths arrays
        """
        noise_data, noise_pars, pname = self.load_file(self.noise_filenames[idx], noise_only = True)
        signal_data, signal_pars, pname = self.load_file(self.signal_filenames[idx]) 
        #print(self.n_load_types)
        pars = list(noise_pars) + list(signal_pars)

        tot_data = [torch.cat([torch.Tensor(noise_data[i]).squeeze(), torch.Tensor(signal_data[i]).squeeze()], dim=0) for i in range(self.n_load_types)]

        truths = torch.cat([torch.zeros(len(noise_data[0])), torch.ones(len(signal_data[0]))])

        if self.shuffle:
            shuffle_inds = np.arange(len(truths))
            np.random.shuffle(shuffle_inds)
            truths = truths[shuffle_inds]
            tot_data = [tot_data[i][shuffle_inds] for i in range(len(tot_data))]
            #pars = np.array(pars)[shuffle_inds]


        truths = torch.nn.functional.one_hot(torch.Tensor(truths).to(torch.int32).long(), 2)

        if self.return_parameters:
            return tot_data, truths, pars, pname
        else:
            return tot_data, truths 

    def __getitem__(self, idx):

        
        data_index = idx*self.batch_size 
        # initialise the data (if file ind has been reset load the files in again)
        if self.start_file_ind < 0 and self.n_load_files != "all":
            self.load_files_to_data()

        # set the index of this sub loaded data
        if self.start_file_ind == 0:
            sub_index = data_index
        else:
            sub_index = data_index % (self.total_n_data - self.n_data_in_load)

        # if all data has been seen loadin the next batch of files
        if sub_index + self.batch_size > self.n_data_in_load and self.n_load_files != "all":
            self.load_files_to_data()
        
        # get this batch of data from the loaded files
        all_data, all_truths = [self.all_data[i][sub_index:sub_index+self.batch_size] for i in range(self.n_load_types)], self.all_truths[sub_index:sub_index+self.batch_size]


        if self.shuffle:
            shuffle_inds = np.arange(len(all_truths))
            np.random.shuffle(shuffle_inds)
            all_truths = all_truths[shuffle_inds]
            all_data = [all_data[i][shuffle_inds] for i in range(len(all_data))]
            #pars = np.array(pars)[shuffle_inds]

        return all_data, all_truths

    def reset_file_indices(self,):
        self.start_file_ind = -self.n_load_files
        self.total_n_data = 0
        self.n_data_in_load = 0
        self.shuffle_filenames()

    def load_files_to_data(self):


        if self.n_load_files == "all":
            n_load_files = len(self.noise_filenames)
        else:
            n_load_files = self.n_load_files

        if self.start_file_ind > len(self.noise_filenames):
            self.reset_file_indices()

        self.start_file_ind += n_load_files
    
        all_data = []
        all_truths = []
        all_pars = []
        for i in range(n_load_files):
            noise_data, noise_pars, pname = self.load_file(self.noise_filenames[i + self.start_file_ind], noise_only = True)
            signal_data, signal_pars, pname = self.load_file(self.signal_filenames[i + self.start_file_ind]) 
            pars = list(noise_pars) + list(signal_pars)

            tot_data = [torch.cat([torch.Tensor(noise_data[i]).squeeze(), torch.Tensor(signal_data[i]).squeeze()], dim=0) for i in range(self.n_load_types)]

            truths = torch.cat([torch.zeros(len(noise_data[0])), torch.ones(len(signal_data[0]))])
            truths = torch.nn.functional.one_hot(torch.Tensor(truths).to(torch.int32).long(), 2)

            all_data.append(tot_data)
            all_truths.append(truths)
            all_pars.append(pars)
            del tot_data, truths, pars
        
        self.all_data = [torch.cat([all_data[i][j] for i in range(n_load_files)], dim=0) for j in range(self.n_load_types)]
        self.all_truths = torch.cat(all_truths, dim=0)
        self.n_data_in_load = self.all_truths.shape[0]
        self.total_n_data += self.n_data_in_load

        print("shapes:", len(self.all_data), self.all_data[0].shape, self.all_truths.shape)

        
        
        
        #return all_data, all_truths, all_pars

    def load_file(self, fname, noise_only = False):
        """loads in one hdf5 containing data 

        Args:
            fname (string): filename

        Returns:
            _type_: data and parameters associated with files
        """
        with h5py.File(fname, "r") as f:
            output_data = []
            imgdone = False
            # enforce snr limits
            pars = np.array(f["pars"])
            parnames = [pn.decode() for pn in list(f["parnames"])]

            for data_type in self.load_types:
                if data_type in ["H_imgs", "L_imgs", "vit_imgs"]:
                    if imgdone:
                        continue
                    elif "H_imgs" in self.load_types and "L_imgs" in self.load_types and "vit_imgs" in self.load_types:
                        output_data.append(np.transpose(np.concatenate([np.expand_dims(f["H_imgs"], -1), np.expand_dims(f["L_imgs"], -1), np.expand_dims(f["vit_imgs"], -1)], axis=-1), (0,3,2,1)))
                        imgdone = True
                    elif "H_imgs" in self.load_types and "L_imgs" in self.load_types and "vit_imgs" not in self.load_types:
                        output_data.append(np.transpose(np.concatenate([np.expand_dims(f["H_imgs"], -1), np.expand_dims(f["L_imgs"], -1)], axis=-1), (0,3,2,1)))
                        imgdone = True
                else:
                    output_data.append(np.array(f[data_type]))

        if self.snr_min is not None and self.snr_max is not None and noise_only is False:
            snrs = pars[:, parnames.index("snr")]
            output_data = [output_data[i][np.logical_and(snrs >= self.snr_min, snrs <= self.snr_max)] for i in range(len(output_data))]
            pars = pars[np.logical_and(snrs >= self.snr_min, snrs <= self.snr_max)]

        return output_data, pars, parnames

    def shuffle_filenames(self):
        np.random.shuffle(self.noise_filenames)
        np.random.shuffle(self.signal_filenames)

    def get_filenames(self):

        self.noise_filenames = [os.path.join(self.noise_load_directory, fname) for fname in os.listdir(self.noise_load_directory)][:4]
        self.signal_filenames = [os.path.join(self.signal_load_directory, fname) for fname in os.listdir(self.signal_load_directory)][:4]

        if self.shuffle:
            self.shuffle_filenames()

        if self.nfile_load != "all":
            self.noise_filenames = self.noise_filenames[:self.nfile_load]
            self.signal_filenames = self.signal_filenames[:self.nfile_load]


class LoadData(torch.utils.data.Dataset):

    def __init__(self, noise_load_directory, signal_load_directory, load_types = ["stats", "vit_imgs", "H_imgs", "L_imgs"], 
                shuffle=True, nfile_load="all", snr_min=None, snr_max=None, return_parameters=False, n_load_files=1, 
                batch_size=128, sort_filenames=False, hwinj_file=None):
        self.load_types = load_types
        self.noise_load_directory = noise_load_directory
        self.signal_load_directory = signal_load_directory
        self.shuffle = shuffle
        self.nfile_load = nfile_load
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.return_parameters = return_parameters
        self.n_load_files = n_load_files
        self.batch_size = 128
        self.all_data = []
        self.all_truths = []
        self.n_data_in_load = 0
        self.total_n_data = 0
        self.parkeys = ['fmin', 'fmax', 'width', 'av_sh', 'tref','snr', 'h0', 'depth', 'f', 'fd', 'alpha', 'sindelta', 'phi0', 'psi', 'cosi']
        self.sort_filenames = sort_filenames

        if hwinj_file is not None:
            self.hardware_injections = pd.read_html(hwinj_file, header=0)[0]
        else:
            self.hardware_injections = None

        if self.sort_filenames and self.shuffle:
            raise Exception("Cannot sort and shuffle filenames, please select one or the other")
        


        if "H_imgs" in self.load_types and "L_imgs" in self.load_types and "vit_imgs" in self.load_types:
            self.n_load_types = len(self.load_types) - 2
        elif "H_imgs" in self.load_types and "L_imgs" in self.load_types and "vit_imgs" not in self.load_types:
            self.n_load_types = len(self.load_types) - 1
        else:
            self.n_load_types = len(self.load_types)

        self.get_filenames()
        self.load_files_to_data()

        



    def __len__(self,):
        return self.total_n_data

    def get_image_size(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        with h5py.File(self.noise_filenames[0], "r") as f:
            img = f["H_imgs"]
            img_shape = np.shape(img)
        return img_shape


    def __getitem__(self, idx):

        #all_data, all_truths = [self.all_data[i][data_index:data_index+self.batch_size] for i in range(self.n_load_types)], self.all_truths[data_index:data_index+self.batch_size]
        all_data, all_truths = [self.all_data[i][idx] for i in range(self.n_load_types)], self.all_truths[idx]
        all_pars = self.all_pars[idx]


        #if self.shuffle:
        #    shuffle_inds = np.arange(len(all_truths))
        #    np.random.shuffle(shuffle_inds)
        #    all_truths = all_truths[shuffle_inds]
        #    all_data = [all_data[i][shuffle_inds] for i in range(len(all_data))]
            #pars = np.array(pars)[shuffle_inds]

        return all_data, all_truths, all_pars


    def load_files_to_data(self):


    
        all_data = []
        all_truths = []
        all_pars = []
        for i in range(len(self.noise_filenames)):
            print("Loading: ", self.noise_filenames[i], self.signal_filenames[i])
            noise_data, noise_pars = self.load_file(self.noise_filenames[i], noise_only = True)
            signal_data, signal_pars = self.load_file(self.signal_filenames[i]) 
            pars = torch.cat([torch.Tensor(noise_pars), torch.Tensor(signal_pars)], dim=0)

            tot_data = [torch.cat([torch.Tensor(noise_data[i]).squeeze(), torch.Tensor(signal_data[i]).squeeze()], dim=0) for i in range(self.n_load_types)]

            truths = torch.cat([torch.zeros(len(noise_data[0])), torch.ones(len(signal_data[0]))])
            truths = torch.nn.functional.one_hot(torch.Tensor(truths).to(torch.int32).long(), 2)

            all_data.append(tot_data)
            all_truths.append(truths)
            all_pars.append(pars)
            del tot_data, truths, pars
        
        self.all_data = [torch.cat([all_data[i][j] for i in range(len(self.noise_filenames))], dim=0) for j in range(self.n_load_types)]
        self.all_truths = torch.cat(all_truths, dim=0)
        self.all_pars = torch.cat(all_pars, dim=0)
        self.n_data_in_load = self.all_truths.shape[0]
        self.total_n_data = self.n_data_in_load

        print("shapes:", len(self.all_data), self.all_data[0].shape, self.all_truths.shape)

        #return all_data, all_truths, all_pars

    def load_file(self, fname, noise_only = False):
        """loads in one hdf5 containing data 

        Args:
            fname (string): filename

        Returns:
            _type_: data and parameters associated with files
        """
        with h5py.File(fname, "r") as f:
            output_data = []
            imgdone = False
            # enforce snr limits
            #pars = np.array(f["pars"])
            parnames = [pn.decode() for pn in list(f["parnames"])]
            pars = np.array([np.array(f["pars"])[:, parnames.index(pk)] for pk in self.parkeys]).T

            for data_type in self.load_types:
                if data_type in ["H_imgs", "L_imgs", "vit_imgs"]:
                    if imgdone:
                        continue
                    elif "H_imgs" in self.load_types and "L_imgs" in self.load_types and "vit_imgs" in self.load_types:
                        output_data.append(np.transpose(np.concatenate([np.expand_dims(f["H_imgs"], -1), np.expand_dims(f["L_imgs"], -1), np.expand_dims(f["vit_imgs"], -1)], axis=-1), (0,3,2,1)))
                        imgdone = True
                    elif "H_imgs" in self.load_types and "L_imgs" in self.load_types and "vit_imgs" not in self.load_types:
                        output_data.append(np.transpose(np.concatenate([np.expand_dims(f["H_imgs"], -1), np.expand_dims(f["L_imgs"], -1)], axis=-1), (0,3,2,1)))
                        imgdone = True
                else:
                    output_data.append(np.array(f[data_type]))

        if self.snr_min is not None and self.snr_max is not None and noise_only is False:
            snrs = pars[:, parnames.index("snr")]
            output_data = [output_data[i][np.logical_and(snrs >= self.snr_min, snrs <= self.snr_max)] for i in range(len(output_data))]
            pars = pars[np.logical_and(snrs >= self.snr_min, snrs <= self.snr_max)]


        if self.hardware_injections is not None:
            for _, hinj in self.hardware_injections.iterrows():
                #logging.info(hinj["f0 (epoch start)"])
                #logging.info(pars[:, parnames.index("fmin")])
                inband = np.logical_and(pars[:, parnames.index("fmin")] <= hinj["f0 (epoch start)"], pars[:, parnames.index("fmax")] >= hinj["f0 (epoch start)"])
                pars = pars[~inband]
                output_data = [output_data[i][~inband] for i in range(len(output_data))]


        return output_data, pars

    def shuffle_filenames(self):
        np.random.shuffle(self.noise_filenames)
        np.random.shuffle(self.signal_filenames)

    def get_filenames(self):

        self.noise_filenames = [os.path.join(self.noise_load_directory, fname) for fname in os.listdir(self.noise_load_directory)]
        self.signal_filenames = [os.path.join(self.signal_load_directory, fname) for fname in os.listdir(self.signal_load_directory)]

        if self.shuffle:
            self.shuffle_filenames()
        elif self.sort_filenames:
            self.noise_filenames = sorted(self.noise_filenames, key = lambda a: float(a.split("_")[-3]))
            self.signal_filenames = sorted(self.signal_filenames, key = lambda a: float(a.split("_")[-3]))
        else:
            pass

        if self.nfile_load != "all":
            self.noise_filenames = self.noise_filenames[:self.nfile_load]
            self.signal_filenames = self.signal_filenames[:self.nfile_load]

def train_batch(model, optimiser, loss_fn, batch_data, batch_labels, model_type="spectrogram", train=True, device="cpu"):
    if train:
        model.train()
        optimiser.zero_grad()
    else:
        model.eval()

    if model_type in ["spectrogram", "vitmapspectrogram"]:
        output = model(torch.Tensor(batch_data[0]).to(device))
        loss = loss_fn(output, batch_labels.to(device).to(torch.float32))
        if train:
            loss.backward()
            optimiser.step()
    
    return loss.item()

def train_multi_batch(model, optimiser, loss_fn, batch_data, batch_labels, model_type="spectrogram", train=True, device="cpu", n_train_multi_size=2):
    """Train each batch, but train the same batch multiple times changing the length of time
          = for use with the Adaptive average pool 

    Args:
        model (_type_): _description_
        optimiser (_type_): _description_
        loss_fn (_type_): _description_
        batch_data (_type_): _description_
        batch_labels (_type_): _description_
        model_type (str, optional): _description_. Defaults to "spectrogram".
        train (bool, optional): _description_. Defaults to True.
        device (str, optional): _description_. Defaults to "cpu".
        n_train_multi_size (int, optional): number of times to split up the time axis, 3 with split into N/1, N/2 and N/3. Defaults to 2.

    Returns:
        _type_: _description_
    """
    if train:
        model.train()
    else:
        model.eval()

    nsize = batch_data[0].size(-1)
    
    total_loss = []

    for i in range(n_train_multi_size):
        seglen = int(nsize/(i+1))
        # set the start indices for the subdata
        # repeat N start indices where N in the number of times to break up data
        start_inds = np.random.uniform(0, (nsize - seglen - 1), size=i+1).astype(int)

        if train:
            optimiser.zero_grad()

        # loop over the subdata start inds and average the loss
        temp_loss = 0
        for sind in start_inds:
            if model_type in ["spectrogram", "vitmapspectrogram"]:
                output = model(torch.Tensor(batch_data[0][:,:,:,sind: sind + seglen]).to(device))
                tloss = loss_fn(output, batch_labels.to(device).to(torch.float32))
                temp_loss += tloss
        
        temp_loss = temp_loss/len(start_inds)
        total_loss.append(temp_loss.item())

        # take the backward pass after averaging loss
        if train:
            temp_loss.backward()
            optimiser.step()
    
    return np.mean(total_loss)

def train_model(
    model_type, 
    save_dir, 
    load_dir, 
    learning_rate, 
    img_dim,
    in_channels,
    conv_layers, 
    fc_layers, 
    avg_pool_size=None, 
    device="cpu", 
    load_model=None, 
    bandtype="even", 
    snrmin=40,
    snrmax=200, 
    fmin=20,
    fmax=500, 
    n_epochs=10, 
    save_interval=100, 
    verbose=False,
    n_train_multi_size=None,
    scheduler_start = 0,
    scheduler_length = 100,
    scheduler_weight = 0.1,
    continue_train=False,
    overwrite=False,
    train_snr_min = None,
    train_snr_max = None,
    n_updates_per_batch=1,
    hwinj_file=None
    ):
    """_summary_

    Args:
        model_type (_type_): _description_
        save_dir (_type_): _description_
        load_dir (_type_): _description_
        learning_rate (_type_): _description_
        img_dim (_type_): _description_
        conv_layers (_type_): _description_
        fc_layers (_type_): _description_
        avg_pool_size (_type_, optional): _description_. Defaults to None.
        device (str, optional): _description_. Defaults to "cpu".
        load_model (_type_, optional): _description_. Defaults to None.
        bandtype (str, optional): _description_. Defaults to "even".
        snrmin (int, optional): _description_. Defaults to 40.
        snrmax (int, optional): _description_. Defaults to 200.
        fmin (int, optional): _description_. Defaults to 20.
        fmax (int, optional): _description_. Defaults to 500.
        n_epochs (int, optional): _description_. Defaults to 10.
        save_interval (int, optional): _description_. Defaults to 100.

    Raises:
        Exception: _description_
    """

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    other_bandtype = "odd" if bandtype == "even" else "even"
    train_noise_dir = os.path.join(load_dir, "train", bandtype, f"band_{fmin:.1f}_{fmax:.1f}", "snr_0.0_0.0")
    train_signal_dir = os.path.join(load_dir, "train", bandtype, f"band_{fmin:.1f}_{fmax:.1f}", f"snr_{float(snrmin):.1f}_{float(snrmax):.1f}")

    val_noise_dir = os.path.join(load_dir, "train", other_bandtype, f"band_{fmin:.1f}_{fmax:.1f}", "snr_0.0_0.0")
    val_signal_dir = os.path.join(load_dir, "train", other_bandtype, f"band_{fmin:.1f}_{fmax:.1f}", f"snr_{float(snrmin):.1f}_{float(snrmax):.1f}")

    model_fname = os.path.join(save_dir, f"model_{model_type}_for_{other_bandtype}_F{fmin}_{fmax}.pt")

    if os.path.isfile(model_fname) and not overwrite and not continue_train:
        raise Exception(f"Model file {model_fname} already exists, set overwrite=True to overwrite or continue_train=True to continue training")

    if verbose:
        print("Loading data from: ", train_noise_dir, train_signal_dir)
        print("Saving model to: ", model_fname)

    if model_type == "spectrogram":
        load_types = ["H_imgs", "L_imgs"]
        #inchannels = 2
    elif model_type == "vitmap":
        load_types = ["vit_imgs"]
        #inchannels = 1
    elif model_type == "vitmapspectrogram":
        load_types = ["vit_imgs", "H_imgs", "L_imgs"]
        #inchannels = 3
    elif model_type == "vitmapspectrogramstat":
        load_types = ["vit_imgs", "H_imgs", "L_imgs", "stat"]
    else:
        raise Exception(f"Load type {model_type} not defined select from [spectrogram, vitmap, vit_imgs, vitmapspectrogram, vitmapspectstatgram]")

    if train_snr_min is None:
        train_snr_min = snrmin
    if train_snr_max is None:
        train_snr_max = snrmax

    train_dataset_1 = LoadData(
        train_noise_dir, 
        train_signal_dir, 
        load_types=load_types,
        snr_min = train_snr_min,
        snr_max = train_snr_max,
        hwinj_file=hwinj_file)



    validation_dataset_1 = LoadData(
        val_noise_dir, 
        val_signal_dir, 
        load_types=load_types, 
        nfile_load=2,
        snr_min = train_snr_min,
        snr_max = train_snr_max,
        hwinj_file=hwinj_file)

    train_dataset = torch.utils.data.DataLoader(train_dataset_1, batch_size=128, shuffle=True)
    validation_dataset = torch.utils.data.DataLoader(validation_dataset_1, batch_size=64, shuffle=False)

    print("Training data len: ", len(train_dataset))
    print("Validation data len: ", len(validation_dataset))
    trd0 = train_dataset_1[0]
    print("datashape: ", [np.shape(trd0[0][i]) for i in range(len(trd0[0]))])
    print(train_noise_dir)
    print("data loaded")

    img_dim = np.array(train_dataset_1.get_image_size()[1:])[::-1]

    if model_type == "spectrogram":
        #inchannels = 2
        model = models.CNN(input_dim=img_dim, fc_layers=fc_layers, conv_layers=conv_layers, inchannels=in_channels, avg_pool_size=avg_pool_size, device=device).to(device)
    elif model_type == "vitmap":
        #inchannels = 1
        model = models.CNN(input_dim=img_dim, fc_layers=fc_layers, conv_layers=conv_layers, inchannels=in_channels, avg_pool_size=avg_pool_size, device=device).to(device)
    elif model_type == "vitmapspectrogram":
        inchannels = 3
        model = models.CNN(input_dim=img_dim, fc_layers=fc_layers, conv_layers=conv_layers, inchannels=in_channels, avg_pool_size=avg_pool_size, device=device).to(device)
    elif model_type == "vitmapspectrogramstat":
        load_types = ["vit_imgs", "H_imgs", "L_imgs", "stat"]
    else:
        raise Exception(f"Load type {model_type} not defined select from [spectrogram, vitmap, vit_imgs, vitmapspectrogram, vitmapspectstatgram]")

    print("model")
    print(torchsummary.summary(model, (in_channels, img_dim[0], img_dim[1])))

    optimiser = torch.optim.AdamW(model.parameters(), lr = learning_rate)
    scheduler_end = scheduler_start + scheduler_length
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, eta_min = scheduler_weight*learning_rate, T_max=scheduler_length)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    if load_model is not None:
        if os.path.isfile(load_model):
            load_model_fname = load_model
        elif os.path.isdir(load_model):
            load_model_fname = os.path.join(load_model, f"model_{model_type}_for_{other_bandtype}_F{fmin}_{fmax}.pt")
        else:
            raise Exception(f"Model path {load_model} not found")
        checkpoint = torch.load(load_model_fname, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if continue_train:
            optimiser.load_state_dict(checkpoint["optimiser_state_dict"])

            with open(os.path.join(save_dir, f"losses_for_{model_type}_{other_bandtype}_F{fmin}_{fmax}.txt"), "r") as f:
                loaded_losses = np.loadtxt(f, skiprows=1)
                all_losses = list(loaded_losses[:,0])
                all_val_losses = list(loaded_losses[:,1])
                start_epoch = len(all_val_losses)
        else:
            all_losses = []
            all_val_losses = []
            start_epoch = 0

    else:
        all_losses = []
        all_val_losses = []
        start_epoch = 0

    print("model loaded")

    if n_train_multi_size not in [None, "none"] and avg_pool_size in [None, "none"]:
        raise Exception("Train multi batch can only be used with adaptive average pooling avg_pool_size")

    all_batch_losses = []
    min_val_loss = np.inf
    print("training....")
    for epoch in range(start_epoch, n_epochs):
        epoch_start = time.time()
        losses = []
        mean_batch_time = [time.time()]
        batch_times = [time.time()]
        batch_number = 0
        loss = 0
        for batch_data, batch_labels, batch_pars in train_dataset:
            bt_start = batch_times[-1]
            if verbose:
                print(f"Batch {batch_number}, mean_batch_time: {np.mean(mean_batch_time[1:])}, loss: {loss}")
            for _ in range(n_updates_per_batch):
                if n_train_multi_size not in [None, "none"]:
                    loss = train_multi_batch(model, optimiser, loss_fn, batch_data, batch_labels, device=device, n_train_multi_size=n_train_multi_size)
                else:
                    loss = train_batch(model, optimiser, loss_fn, batch_data, batch_labels, device=device)    
                losses.append(loss)
                all_batch_losses.append(loss)
            batch_times.append(time.time())
            batch_time = batch_times[-1] - bt_start
            mean_batch_time.append(batch_time)
            batch_number += 1
            #if verbose:
            #    print(f"batch_time: {batch_time}")

            if batch_number % 10 == 0 and verbose:
                fig, ax = plt.subplots()
                ax.plot(all_batch_losses, label="training loss")
                #ax.plot(all_val_losses, label="validation loss")
                ax.set_xlabel("iteration")
                ax.set_ylabel("Loss")
                ax.set_yscale("log")
                ax.legend()
                fig.savefig(os.path.join(save_dir, f"batch_losses_for_{model_type}_{other_bandtype}_F{fmin}_{fmax}.png"))
        
        with torch.no_grad():
            val_losses = []
            for i, (batch_data, batch_labels, batch_pars) in enumerate(validation_dataset):
                if n_train_multi_size not in [None, "none"]:
                    vloss = train_multi_batch(model, optimiser, loss_fn, batch_data, batch_labels, train=False, device=device, n_train_multi_size=n_train_multi_size)
                else:
                    vloss = train_batch(model, optimiser, loss_fn, batch_data, batch_labels, train=False, device=device)    
                val_losses.append(vloss)
                if i > 10:
                    break
        
        if scheduler_end > epoch >= scheduler_start:
            scheduler.step()

        
        mloss = np.mean(losses)
        mvloss = np.mean(val_losses)
        all_losses.append(mloss)
        all_val_losses.append(mvloss)
        print(f"Epoch: {epoch}, Loss: {mloss} val_loss: {mvloss}, epoch_time: {time.time() - epoch_start}")

        # shuffle the filenames for the next epoch
        #train_dataset.shuffle_filenames()

        if epoch % save_interval == 0:
            if mvloss < min_val_loss or epoch in [0,1]:
                if verbose:
                    print(f"saving model to {model_fname}")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimiser_state_dict": optimiser.state_dict(),
                    "input_dim":img_dim,
                    "fc_layers":fc_layers, 
                    "conv_layers":conv_layers, 
                    "inchannels":in_channels, 
                    "avg_pool_size":avg_pool_size,
                }, model_fname)
                min_val_loss = mvloss


            fig, ax = plt.subplots()
            ax.plot(all_losses, label="training loss")
            ax.plot(all_val_losses, label="validation loss")
            ax.set_xlabel("iteration")
            ax.set_ylabel("Loss")
            ax.set_yscale("log")
            ax.legend()
            fig.savefig(os.path.join(save_dir, f"losses_for_{model_type}_{other_bandtype}_F{fmin}_{fmax}.png"))

            with open(os.path.join(save_dir, f"losses_for_{model_type}_{other_bandtype}_F{fmin}_{fmax}.txt"), "w") as f:
                np.savetxt(f, np.array([all_losses, all_val_losses]).T, header="train_loss, val_loss")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', help='display status', action='store_true')
    parser.add_argument("-c", "--config-file", help="config file contatining parameters")
    parser.add_argument("-bt", "--band-type", help="force to train either even or odd band", default=None)
    parser.add_argument("-ct", "--continue-train", help="load_model and continue training", action='store_true')
    parser.add_argument("-ow", "--overwrite", help="overwrite model", action='store_true')
    parser.add_argument("-fmin", "--fmin", help="low frequency", default=20, type=float)
    parser.add_argument("-fmax", "--fmax", help="high frequency", default=500, type=float)
    parser.add_argument("-lm", "--load-model", help="load model from path", default=None)
    parser.add_argument("-lr", "--learning-rate", help="learning rate", default=None, type=float)
    parser.add_argument("-nu", "--n-updates-per-batch", help="number of updates per batch", default=1, type=int)
    device = "cuda:0"
                                                    
    args = parser.parse_args()  

    if args.continue_train and args.overwrite:
        raise Exception("Cannot continue training and overwrite model, please select only one of continue-train or overwrite")

    from soapcw.soap_config_parser import SOAPConfig

    if args.config_file is not None:
        cfg = SOAPConfig(args.config_file)

    if args.band_type is not None:
        bandtypes = [str(args.band_type), ]
    else:
        bandtypes = cfg["cnn_model"]["band_types"]

    if args.continue_train and args.load_model is not None:
        raise Exception("Must provide one of model path directory (--load-model) or continue training from default checkpoint (--continue-train), not both")
    elif args.continue_train and args.load_model is None:
        print("-------------------------")
        print("continuing from previous checkpoint")
        model_load_dir = cfg["output"]["cnn_model_directory"]
    elif not args.continue_train and args.load_model is not None:
        print("---------------------------")
        print("Loading model from: ", args.load_model)
        model_load_dir = args.load_model
    else:
        model_load_dir = None

    if args.learning_rate is not None:
        cfg["cnn_model"]["learning_rate"] = str(args.learning_rate)

    for bandtype in bandtypes:
        train_model(cfg["cnn_model"]["model_type"], 
                    cfg["output"]["cnn_model_directory"], 
                    cfg["output"]["cnn_train_data_save_dir"], 
                    cfg["cnn_model"]["learning_rate"], 
                    cfg["cnn_model"]["img_dim"],
                    cfg["cnn_model"]["n_channels"],
                    cfg["cnn_model"]["conv_layers"], 
                    cfg["cnn_model"]["fc_layers"],
                    avg_pool_size=cfg["cnn_model"]["avg_pool_size"],
                    bandtype = bandtype,
                    n_epochs = cfg["cnn_model"]["n_epochs"],
                    device=device,
                    save_interval=cfg["cnn_model"]["save_interval"],
                    n_train_multi_size=cfg["cnn_model"]["n_train_multi_size"],
                    load_model=model_load_dir,
                    fmin=args.fmin,
                    fmax=args.fmax,
                    snrmin=cfg["cnn_data"]["snrmin"],
                    snrmax=cfg["cnn_data"]["snrmax"],
                    scheduler_start = cfg["cnn_model"]["scheduler_start"],
                    scheduler_length = cfg["cnn_model"]["scheduler_length"],
                    scheduler_weight = cfg["cnn_model"]["scheduler_weight"],
                    continue_train=args.continue_train,
                    overwrite=args.overwrite,
                    train_snr_max=cfg.get("cnn_model", "train_snr_max"),
                    train_snr_min=cfg.get("cnn_model", "train_snr_min"),
                    verbose=args.verbose,
                    n_updates_per_batch=args.n_updates_per_batch,
                    hwinj_file = cfg.get("input", "hardware_injections")
                    )



if __name__ == "__main__":
    main()
