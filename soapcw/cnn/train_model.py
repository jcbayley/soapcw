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

class LoadData(torch.utils.data.Dataset):

    def __init__(self, noise_load_directory, signal_load_directory, load_types = ["stats", "vit_imgs", "H_imgs", "L_imgs"], shuffle=True, nfile_load="all"):
        self.load_types = load_types
        self.noise_load_directory = noise_load_directory
        self.signal_load_directory = signal_load_directory
        self.shuffle = shuffle
        self.nfile_load = nfile_load
        self.get_filenames()
        if "H_imgs" in self.load_types and "L_imgs" in self.load_types and "vit_imgs" in self.load_types:
            self.n_load_types = len(self.load_types) - 2
        elif "H_imgs" in self.load_types and "L_imgs" in self.load_types and "vit_imgs" not in self.load_types:
            self.n_load_types = len(self.load_types) - 1
        else:
            self.n_load_types = len(self.load_types)

    def __len__(self,):
        return min(len(self.noise_filenames), len(self.signal_filenames))

    def __getitem__(self, idx):
        """_summary_

        Args:
            idx (int): index

        Returns:
            _type_: data, truths arrays
        """
        noise_data, pars, pname = self.load_file(self.noise_filenames[idx])
        signal_data, pars, pname = self.load_file(self.signal_filenames[idx]) 
        #print(self.n_load_types)

        tot_data = [torch.cat([torch.Tensor(noise_data[i]).squeeze(), torch.Tensor(signal_data[i]).squeeze()], dim=0) for i in range(self.n_load_types)]

        truths = torch.cat([torch.zeros(len(noise_data[0])), torch.ones(len(signal_data[0]))])
        if self.shuffle:
            shuffle_inds = np.arange(len(truths))
            np.random.shuffle(shuffle_inds)
            truths = truths[shuffle_inds]
            tot_data = [tot_data[i][shuffle_inds] for i in range(len(tot_data))]


        truths = torch.nn.functional.one_hot(torch.Tensor(truths).to(torch.int32).long(), 2)

        return tot_data, truths 

    def load_file(self, fname):
        """loads in one hdf5 containing data 

        Args:
            fname (string): filename

        Returns:
            _type_: data and parameters associated with files
        """
        with h5py.File(fname, "r") as f:
            output_data = []
            imgdone = False
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

            pars = np.array(f["pars"])
            parnames = list(f["parnames"])

        return output_data, pars, parnames

    def get_filenames(self):

        self.noise_filenames = [os.path.join(self.noise_load_directory, fname) for fname in os.listdir(self.noise_load_directory)]
        self.signal_filenames = [os.path.join(self.signal_load_directory, fname) for fname in os.listdir(self.signal_load_directory)]

        if self.shuffle:
            np.random.shuffle(self.noise_filenames)
            np.random.shuffle(self.signal_filenames)

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
    verbose=True,
    n_train_multi_size=None):
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
    other_bandtype = "odd" if bandtype == "even" else "even"
    train_noise_dir = os.path.join(load_dir, "train", bandtype, f"band_{fmin}_{fmax}", "snr_0.0_0.0")
    train_signal_dir = os.path.join(load_dir, "train", bandtype, f"band_{fmin}_{fmax}", f"snr_{float(snrmin)}_{float(snrmax)}")

    val_noise_dir = os.path.join(load_dir, "train", other_bandtype, f"band_{fmin}_{fmax}", "snr_0.0_0.0")
    val_signal_dir = os.path.join(load_dir, "train", other_bandtype, f"band_{fmin}_{fmax}", f"snr_{float(snrmin)}_{float(snrmax)}")

    if model_type == "spectrogram":
        load_types = ["H_imgs", "L_imgs"]
        #inchannels = 2
        model = models.CNN(input_dim=img_dim, fc_layers=fc_layers, conv_layers=conv_layers, inchannels=in_channels, avg_pool_size=avg_pool_size, device=device).to(device)
    elif model_type == "vitmap":
        load_types = ["vit_imgs"]
        #inchannels = 1
        model = models.CNN(input_dim=img_dim, fc_layers=fc_layers, conv_layers=conv_layers, inchannels=in_channels, avg_pool_size=avg_pool_size, device=device).to(device)
    elif model_type == "vitmapspectrogram":
        load_types = ["vit_imgs", "H_imgs", "L_imgs"]
        #inchannels = 3
        model = models.CNN(input_dim=img_dim, fc_layers=fc_layers, conv_layers=conv_layers, inchannels=in_channels, avg_pool_size=avg_pool_size, device=device).to(device)
    elif model_type == "vitmapspectrogramstat":
        load_types = ["vit_imgs", "H_imgs", "L_imgs", "stat"]
    else:
        raise Exception(f"Load type {model_type} not defined select from [spectrogram, vitmap, vit_imgs, vitmapspectrogram, vitmapspectstatgram]")

    print("model")
    print(torchsummary.summary(model, (in_channels, img_dim[0], img_dim[1])))

    print("model loaded")

    train_dataset = LoadData(train_noise_dir, train_signal_dir, load_types=load_types)
    validation_dataset = LoadData(val_noise_dir, val_signal_dir, load_types=load_types, nfile_load=2)
    print("Training data len: ", len(train_dataset))
    print("Validation data len: ", len(validation_dataset))
    trd0 = train_dataset[0]
    print("datashape: ", [np.shape(trd0[0][i]) for i in range(len(trd0[0]))])
    print(train_noise_dir)
    print("data loaded")

    optimiser = torch.optim.Adam(model.parameters(), lr = learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    if load_model is not None:
        checkpoint = model.load(load_model)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimiser.load_state_dict(checkpoint["optimiser_state_dict"])

    if n_train_multi_size is not None and avg_pool_size is None:
        raise Exception("Train multi batch can only be used with adaptive average pooling avg_pool_size")

    all_losses = []
    all_val_losses = []
    print("training....")
    for epoch in range(n_epochs):

        epoch_start = time.time()
        losses = []
        mean_batch_time = [time.time()]
        batch_times = [time.time()]
        for batch_data, batch_labels in train_dataset:
            bt_start = batch_times[-1]
            if n_train_multi_size is not None:
                loss = train_multi_batch(model, optimiser, loss_fn, batch_data, batch_labels, device=device, n_train_multi_size=n_train_multi_size)
            else:
                loss = train_batch(model, optimiser, loss_fn, batch_data, batch_labels, device=device)    
            losses.append(loss)
            batch_times.append(time.time())
            batch_time = batch_times[-1] - bt_start
            mean_batch_time.append(batch_time)
            if verbose:
                print(f"batch_time: {batch_time}")

        print(f"mean_batch_time: {np.mean(batch_times)}")
        
        with torch.no_grad():
            val_losses = []
            for i, (batch_data, batch_labels) in enumerate(validation_dataset):
                if n_train_multi_size is not None:
                    vloss = train_multi_batch(model, optimiser, loss_fn, batch_data, batch_labels, train=False, device=device, n_train_multi_size=n_train_multi_size)
                else:
                    vloss = train_batch(model, optimiser, loss_fn, batch_data, batch_labels, train=False, device=device)    
                val_losses.append(vloss)
                if i > 10:
                    break
        
        mloss = np.mean(losses)
        mvloss = np.mean(val_losses)
        all_losses.extend(losses)
        all_val_losses.append(mvloss)
        print(f"Epoch: {epoch}, Loss: {mloss} val_loss: {mvloss}, epoch_time: {time.time() - epoch_start}")
        if epoch % save_interval == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
            }, os.path.join(save_dir, f"model_{model_type}_for_{other_bandtype}.pt"))


            fig, ax = plt.subplots()
            ax.plot(all_losses, label="training loss")
            ax.set_xlabel("iteration")
            ax.set_ylabel("Loss")
            ax.legend()
            fig.savefig(os.path.join(save_dir, f"losses_for_{other_bandtype}.png"))



def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', help='display status', action='store_true')
    parser.add_argument("-c", "--config-file", help="config file contatining parameters")

    device = "cuda:0"

    try:                                                     
        args = parser.parse_args()  
    except:  
        sys.exit(1)

    from soapcw.soap_config_parser import SOAPConfig

    if args.config_file is not None:
        cfg = SOAPConfig(args.config_file)

    for bandtype in cfg["model"]["band_types"]:
        train_model(cfg["model"]["model_type"], 
                    cfg["model"]["save_dir"], 
                    cfg["general"]["save_dir"], 
                    cfg["model"]["learning_rate"], 
                    cfg["model"]["img_dim"],
                    cfg["model"]["n_channels"],
                    cfg["model"]["conv_layers"], 
                    cfg["model"]["fc_layers"],
                    avg_pool_size=cfg["model"]["avg_pool_size"],
                    bandtype = bandtype,
                    n_epochs = cfg["model"]["n_epochs"],
                    device=device,
                    save_interval=cfg["model"]["save_interval"],
                    n_train_multi_size=cfg["model"]["n_train_multi_size"])


if __name__ == "__main__":
    main()
