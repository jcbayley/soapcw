import torch
import torch.nn as nn
import torchsummary
import os
import h5py
import numpy as np
from soapcw.cnn.pytorch import models
import soapcw.cnn.train_model
import argparse
import time
import matplotlib.pyplot as plt

class LoadData(torch.utils.data.Dataset):

    def __init__(self, noise_load_directory, signal_load_directory, load_types = ["stats", "vit_imgs", "H_imgs", "L_imgs"], shuffle=False, nfile_load="all"):
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
        """_summary_

        Args:
            idx (int): index

        Returns:
            _type_: data, truths arrays
        """
        noise_data, pars_noise, pname_noise = self.load_file(self.noise_filenames[idx])
        signal_data, pars_signal, pname_signal = self.load_file(self.signal_filenames[idx]) 
        #print(self.n_load_types)
 
        
        truths_noise = torch.zeros(len(noise_data[0]))
        truths_signal = torch.ones(len(signal_data[0]))

      
        return noise_data, signal_data, pname_noise, pars_noise, pname_signal, pars_signal, truths_noise, truths_signal

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


def run_batch(model, optimiser, loss_fn, batch_data, batch_labels, model_type="spectrogram", train=True, device="cpu"):
    if train:
        model.train()
        optimiser.zero_grad()
    else:
        model.eval()

    if model_type in ["spectrogram", "vitmapspectrogram"]:
        output = model(torch.Tensor(batch_data).to(device))
        loss = loss_fn(output, batch_labels.to(device).to(torch.float32))
        if train:
            loss.backward()
            optimiser.step()
    
    return loss.item(), output


def run_model(
    model_type, 
    save_dir, 
    load_dir, 
    learning_rate, 
    img_dim,
    in_channels,
    conv_layers, 
    fc_layers, 
    avg_pool_size=None, 
    load_model=None,
    device="cpu", 
    bandtype="even", 
    snrmin=40,
    snrmax=200, 
    fmin=20,
    fmax=500, 
    n_test=100,
    verbose=False,):
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

    test_save_dir = os.path.join(save_dir, f"test_model_{model_type}_for_{other_bandtype}_F{fmin}_{fmax}")
    if not os.path.isdir(test_save_dir):
        os.makedirs(test_save_dir)
        
    train_noise_dir = os.path.join(load_dir, "train", other_bandtype, f"band_{fmin:.1f}_{fmax:.1f}", "snr_0.0_0.0")
    train_signal_dir = os.path.join(load_dir, "train", other_bandtype, f"band_{fmin:.1f}_{fmax:.1f}", f"snr_{float(snrmin):.1f}_{float(snrmax):.1f}")

    model_fname = os.path.join(save_dir, f"model_{model_type}_for_{other_bandtype}_F{fmin}_{fmax}.pt")


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
        load_types = ["vit_imgs", "H_imgs", "L_imgs", "stats"]
    else:
        raise Exception(f"Load type {model_type} not defined select from [spectrogram, vitmap, vit_imgs, vitmapspectrogram, vitmapspectstatgram]")


    load_types = ["vit_imgs", "H_imgs", "L_imgs", "stats"]
    print(f"Loading data from {train_noise_dir} and {train_signal_dir}")

    test_data = soapcw.cnn.train_model.LoadData(train_noise_dir, train_signal_dir, load_types=load_types, shuffle=True, nfile_load=n_test)

    test_dataset = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True)


    img_dim = np.array(test_data.get_image_size()[1:])[::-1]

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

    #print("model")
    print(torchsummary.summary(model, (in_channels, img_dim[0], img_dim[1])))

    print("model created")

    optimiser = torch.optim.Adam(model.parameters(), lr = learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    
    load_model_fname = os.path.join(load_model, f"model_{model_type}_for_{other_bandtype}_F{fmin}_{fmax}.pt")
    print(f"Loading model from {load_model_fname}") 
    checkpoint = torch.load(load_model_fname, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimiser.load_state_dict(checkpoint["optimiser_state_dict"])

    """
    with open(os.path.join(save_dir, f"losses_for_{model_type}_{other_bandtype}_F{fmin}_{fmax}.txt"), "r") as f:
        loaded_losses = np.loadtxt(f, skiprows=1)
        all_losses = list(loaded_losses[:,0])
        all_val_losses = list(loaded_losses[:,1])
        start_epoch = len(all_val_losses)

    """
    model.eval()
    with torch.no_grad():
        test_losses = []
        test_pars = {}
        for pname in test_data.parkeys:
                test_pars.setdefault(pname, [])
        test_labels = []
        test_statistic = []
        test_lineaware = []
        pnames = None
        print("Testing model ...")
        for i, (batch_data, batch_labels, batch_pars) in enumerate(test_dataset):
 
            #tot_data = [torch.cat([torch.Tensor(noise_data[j]).squeeze(), torch.Tensor(signal_data[j]).squeeze()], dim=0) for j in range(test_dataset.n_load_types-1)]

            #t_labels = torch.cat([truths_noise, truths_signal])
            #tot_labels = torch.nn.functional.one_hot(torch.Tensor(t_labels).to(torch.int32).long(), 2)
            test_lineaware.extend(batch_data[-1])
            vloss, stat = run_batch(model, optimiser, loss_fn, batch_data[0], batch_labels, train=False, device=device)   
            for i,key in enumerate(test_data.parkeys):
                test_pars[key].extend(batch_pars[:,i])

            test_labels.extend(batch_labels.cpu().numpy())
            test_statistic.extend(stat.cpu().numpy())
            

        test_statistic = np.array(test_statistic)
        test_lineaware = np.array(test_lineaware).flatten()
        test_labels = np.array(test_labels)
        #print("CNN shape", np.shape(test_statistic))
        #print("Lineaware shape", np.shape(test_lineaware))
        #print("Labels shape", np.shape(test_labels))

        line_snr_range, line_sensitivity = make_sensitivity_plot(
            np.array(test_lineaware).flatten(), 
            np.array(test_labels)[:,1], 
            np.array(test_pars["snr"]), 
            false_alarm=0.01, 
            window_width=5)

        cnn_snr_range, cnn_sensitivity = make_sensitivity_plot(
            np.array(test_statistic)[:,1], 
            np.array(test_labels)[:,1], 
            np.array(test_pars["snr"]), 
            false_alarm=0.01, 
            window_width=5)


        with h5py.File(os.path.join(test_save_dir, f"test_results_{model_type}_{other_bandtype}_F{fmin}_{fmax}.h5"), "w") as f:
            f.create_dataset("losses", data=np.array(test_losses))
            f.create_dataset("labels", data=np.array(test_labels))
            f.create_dataset("cnn_statistic", data=np.array(test_statistic))
            f.create_dataset("lineaware_statistic", data=np.array(test_lineaware))
            f.create_dataset("lineaware_sensitivity", data=np.column_stack((line_snr_range, line_sensitivity)))
            f.create_dataset("cnn_sensitivity", data=np.column_stack((cnn_snr_range, cnn_sensitivity)))
            for pname, pval in test_pars.items():
                f.create_dataset(pname, data=pval)

    print("Making plots ...")
    fig, ax = plt.subplots()
    ax.plot(test_pars["snr"], np.array(test_statistic)[:,1], ".")
    ax.set_xlabel("SNR")
    ax.set_ylabel("CNN Statistic")
    fig.savefig(os.path.join(test_save_dir, f"test_snr_vs_cnnstat_{model_type}_{other_bandtype}_F{fmin}_{fmax}.png"))

    fig, ax = plt.subplots()
    cnn_bin_min, cnn_bin_max = np.min(test_statistic[:,1]), np.max(test_statistic[:,1])
    ax.hist(test_statistic[:,1][test_labels[:,1] == 0], bins=100, histtype="step", label="Noise", range=(cnn_bin_min, cnn_bin_max))
    ax.hist(test_statistic[:,1][test_labels[:,1] == 1], bins=100, histtype="step", label="Signal", range=(cnn_bin_min, cnn_bin_max))
    ax.set_xlabel("CNN Statistic")
    ax.set_ylabel("Counts")
    ax.legend()
    fig.savefig(os.path.join(test_save_dir, f"test_cnnstat_histogram_{model_type}_{other_bandtype}_F{fmin}_{fmax}.png"))

    fig, ax = plt.subplots()
    line_aware_bin_min, line_aware_bin_max = np.min(test_lineaware), np.max(test_lineaware)
    ax.hist(test_lineaware[test_labels[:,1] == 0], bins=100, histtype="step", label="Noise",  range=(line_aware_bin_min, line_aware_bin_max))
    ax.hist(test_lineaware[test_labels[:,1] == 1], bins=100, histtype="step", label="Signal", range=(line_aware_bin_min, line_aware_bin_max))
    ax.set_xlabel("Lineaware Statistic")
    ax.set_ylabel("Counts")
    ax.legend()
    fig.savefig(os.path.join(test_save_dir, f"test_lineawarestat_histogram_{model_type}_{other_bandtype}_F{fmin}_{fmax}.png"))
    
    fig, ax = plt.subplots()
    ax.plot(np.array(test_lineaware).flatten(), np.array(test_statistic)[:,1],".")
    ax.set_xlabel("line aware statistic")
    ax.set_ylabel("CNN Statistic")
    fig.savefig(os.path.join(test_save_dir, f"test_lineawarestat_vs_cnnstat_{model_type}_{other_bandtype}_F{fmin}_{fmax}.png"))
         
    fig, ax = plt.subplots()
    ax.plot(test_pars["snr"], np.array(test_lineaware).flatten(),".")
    ax.set_xlabel("SNR")
    ax.set_ylabel("Line aware statistic")
    fig.savefig(os.path.join(test_save_dir, f"test_snr_vs_lineawarestat_{model_type}_{other_bandtype}_F{fmin}_{fmax}.png"))

    fig, ax = plt.subplots()
    ax.plot(test_pars["snr"], (np.array(test_lineaware).flatten() + 650) * np.array(test_statistic)[:,1],".")
    ax.set_xlabel("SNR")
    ax.set_ylabel("CNN Statistic * line aware statistic")
    fig.savefig(os.path.join(test_save_dir, f"test_snr_vs_bothmultiply_{model_type}_{other_bandtype}_F{fmin}_{fmax}.png"))


    odds = np.exp(np.array(test_statistic)[:,1])/np.exp(np.array(test_statistic)[:,0])
    fig, ax = plt.subplots()
    ax.plot(test_pars["snr"], odds,".")
    ax.set_xlabel("SNR")
    ax.set_ylabel("odds")
    ax.set_yscale("log")
    fig.savefig(os.path.join(test_save_dir, f"test_snr_vs_odds_{model_type}_{other_bandtype}_F{fmin}_{fmax}.png"))
    


    # Plot the sensitivity curve
    fig, ax = plt.subplots()
    ax.plot(cnn_snr_range, cnn_sensitivity, label='CNN sensitivity')
    ax.plot(line_snr_range, line_sensitivity, label='Line aware sensitivity')
    ax.set_xlabel('SNR')
    ax.set_ylabel('Sensitivity')
    ax.legend()
    ax.grid()
    fig.savefig(os.path.join(test_save_dir, f"test_sensitivity_{model_type}_{other_bandtype}_F{fmin}_{fmax}.png"))


def make_sensitivity_plot(statistic, labels, snr, false_alarm=0.01, window_width =5):
    # Find the 1% false alarm threshold
    false_alarm_threshold = np.quantile(statistic[labels == 0], 1-false_alarm)

    # Compute the fraction of statistics above the threshold in a sliding window of SNR
    snr_range = np.linspace(snr.min(), snr.max(), 100)
    sensitivity = []

    for snr_value in snr_range:
        window_indices = (snr >= snr_value - window_width) & (snr <= snr_value + window_width)
        window_statistics = statistic[window_indices]
        window_labels = labels[window_indices]

        if len(window_statistics) > 0:
            fraction_above_threshold = np.mean(window_statistics[window_labels == 1] > false_alarm_threshold)
        else:
            fraction_above_threshold = 0

        sensitivity.append(fraction_above_threshold)

    return snr_range, sensitivity



def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', help='display status', action='store_true')
    parser.add_argument("-c", "--config-file", help="config file contatining parameters")
    parser.add_argument("-bt", "--band-type", help="force to train either even or odd band", default=None)
    parser.add_argument("-fmin", "--fmin", help="low frequency", default=20, type=float)
    parser.add_argument("-fmax", "--fmax", help="high frequency", default=500, type=float)
    parser.add_argument("-ntest", "--ntest", help="n test data", default=100, type=int)
    parser.add_argument("-d", "--device", help="device to run on", default="cuda:0")
    #device = "cuda:0"

    try:                                                     
        args = parser.parse_args()  
    except:  
        sys.exit(1)

    from soapcw.soap_config_parser import SOAPConfig

    if args.config_file is not None:
        cfg = SOAPConfig(args.config_file)

    if args.band_type is not None:
        bandtypes = [str(args.band_type), ]
    else:
        bandtypes = cfg["cnn_model"]["band_types"]

    print("-----------------------------------------")
    print([(key, val) for key,val in cfg["output"].items()])

    for bandtype in bandtypes:
        run_model(cfg["cnn_model"]["model_type"], 
                    cfg["output"]["cnn_model_directory"], 
                    cfg["output"]["cnn_train_data_save_dir"], 
                    cfg["cnn_model"]["learning_rate"], 
                    cfg["cnn_model"]["img_dim"],
                    cfg["cnn_model"]["n_channels"],
                    cfg["cnn_model"]["conv_layers"], 
                    cfg["cnn_model"]["fc_layers"],
                    avg_pool_size=cfg["cnn_model"]["avg_pool_size"],
                    bandtype = bandtype,
                    n_test = args.ntest,
                    device=args.device,
                    load_model=cfg["output"]["cnn_model_directory"],
                    fmin=args.fmin,
                    fmax=args.fmax,
                    snrmin=cfg["cnn_data"]["snrmin"],
                    snrmax=cfg["cnn_data"]["snrmax"])



if __name__ == "__main__":
    main()
