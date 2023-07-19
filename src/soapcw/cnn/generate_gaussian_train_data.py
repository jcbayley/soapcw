from soapcw.cnn import cnn_data_gen
import numpy as np
from soapcw import cw
import time
import os
import datetime
import pickle 
import sys
import h5py

def loop_band_train_augment(config, path,bandmin,bandmax,band_width,loadpath=None,nonoise=False,snr_min = 50,snr_max=150,nperband=10,test=False, save_options=None):
    """
    loop over bands to create training data
    """

    tsft = 1800.
    for i, (minband, maxband) in enumerate(zip(config["data"]["band_starts"], config["data"]["band_ends"])):
        if minband <= bandmin < maxband:
            stride = config["data"]["strides"][i]
            degfree = int(2 * config["data"]["n_summed_sfts"] * stride)
            width = config["data"]["band_widths"][i]
            nbin = 180#int(width*tsft)
            plotcut = stride * nbin
            brange = f"band_{int(minband):d}_{int(maxband):d}"
            break
        else:
            continue


    start_time = time.time()
    # in 2Hz band, set frequencies and indicies with 0.1 Hz bands not overlapping
    range_bands = np.round(np.arange(bandmin,bandmax-width,width),1)
    range_index = np.arange(0,(bandmax-width-bandmin)*int(tsft),180*stride).astype(int)

    # set which bands are odd and even with and ondex of 1 or 0
    odd_even = np.zeros(len(range_bands))
    odd_even[1::2] = 1
    
    tmin, tmax = None,None#931035615.,931035615.+1800*100

    # set integer Hz as vetos, this will cut out a few bins surrounding
    vetolist = list(np.arange(bandmin,bandmax))

    # find the appropriate sft files
    if config["general"]["load_dir"] is not None:
        hname,lname = cnn_data_gen.find_sft_file(config["general"]["load_dir"],bandmin,bandmax)
        
        # load in the narrowbanded sfts, normalise them, fill in the gaps and save the running median
        sfts_noise = cw.LoadSFT("{};{}".format(hname,lname),norm=True,filled=True,vetolist = vetolist,save_rngmed=True)
        
        noise_est = sfts_noise.H1.get_sh(pos="median"),sfts_noise.L1.get_sh(pos="median")

        # sum the sfts
        sfts_noise.sum_sfts()

        nsft = len(noise_est[0])
        tstart = sfts_noise.H1.epochs[0]

    else:
        noise_est = 1,1 
        nsft = config["data"]["nsfts"]
        tstart = config["data"]["tstart"]

    # make appropriate directoes
    if not os.path.isdir(path):
        os.makedirs(path)
    erange = 15

    snrrange = [snr_min,snr_max]
    
    noise_out_snr = "snr_0.0_0.0"
    signal_out_snr = "snr_{}_{}".format(snrrange[0],snrrange[1])
    #save_options = ["stats","paths", "pars", "powers"]
    #save_options = ["stats","paths", "pars", "powers","vit_imgs","H_imgs","L_imgs"]
    # define all save directories for images, stats, pars, plots etc. out_snr only exists for training data

    even_noise_filenames = {}
    even_signal_filenames = {}
    odd_noise_filenames = {}
    odd_signal_filenames = {}

    even_noise_save_outs = {}
    even_signal_save_outs = {}
    odd_noise_save_outs = {}
    odd_signal_save_outs = {}

    for dirname in save_options:

        even_noise_save_outs[dirname] = []
        even_signal_save_outs[dirname] = []
        odd_noise_save_outs[dirname] = []
        odd_signal_save_outs[dirname] = []

                                                     
    
    # main loop over all sub bands
    for i in range(len(range_bands)):
        
        # set initial band frequency
        k = range_bands[i]

        ms = np.round((range_bands[i] - bandmin)*1800).astype(int)
        #snr_start,snr_end = 80,150
        #ms = int(range_index[i])
        me = int(ms+180*stride)
        
        # put odd bands into one folder and even bands into another folder
        if odd_even[i] == 1:
            split = "odd"
        else:
            split = "even"

        outpath = os.path.join(path,split)                     
        print("inds",ms,me)
        # get the median of the running medians as an estimate of the noise floow in this band.
        if loadpath is not None:
            av_sh = [np.nanmedian(sfts_noise.H1.rng_med[:,ms:me],axis=1),np.nanmedian(sfts_noise.L1.rng_med[:,ms:me],axis=1)]
        else:
            av_sh = [1,1]
                
        # set lower frequency
        #flow = np.round(k + ms / 1800. , 2)
        flow = k

        # run injections function with new data, this saves noise iamge and injects signal and saves signal images
        for i in range(nperband):
            noise_outputs, signal_outputs = run_and_inj(flow,width,outpath,resize_image=config["data"]["resize_image"],av_sh=av_sh,snrrange=snrrange,test=test, save_options=save_options, nsft = nsft, tstart = tstart, stride = stride, degfree = degfree, brange = brange)
            for key in save_options:
                if split == "even":
                    even_noise_save_outs[key].append(noise_outputs[key])
                    even_signal_save_outs[key].append(signal_outputs[key])
                elif split == "odd":
                    odd_noise_save_outs[key].append(noise_outputs[key])
                    odd_signal_save_outs[key].append(signal_outputs[key])

             
    even_noise_filenames = os.path.join(*[path,"even",brange,noise_out_snr,f"freq_{bandmin}_{bandmax}_{nperband*len(range_bands)}.hdf5"])
    even_signal_filenames = os.path.join(*[path,"even",brange,signal_out_snr,f"freq_{bandmin}_{bandmax}_{nperband*len(range_bands)}.hdf5"])
    odd_noise_filenames = os.path.join(*[path,"odd",brange,noise_out_snr,f"freq_{bandmin}_{bandmax}_{nperband*len(range_bands)}.hdf5"])
    odd_signal_filenames = os.path.join(*[path,"odd",brange,signal_out_snr,f"freq_{bandmin}_{bandmax}_{nperband*len(range_bands)}.hdf5"])

    for fname, temp_data in zip([even_noise_filenames, even_signal_filenames, odd_noise_filenames, odd_signal_filenames], [even_noise_save_outs, even_signal_save_outs, odd_noise_save_outs, odd_signal_save_outs]):
        if not os.path.isdir(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        print(fname, save_options)
        with h5py.File(fname,"w") as f:
            for key in save_options:
                if key == "pars":
                    # pars are stored as a dictionary so have to be saved differently to hdf5
                    parkeys = list(temp_data[key][0].keys())
                    f.create_dataset("parnames", data = parkeys)
                    f.create_dataset("pars", data = np.array([[td[key] for key in parkeys] for td in temp_data[key]]))
                else:
                    try:
                        f.create_dataset(key, shape=np.shape(temp_data[key]), data=np.array(temp_data[key]))
                    except ValueError as e:
                        print(key)
                        print(len(temp_data[key]))
                        lengths = [temp_data[key][i] for i in range(len(temp_data[key]))]
                        print(len(temp_data[key][0]))
                        print(max(lengths), min(lengths))
                        #print(np.shape(temp_data[key]))
                        #print(np.array(temp_data[key]).dtype)
                        #print(np.array(temp_data[key]).shape)
                        print(e)
                        raise Exception("Error on hdf5 save array")

    end_time = time.time()
    date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    with open(os.path.join(path,"timing.txt"),"a") as f:
        f.write("time-for-gen    {}    {} \n".format(end_time-start_time,date))


def run_and_inj(fmin,width,outpath,resize_image=False,snrrange = (50,150),av_sh=None,nsft=22538,tstart=931052949,tsft=1800.,test = False,save_options=None, stride = 1, degfree = 96, brange = ""):
    
    fmax = np.round(fmin + width,2)

    sig = cw.GenerateSignal()
    sig.alpha = 0
    sig.delta = 0
    sig.cosi = 0
    sig.phi0 = 0
    sig.psi = 0
    sig.f = [0,0]
    sig.earth_ephem = "/home/joseph.bayley/repositories/lalsuite/lalpulsar/lib/earth00-40-DE430.dat.gz"
    sig.sun_ephem = "/home/joseph.bayley/repositories/lalsuite/lalpulsar/lib/sun00-40-DE430.dat.gz"
    sig.snr = 0
    sig.fmin = fmin
    sig.fmax = fmax
    sig.tref = tstart
    
    Sn = {"H1":av_sh[0],"L1":av_sh[1]}
    if test:
        out_snr_noise = "./"
    else:
        out_snr_noise = "snr_0.0_0.0"
    data = sig.get_spectrogram(tstart = tstart, nsft = nsft,tref=tstart,tsft=tsft,fmin=fmin,fmax=fmax,snr=0,dets= ["H1","L1"],Sn=Sn)
    data.sum_sfts()
    data.downsamp_frequency(stride=stride,data_type="summed_norm_sft_power",remove_original=False)
    if np.shape(data.L1.downsamp_summed_norm_sft_power) != np.shape(data.H1.downsamp_summed_norm_sft_power):
        print(fmin,np.shape(data.L1.downsamp_summed_norm_sft_power),np.shape(data.H1.downsamp_summed_norm_sft_power))
        sys.exit()
    noise_outputs = cnn_data_gen.return_outputs(out_snr=out_snr_noise,datah=data.H1.downsamp_summed_norm_sft_power,datal=data.L1.downsamp_summed_norm_sft_power, fmin=fmin,fmax=fmax,snr=0,depth=0,h0=0, resize_image=resize_image, save_options = save_options, epochs = data.H1.summed_epochs, degfree = degfree, brange = brange)
    noise_outputs["pars"] = {"snr":0}
    noise_outputs["pars"]["tref"] = tstart
    noise_outputs["pars"]["snr"] = 0
    noise_outputs["pars"]["av_sh"] = np.nanmedian(av_sh)
    noise_outputs["pars"]["width"] = width
    noise_outputs["pars"]["fmin"] = data.H1.fmin
    noise_outputs["pars"]["fmax"] = data.H1.fmax

    del data, sig

    # inject signal into band

    sig = cw.GenerateSignal()
    
    snrstart,snrend = snrrange[0],snrrange[1]
            
    sigfreq = np.random.uniform(0,1)*(width)*0.5 + fmin + (width)*0.25
        
    snr = int(np.random.rand(1)*(snrend-snrstart) + snrstart)
        
    # freq, spindown^n, alpha, sin(delta), phi0, psi, cosi
    #params_low =  [fmin + 0.25*width,       -1e-9 ,0      ,-1,0      ,0      ,-1.0]
    #params_high = [fmin + width-0.25*width, -1e-16,2*np.pi, 1,2*np.pi,np.pi/2,1.0]
    params_low =  [fmin,       -1e-9 ,0      ,-1,0      ,0      ,-1.0]
    params_high = [fmin + width, -1e-16,2*np.pi, 1,2*np.pi,np.pi/2,1.0]

    data_list_signal = []
    data_list_noise = []
    nsft = int(nsft)
    param_list = ["f","fd","alpha","sindelta","phi0","psi","cosi"]
    pars = {}
    for j in np.linspace(0,len(param_list)-1,len(param_list)).astype(int):
        pars[param_list[j]] = params_low[j] + np.random.uniform(0,1)*(params_high[j] - params_low[j])

    #pars["fd"] = -10**(pars["logfd"])
    #pars["fd"] = np.random.uniform(-1,1)*10**pars["fd"]

    sig.alpha = pars["alpha"]
    sig.delta = np.arcsin(pars["sindelta"])
    sig.cosi = pars["cosi"]
    sig.phi0 = pars["phi0"]
    sig.psi = pars["psi"]
    sig.f = [pars["f"],pars["fd"]]
    sig.earth_ephem = "/home/joseph.bayley/repositories/lalsuite/lalpulsar/lib/earth00-40-DE430.dat.gz"
    sig.sun_ephem = "/home/joseph.bayley/repositories/lalsuite/lalpulsar/lib/sun00-40-DE430.dat.gz"
    sig.snr = snr
    sig.fmin = fmin
    sig.fmax = fmax
    sig.tref = tstart

    epochs = []

    print("in minmaxf", fmin, fmax)  
    data = sig.get_spectrogram(tstart = tstart, nsft = nsft,tref=tstart,tsft=tsft,fmin=fmin,fmax=fmax,snr=snr,dets= ["H1","L1"],Sn=Sn)
    data.sum_sfts()
    data.downsamp_frequency(stride=stride,data_type="summed_norm_sft_power",remove_original=False)
    if np.shape(data.L1.downsamp_summed_norm_sft_power) != np.shape(data.H1.downsamp_summed_norm_sft_power):
        print(np.shape(data.L1.downsamp_summed_norm_sft_power),np.shape(data.H1.downsamp_summed_norm_sft_power))
        sys.exit()

    print("freqs",fmin,fmax, data.H1.frequencies[0], data.H1.frequencies[-1] + 1./data.tsft)
    epochs = data.H1.summed_epochs

    h0 = np.sqrt(np.nanmedian(av_sh))/data.depth

    inj_track = (sig.get_pulsar_path(epochs=epochs,det="H1") - sig.fmin)*data.tsft 

    pars["tref"] = tstart
    pars["snr"] = snr
    pars["av_sh"] = np.nanmedian(av_sh)
    pars["h0"] = h0
    pars["depth"] = data.depth
    pars["width"] = width
    pars["fmin"] = data.H1.fmin
    pars["fmax"] = data.H1.fmax
    if test:
        out_snr_inj = "./"
    else:
        out_snr_inj = "snr_{}_{}".format(snrstart,snrend)

    signal_outputs = cnn_data_gen.return_outputs(out_snr=out_snr_inj,datah=data.H1.downsamp_summed_norm_sft_power,datal=data.L1.downsamp_summed_norm_sft_power, fmin=fmin,fmax=fmax,snr=snr, depth=data.depth,h0=h0,resize_image=resize_image,pars=pars,inj_track=inj_track,epochs=epochs, save_options=save_options, degfree = degfree, brange = brange)

    return noise_outputs, signal_outputs
