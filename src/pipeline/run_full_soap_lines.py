#!/usr/bin/env python
from __future__ import print_function
import matplotlib
matplotlib.use("agg")
import soapcw as soap
from soapcw import cw
import sys
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import configparser
import pickle
import h5py
from astropy.time import Time
import sys
import fcntl
import torch
import lalpulsar
import lal
from .soap_config_parser import SOAPConfig
from collections import OrderedDict

def set_limit(data,cut=180):
    """
    set the limits for matplotlib plot whet an upper cut has been applied
    """
    if data.max() > cut:
        vmax = cut
    else:
        vmax = data.max()
    vmin = data.min()
    return vmin,vmax

def get_sfts_in_range(tmin,tmax,sftlist):
    """
    load the sfts within some tmin and tmax range
    """
    in_range = []
    
    for sft in sftlist:
        stt = float(sft.split("-")[-2])
        #print(stt - tmin,tmax-stt)
        if stt >= tmin and stt < tmax:
            in_range.append(sft)
    return in_range

def get_sft_files(sftpaths):
    """
    look through sftpaths to find all sft files
    """
    sftlist = []
    if type(sftpaths) == str:
        sftpaths = sftpaths.split(",")
    for sftpath in sftpaths:
        #print(sftpath)
        for path, subdirs, files in os.walk(sftpath):
            #print(path, subdirs, files)
            for name in files:
                if name != "" and name.endswith(".sft"):
                    sftlist.append(os.path.join(path, name))
    return sftlist

def get_sft_files_find(output_dir, sftpaths, overwrite_file=False):
    sftlist = []
    if type(sftpaths) == str:
        sftpaths = sftpaths.split(",")
    fnames = []
    for sftpath in sftpaths:
        pathsplit = sftpath.split('/')
        pathappend = pathsplit[-2] if pathsplit[-1] == "" else pathsplit[-1]
        fname = os.path.join(output_dir, f"{pathappend}_sftfile.txt")
        fnames.append(fname)
        sftpath = sftpath.strip()
        print("sftfilename", fname)
        if not os.path.isfile(fname) or overwrite_file:
            with open(fname, "w") as f:
                findout1 = subprocess.Popen([
                        "find",
                        sftpath,
                        "-name",
                        "*.sft"
                    ], 
                    stdout=subprocess.PIPE,
                    shell=False)
                findout = subprocess.run([
                    "sort"],
                    stdin=findout1.stdout,
                    stdout=f,
                    shell=False
                )

                print(findout)
        else:
            print("SFT filelist exists")


    return fnames

def av_noise(noise,nsft=48):
    """
    get the median noise floor for every nsft sfts
    """
    av_noise = []
    
    for i in np.arange(len(noise))[::nsft]:
        if i + nsft > len(noise) - 1:
            av_noise.append(np.nanmedian(noise[i:]))
        else:
            av_noise.append(np.nanmedian(noise[i:i+nsft]))
    
    return np.array(av_noise)


def run_soap_onedet(
    config,
    degfree, 
    sft, 
    ind_start, 
    ind_end
    ):

    tr = soap.tools.transition_matrix(config["transitionmatrix"]["left_right_prob"])

    # run soap search using the two detector line aware statistic (one detector in gaps)
    # SOAP using SNR based statistic
    soaprunH1 = soap.single_detector(tr,sft.H1.downsamp_summed_norm_sft_power[:,ind_start:ind_end])
    soaprunL1 = soap.single_detector(tr,sft.L1.downsamp_summed_norm_sft_power[:,ind_start:ind_end])

    return soaprunH1, soaprunL1

def load_sft_band(config, filenames, minfreq, maxfreq, tmin, tmax, verbose = False):

    print("summedtype", type(config["data"]["n_summed_sfts"]))
    sft = cw.LoadSFT(
        filenames,
        norm=True,
        filled=True,
        save_rngmed=True,
        remove_sft=True,
        summed=config["data"]["n_summed_sfts"],
        fmin=minfreq,
        fmax=maxfreq,
        tmin=tmin,
        tmax=tmax,
        verbose=True
        )

    # set how many frequency bins to average over for each frequency band (based on spread of signal in 1800s sft)
    # hardcoded at the moment 
    for i, (minband, maxband) in enumerate(zip(config["data"]["band_starts"], config["data"]["band_ends"])):
        if minband <= minfreq < maxband:
            stride = config["data"]["strides"][i]
            degfree = int(2 * config["data"]["n_summed_sfts"] * stride)
            plotcut = stride * 180
            width = config["data"]["band_widths"][i]
            break
        else:
            continue


    if verbose:
        print(f"Nsum: {config['data']['n_summed_sfts']}")
        print("H1shape: {}, nsft: {}".format(np.shape(sft.H1.norm_sft_power), sft.H1.nsft))
        print("L1shape: {}, nsft: {}".format(np.shape(sft.L1.norm_sft_power), sft.L1.nsft))

    sft.downsamp_frequency(stride=stride,data_type="summed_norm_sft_power",remove_original=False)

    if "H1" not in sft.det_names:
        sft.H1 = cw.SFT()
        sft.H1.downsamp_summed_norm_sft_power = np.ones(sft.L1.downsamp_summed_norm_sft_power.shape)*np.nan
        sft.det_names = ["H1", "L1"]
        sft.H1.rng_med = sft.L1.rng_med * np.nan
    elif "L1" not in sft.det_names:
        sft.L1 = cw.SFT()
        sft.L1.downsamp_summed_norm_sft_power = np.ones(sft.H1.downsamp_summed_norm_sft_power.shape)*np.nan
        sft.det_names = ["H1", "L1"]
        sft.L1.rng_med = sft.H1.rng_med * np.nan


    # print some variables shapes to confirm what is expected
    if verbose:
        print("H1shape downsamp: {}, freq: {}".format(np.shape(sft.H1.downsamp_summed_norm_sft_power), minfreq))
        print("L1shape downsamp: {}, freq: {}".format(np.shape(sft.L1.downsamp_summed_norm_sft_power), minfreq))
        print("H1shape sum: {}, freq: {}".format(np.shape(sft.H1.summed_norm_sft_power), minfreq))
        print("L1shape sum: {}, freq: {}".format(np.shape(sft.L1.summed_norm_sft_power), minfreq))
        print("H1epochs len: {}, sumep: {}, start {} end {}".format(np.shape(sft.H1.epochs), np.shape(sft.H1.summed_epochs), sft.H1.summed_epochs[0], sft.H1.summed_epochs[-1]))
        print("L1epochs len: {}, sumep: {},  start {} end {}".format(np.shape(sft.L1.epochs), np.shape(sft.L1.summed_epochs), sft.L1.summed_epochs[0], sft.L1.summed_epochs[-1]))


    return sft, stride, degfree, plotcut, width

def plot_setup(
    fig, 
    ax, 
    plotcut, 
    H_temp_sft, 
    L_temp_sft, 
    temp_noise,
    sum_nsft, 
    soaprunH1,
    soaprunL1, 
    label_fontsize, 
    tick_fontsize, 
    xticks, 
    xticklabels, 
    xticktimes,
    yticks, 
    yticklabels):
    """_summary_

    Args:
        fig (_type_): _description_
        ax (_type_): _description_
        plotcut (_type_): _description_
        H_temp_sft (_type_): _description_
        L_temp_sft (_type_): _description_
        sum_nsft (_type_): _description_
        soaprun (_type_): _description_
        label_fontsize (_type_): _description_
        tick_fontsize (_type_): _description_
        xticks (_type_): _description_
        xticklabels (_type_): _description_
        yticks (_type_): _description_
        yticklabels (_type_): _description_

    Returns:
        _type_: _description_
    """

    # plot images 
    H_imlim = set_limit(H_temp_sft, cut=plotcut)
    H_im = ax[0].imshow(H_temp_sft.T,cmap="cividis",aspect="auto",origin="lower",vmin=H_imlim[0],vmax=H_imlim[1], interpolation="none")
    L_imlim = set_limit(L_temp_sft, cut=plotcut)
    L_im = ax[1].imshow(L_temp_sft.T,cmap="cividis",aspect="auto",origin="lower",vmin=L_imlim[0],vmax=L_imlim[1], interpolation="none")
    imvh1 = ax[2].imshow(soaprunH1.vitmap.T,cmap="cividis",aspect="auto",origin="lower", interpolation="none")
    imvl1 = ax[3].imshow(soaprunL1.vitmap.T,cmap="cividis",aspect="auto",origin="lower", interpolation="none")
            
    # plot track power s
    Hpwline, = ax[4].plot(np.arange(sum_nsft),np.array(soap.tools.track_power(soaprunH1.vit_track,H_temp_sft)),color="C3",label="H1")
    Lpwline, = ax[4].plot(np.arange(sum_nsft),np.array(soap.tools.track_power(soaprunL1.vit_track,L_temp_sft)),color="C2",label="L1")

    # plot viterbi statistic
    Hstatline, = ax[5].plot(np.arange(sum_nsft),np.array(soap.tools.stat_on_path(soaprunH1.vit_track,soaprunH1.V)),color="C3",label="H1")
    Lstatline, = ax[5].plot(np.arange(sum_nsft),np.array(soap.tools.stat_on_path(soaprunL1.vit_track,soaprunL1.V)),color="C2",label="L1")

    # plot noise 
    Hnoise, = ax[6].plot(np.arange(sum_nsft),temp_noise["H1"],color="C3",label="H1")
    Lnoise, = ax[6].plot(np.arange(sum_nsft),temp_noise["L1"],color="C2",label="L1")

                
    # set colorbar parameters
    Hcbar = fig.colorbar(H_im,ax = ax[0])
    Lcbar = fig.colorbar(L_im,ax = ax[1])
    cbarvh1 = fig.colorbar(imvh1,ax = ax[2])
    cbarvl1 = fig.colorbar(imvl1,ax = ax[3])
    Hcbar.set_label("Normalised SFT \n power (H1)",fontsize = label_fontsize)
    Lcbar.set_label("Normalised SFT \n power (L1)",fontsize = label_fontsize)
    cbarvh1.set_label("Normalised \n log-odds (H1)",fontsize = label_fontsize)
    cbarvl1.set_label("Normalised \n log-odds (L1)",fontsize = label_fontsize)
    Hcbar.ax.tick_params(labelsize=tick_fontsize)
    Lcbar.ax.tick_params(labelsize=tick_fontsize)
    cbarvh1.ax.tick_params(labelsize=tick_fontsize)
    cbarvl1.ax.tick_params(labelsize=tick_fontsize)

    # set tick labels
    for i in range(6):
        ax[i].set_xticks(xticks)

    ax[5].set_xticklabels(xticklabels)

    for i in range(3):
        ax[i].set_yticks(yticks)
        ax[i].set_yticklabels(yticklabels)

    ax[5].xaxis.set_tick_params(rotation=0,labelsize=tick_fontsize)
    for i in range(6):
        ax[i].yaxis.set_tick_params(rotation=0,labelsize=tick_fontsize)

    # set axis labels
    ax[5].set_xlabel("Time - {}[s]".format(xticktimes[0].year),fontsize=label_fontsize)
    ax[0].set_ylabel("Frequency [Hz]",fontsize=label_fontsize)
    ax[1].set_ylabel("Frequency [Hz]",fontsize=label_fontsize)
    ax[2].set_ylabel("Frequency [Hz]",fontsize=label_fontsize)
    ax[3].set_ylabel("Frequency [Hz]",fontsize=label_fontsize)
    ax[4].set_ylabel("Normalised SFT power \n along track",fontsize=label_fontsize)
    ax[5].set_ylabel("Statistic along path",fontsize=label_fontsize)
    ax[6].set_ylabel("Mean noise floor \n for band",fontsize=label_fontsize)

    # set axis limits
    for i in [0,1,2,4,5]:
        ax[i].set_xlim([0,sum_nsft])

    ax[0].set_ylim([0,len(H_temp_sft[0])])
    ax[1].set_ylim([0,len(L_temp_sft[0])])
    ax[2].set_ylim([0,len(L_temp_sft[0])])
    ax[3].set_ylim([0,len(L_temp_sft[0])])
            
    ax[4].legend()
    ax[5].legend()

    for i in [4,5,6]:
        ax[i].grid()

    Hcbar.outline.set_visible(False)
    Lcbar.outline.set_visible(False)
    cbarvh1.outline.set_visible(False)
    cbarvl1.outline.set_visible(False)
            
    # set widths oof plots
    fig.subplots_adjust(hspace=0)

    box = ax[4].get_position()
    ax[4].set_position([box.x0, box.y0, box.width * 0.8 , box.height])

    box = ax[5].get_position()
    ax[5].set_position([box.x0, box.y0, box.width * 0.8 , box.height])
            
    box = ax[6].get_position()
    ax[6].set_position([box.x0, box.y0, box.width * 0.8 , box.height])

    return H_im, L_im, imvh1, imvl1, Hnoise, Lnoise, Hpwline, Lpwline, Hstatline

def update_plot(fig, ax, H_im, L_im, imvh1, imvl1, Hnoise, Lnoise, Hpwline, Lpwline, Hstatline, H_temp_sft, L_temp_sft, soaprunH1, soaprunL1, temp_noise, yticks, yticklabels, plotcut, ):

        # set the yaxis labels for the first two figures, i.e. the sfts and feather plot
    for axs in fig.axes[:3]:
        try:
            axs.set_yticks(yticks)
            axs.set_yticklabels(yticklabels)
        except:
            pass
            
    # set the new data and limits for the sft plot
    H_im.set_data(H_temp_sft.T)
    L_im.set_data(L_temp_sft.T)
    H_imlim = set_limit(H_temp_sft, cut = plotcut)
    H_im.set_clim(vmin=H_imlim[0],vmax=H_imlim[1])
    L_imlim = set_limit(L_temp_sft, cut = plotcut)
    L_im.set_clim(vmin=L_imlim[0],vmax=L_imlim[1])
    
    # set the new data and limits for the feather plot
    # take the log of vitmap to make it more visible (also add small value so no taking log of 0)
    # vitmaps are normalised between 0 and 1
    imvh1.set_data(soaprunH1.vitmap.T)
    #imvh1.set_clim(vmin=soaprunH1.vitmap.min(),vmax=soaprunH1.vitmap.max())
    imvl1.set_data(soaprunL1.vitmap.T)
    #imvl1.set_clim(vmin=soaprunL1.vitmap.min(),vmax=soaprunL1.vitmap.max())
    
    # set the new data and limits for the noise power plot
    Hnoise.set_ydata(temp_noise["H1"])
    Lnoise.set_ydata(temp_noise["L1"])
    ax[6].relim()
    ax[6].autoscale_view()
    
    # set the new data and limits for the track power plot
    Hpwline.set_ydata(soap.tools.track_power(soaprunH1.vit_track,H_temp_sft))
    Lpwline.set_ydata(soap.tools.track_power(soaprunL1.vit_track,L_temp_sft))
    ax[4].relim()
    ax[4].autoscale_view()

    # set the new data and limits for the vitstat plot
    Hstatline.set_ydata(soap.tools.stat_on_path(soaprunH1.vit_track,soaprunH1.V))
    Hstatline.set_ydata(soap.tools.stat_on_path(soaprunL1.vit_track,soaprunL1.V))
    ax[5].relim()
    ax[5].autoscale_view()
    
    # refresh the figure
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    # remove the soaprun track from the sft plot
    """
    if num !=0:
        try:
            Hline.remove()
            Lline.remove()
        except:
            pass
    """

def write_locked_file(filepath, data, write_type="rb"):
    # write table of statistics 
    with open(filepath,write_type) as f:  
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                total_data = pickle.load(f) 
                total_data.update(data)
                fcntl.flock(f, fcntl.LOCK_UN)
                break
            except IOError or EOFError:
                time.sleep(0.1)
    return total_data



def run_soap_in_band(config, minfreq, maxfreq, weekstarttime=None, verbose = False, force_overwrite = False):
    """_summary_

    Args:
        config (_type_): _description_
        sftfiles (_type_): _description_
        minfreq (_type_): _description_
        maxfreq (_type_): _description_
        verbose (bool, optional): _description_. Defaults to False.

    Raises:
        Exception: _description_
        Exception: _description_
        Exception: _description_
        Exception: _description_
    """
    
    # get the sft files and start/end times
    outpath = os.path.join(*[config["output"]["save_directory"],config["data"]["obs_run"]])    

    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    if weekstarttime is None:
        outpath_save = os.path.join(outpath,config["output"]["sub_directory"])
        if os.path.isdir(outpath_save):
            print(f"This week has already been generated: {outpath_save}")
            if force_overwrite == False:
                return 0
            else:
                print(f"WARNING!!! - Overwriting current existing directory: {outpath_save}")

    # get the list of sft filenames by iterating over subdirs 
    # this saves the filelise to a file
    sftfiles = config["input"]["load_directory"]
    print("num files: ", len(sftfiles))
    start_load = time.time()
    sft_filelists = get_sft_files_find(outpath, sftfiles, overwrite_file=config["output"]["overwrite_files"])

    print("SFT filelist", sft_filelists)
    # load in the sft filelists 
    sftlist = []
    for sftfile in sft_filelists:
        with open(sftfile,"r") as f:
            sftlist.extend(f.readlines())
    
    if len(sftlist) == 0:
        raise Exception("No SFTs found")

    # get all the start times from the files
    try:
        sttime = [float(os.path.basename(name).split("-")[-2]) for name in sftlist]
    except Exception as e:
        print(os.path.basename(name))
        raise Exception(e)

    # define start and end times
    print("numsfts: {}".format(len(sftlist)))
    print("Time find fnames: ", time.time() - start_load)
    if "start_time" in config["data"].keys():
        tmin = float(config["data"]["start_time"])
        tmax = float(config["data"]["end_time"])
    else:
        if weekstarttime is not None:
            weeklength = 3600*24*7
            tmin = weekstarttime
            tmax = tmin + weeklength
        else:
            tmin = np.min(sttime) 
            tmax = np.max(sttime) + 1800 # start of last sft plus length (in this case we are only using 1800s sfts)
            #tmax = min(sttime) + 48*10*1800 # smaller range for testing
    
    # get the sfts in the specified time range 
    start_range = time.time()
    sftlist = get_sfts_in_range(tmin,tmax,sftlist)
    time_getsftsrange = time.time() - start_range
    print("time_getsftsrange: ", time_getsftsrange)

    # write time of last run to file
    lastruntime = os.path.join(outpath, "lastruntime.txt")
    if os.path.isfile(lastruntime):
        with open(lastruntime,"r") as f:
            mnt = np.loadtxt(f)
    
    # initialise figure and axes
    fig, axes = None, None
    
    if len(sftlist) == 0:
        raise Exception("No SFTs found in range")

    # define start times in gps and date format
    start_time_gps = Time(tmin,format="gps")
    end_time_gps = Time(tmax,format="gps")
    startdatestring = start_time_gps.iso.split(" ")[0].replace("-","")
    enddatestring = end_time_gps.iso.split(" ")[0].replace("-","")

    # set output directories
    outpath_save = os.path.join(outpath,config["output"]["sub_directory"])

    if os.path.isdir(outpath_save):
        pathexists = True
    else:
        pathexists = False

    # combine all filenames infor format input to lalpulsar loading (via LoadSFT)
    filenames = ";".join([s.strip() for s in sftlist])

    # load in the sfts, normalise them to the running median and fill the gaps
    start_load_sft = time.time()
    sft, stride, degfree, plotcut, width = load_sft_band(config, filenames, minfreq, maxfreq, tmin, tmax, verbose=verbose)
    # print the time takes to load the data
    load_time = time.time() - start_load_sft
    print("Load time: ", load_time)

    # define varialbes of sft
    detector = sft.det_names[0]
    tsft = getattr(sft,detector).tsft
    sum_nsft = len(getattr(sft,detector).summed_epochs)
    bin_width = stride/float(tsft)
    nbins = len(getattr(sft,detector).downsamp_frequencies)
            
    # set the start time of data and the indicies of the frequency bands the sft will be split into
    tstart = getattr(sft,detector).epochs[0]
    #start_indicies = np.arange(minfreq*sft.tsft,maxfreq*sft.tsft,width*sft.tsft/2).astype(int)
    start_indicies = np.round(np.arange(0,nbins,width/(2*bin_width))).astype(int)[:-1]

    # arrays are for timing different parts of the run
    av_soapruns = []
    av_cnnrun = []
    av_images = []
    av_save = []


    # initialise the plot figure and set some parameters
    fig, ax = plt.subplots(nrows=6,figsize=(18,25),sharex=True)
    tick_fontsize = 13
    label_fontsize = 20
    
    table_data = OrderedDict()
    
    start_run = time.time()
    breakloop = False
    # loop over all of the sub-bands 


    # write time of last run to file
    lastruntime = os.path.join(outpath, "lastruntime.txt")
    if os.path.isfile(lastruntime):
        with open(lastruntime,"r") as f:
            mnt = np.loadtxt(f)
    
    # initialise figure and axes
    fig, axes = None, None
    
    if len(sftlist) == 0:
        raise Exception("No SFTs found")

    # define start times in gps and date format
    start_time_gps = Time(tmin,format="gps")
    end_time_gps = Time(tmax,format="gps")
    startdatestring = start_time_gps.iso.split(" ")[0].replace("-","")
    enddatestring = end_time_gps.iso.split(" ")[0].replace("-","")

    # set output directories
    if weekstarttime is not None:
        outpath_save = os.path.join(outpath,config["output"]["sub_directory"], f"week_{startdatestring}_{enddatestring}")
    else:
        outpath_save = os.path.join(outpath,config["output"]["sub_directory"])

    if os.path.isdir(outpath_save):
        pathexists = True
        print(f"This week has already been generated: {outpath_save}")
        if force_overwrite == False:
            return 0
        else:
            print(f"WARNING!!! - Overwriting current existing directory: {outpath_save}")
    else:
        pathexists = False
                
    outpath_save_plots = os.path.join(outpath_save,"plots")
    outpath_save_track = os.path.join(outpath_save,"tracks")
    output_save_files = os.path.join(outpath_save, "data")
            
    for direc in [outpath_save_plots,outpath_save_track, output_save_files]:
        if not os.path.isdir(direc):
            os.makedirs(direc)
        else:
            pass

    tablefile = os.path.join(output_save_files,f"table_{minfreq}_{maxfreq}.hdf5")
    if os.path.isfile(tablefile):
        if not force_overwrite:
            raise FileExistsError(f"File {tablefile} already exists, add force-overwrite to arguments to overwrite file" )

    # combine all filenames infor format input to lalpulsar loading (via LoadSFT)
    filenames = ";".join([s.strip() for s in sftlist])

    # load in the sfts, normalise them to the running median and fill the gaps
    start_load_sft = time.time()
    sft, stride, degfree, plotcut, width = load_sft_band(config, filenames, minfreq, maxfreq, tmin, tmax, verbose=verbose)
    # print the time takes to load the data
    load_time = time.time() - start_load_sft
    print("Load time: ", load_time)

    # define varialbes of sft
    detector = sft.det_names[0]
    tsft = getattr(sft,detector).tsft
    sum_nsft = len(getattr(sft,detector).summed_epochs)
    bin_width = stride/float(tsft)
    nbins = len(getattr(sft,detector).downsamp_frequencies)

            
    # set the start time of data and the indicies of the frequency bands the sft will be split into
    tstart = getattr(sft,detector).epochs[0]
    #start_indicies = np.arange(minfreq*sft.tsft,maxfreq*sft.tsft,width*sft.tsft/2).astype(int)

    ####################
    ### Key difference for line search is no overlapping bands
    ###################
    start_indicies = np.round(np.arange(0,nbins,width/(bin_width))).astype(int)[:-1]
    #####################

    
    # arrays are for timing different parts of the run
    av_soapruns = []
    av_cnnrun = []
    av_images = []
    av_save = []


    # initialise the plot figure and set some parameters
    fig, ax = plt.subplots(nrows=7,figsize=(18,25),sharex=True)
    tick_fontsize = 13
    label_fontsize = 20
    
    table_data = OrderedDict()
    
    start_run = time.time()
    breakloop = False
    # loop over all of the sub-bands 

    for num,start in enumerate(start_indicies):
        
        # set the minimum freuency and maximum frequency of the sub-band
        fmin,fmax = minfreq + start*bin_width, minfreq + start*bin_width + width 

        # start and end index of sfts
        ind_start = int(start)
        ind_end = int(start + width/bin_width)

        if breakloop == True:
            break
        if fmax >= maxfreq:
            breakloop = True

        if ind_end > nbins:
            continue

        # get the subband from the sft
        temp_sft = {}
        temp_noise = {}
        for det in getattr(sft,"det_names"):
            # get noise floor of subband
            temp_n = np.sqrt(np.nanmedian(getattr(sft,det).rng_med[:,ind_start:ind_end],axis=1))   
            print("tempn", det, np.shape(temp_n)) 
            # get nsfts and the average noise floor
            temp_noise[det] = av_noise(temp_n, nsft=config["data"]["n_summed_sfts"])[:sum_nsft]

            print("tempnoise", np.shape(temp_noise[det]))


        # run the single detector viterbi search and append the time taken to run to previous list
        st_vt = time.time()
        soaprunH1, soaprunL1 = run_soap_onedet(config, degfree, sft, ind_start, ind_end)
        av_soapruns.append(time.time()-st_vt)

        save_path = os.path.join(outpath_save_plots,"notrack_F{}_{}.png".format(np.round(fmin,2),np.round(fmax,2)))
        save_path_track = os.path.join(outpath_save_plots,"track_F{}_{}.png".format(np.round(fmin,2),np.round(fmax,2)))
        
        # Set up the table data
        table_save_data = OrderedDict()
        table_save_data["fmin"] = fmin
        table_save_data["fmax"] = fmax
        table_save_data["H1_viterbistat"] = soaprunH1.max_end_prob
        table_save_data["L1_viterbistat"] = soaprunL1.max_end_prob
        
        # if cnn model provided run the cnn on viterbi map  
        st_cnn = time.time()    

        table_save_data["plot_path"] = save_path_track


        for key, val in table_save_data.items():
            table_data.setdefault(key, [])
            if type(val) == str:
                val = val.encode("ascii")
            table_data[key].append(val)

        av_cnnrun.append(time.time()-st_cnn)

        # only for two detectors, will generalise laterc
        tracks = {}
        tracks["H1"]  = np.array(soaprunH1.vit_track)*bin_width + fmin
        tracks["L1"]  = np.array(soaprunL1.vit_track)*bin_width + fmin


        # save the tracks for each detector
        track_times = sft.H1.summed_epochs 
        with open(os.path.join(outpath_save_track,"track_F{}_{}.txt".format(np.round(fmin,2),np.round(fmax,2))),"w") as f:
            np.savetxt(f,np.transpose([track_times,tracks["H1"],tracks["L1"]]).astype(float))

        #table_data["{}_{}".format(fmin,fmax)] = table_save_data


        # set the save paths for the plots
        st_im = time.time()
                
        # create masked array to plots the sfts
        H_temp_sft = np.ma.masked_where(sft.H1.downsamp_summed_norm_sft_power[:,ind_start:ind_end] == degfree,sft.H1.downsamp_summed_norm_sft_power[:,ind_start:ind_end],copy=True)
        L_temp_sft = np.ma.masked_where(sft.L1.downsamp_summed_norm_sft_power[:,ind_start:ind_end] == degfree,sft.L1.downsamp_summed_norm_sft_power[:,ind_start:ind_end],copy=True)
                
        # define the signal track
        if len(sft.H1.downsamp_summed_norm_sft_power[:,ind_start:ind_end]) != len(soaprunH1.vit_track):
            raise Exception("length of data {} different to viterbi track {} , {}, {}, {}, {}".format(len(sft.H1.downsamp_summed_norm_sft_power[:,ind_start:ind_end]),len(soaprunH1.vit_track1) ,fmin,fmax,minfreq,maxfreq))
            
        # set gaps in viterbi track to nan for plots
        H_track = soaprunH1.vit_track
        H_track[sft.H1.downsamp_summed_norm_sft_power[:,0] == degfree] == np.nan
        L_track = soaprunL1.vit_track
        L_track[sft.L1.downsamp_summed_norm_sft_power[:,0] == degfree] == np.nan
        
        # set gaps to nan in noise estimates
        print("datashape", np.shape(sft.H1.downsamp_summed_norm_sft_power))
        for det in getattr(sft,"det_names"):
            temp_noise[det][sft.H1.downsamp_summed_norm_sft_power[:,0] == degfree] = np.nan
            
        if len(H_track) != sum_nsft:
            sum_nsft = len(H_track)
            print("sum_nsft not equal to length of viterbi")

        if len(H_track) != len(H_temp_sft):
            raise Exception("Viterbi track and tsft not same length")
                
        # set the xaxis of the plot to each month of the year and its date
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        xticks = np.linspace(0,(sum_nsft-1),10)
        xticktimes = [Time(tstart + i*48*tsft,format='gps').datetime for i in xticks]
        xticklabels = ["{}-{}".format(tm.day,months[tm.month-1]) for tm in xticktimes]
        
        # set the yaxis of the plot as the frequency band
        yticks = np.linspace(0,len(H_temp_sft[0]),8).astype(int)
        yticklabels = np.round(np.linspace(fmin,fmax,8),2)
        
        # if its the first plot then set up labels and plots
        if num == 0:

            H_im, L_im, imvh1, imvl1, Hnoise, Lnoise, Hpwline, Lpwline, Hstatline = plot_setup(
                                                                                        fig, 
                                                                                        ax, 
                                                                                        plotcut, 
                                                                                        H_temp_sft, 
                                                                                        L_temp_sft, 
                                                                                        temp_noise,
                                                                                        sum_nsft, 
                                                                                        soaprunH1,
                                                                                        soaprunL1, 
                                                                                        label_fontsize, 
                                                                                        tick_fontsize, 
                                                                                        xticks, 
                                                                                        xticklabels, 
                                                                                        xticktimes,
                                                                                        yticks, 
                                                                                        yticklabels)
                    
        else:
            # for other plots just change data and axis labels (runs faster that generating new figure)
            
            update_plot(fig, ax, H_im, L_im, imvh1, imvl1, Hnoise, Lnoise, Hpwline, Lpwline, Hstatline, H_temp_sft, L_temp_sft, soaprunH1, soaprunL1, temp_noise, yticks, yticklabels, plotcut)

            Hline.set_data([], [])
            Lline.set_data([], [])

        # save the figure without the soaprun track on sft plot
        av_images.append(time.time() - st_im)
        st_sv = time.time()
        fig.savefig(save_path,bbox_inches="tight",dpi=60)
        
        if num == 0:
            # add the soaprun trck to sft plot
            Hline,  = ax[0].plot(np.arange(sum_nsft),H_track,color="red",label="Soaprun track",marker="o",ms=3)
            Lline, = ax[1].plot(np.arange(sum_nsft),L_track,color="red",label="Soaprun track",marker="o",ms=3)
        else:
            Hline.set_data(np.arange(sum_nsft),H_track)
            Lline.set_data(np.arange(sum_nsft),L_track)
        
        # save second figure with soaprun track
        fig.savefig(save_path_track,bbox_inches="tight",dpi=60)
        
        av_save.append(time.time() - st_sv)
        del H_temp_sft,L_temp_sft, temp_noise
     
    # write table of statistics 
    if os.path.isfile(tablefile):
        if not force_overwrite:
            new_tablefile = os.path.join(output_save_files,f"table_{minfreq}_{maxfreq}_{np.random.randint(low=0,high=1000)}.hdf5")
            print(f"File exists: {tablefile}, writing to {new_tablefile}")
            tablefile = new_tablefile
        else:
            tablefile = tablefile
        #with open(new_tablefile,"wb") as f:  
        #    pickle.dump(table_data, f)
    else:
        tablefile = tablefile

    with h5py.File(tablefile, "w", track_order=True) as hf:
        for key, val in table_data.items():
            try:
                hf.create_dataset(key, data=np.array(val))
            except TypeError as e:
                print("ERROR: ",key, val)
                raise TypeError(e)

        hf.attrs["Run Time"] =  np.array(time.time() - start_run)
        hf.attrs["Av SOAP run time"] = np.array(np.mean(av_soapruns))
        hf.attrs["Av image gen time"] =  np.array(np.mean(av_images))
        hf.attrs["Av im save time"] = np.array(np.mean(av_save))
        hf.attrs["Sum SOAP run time"] = np.array(np.sum(av_soapruns))
        hf.attrs["Sum image gen time"] =  np.array(np.sum(av_images))
        hf.attrs["Sum im save time"] = np.array(np.sum(av_save))
        hf.attrs["load SFT time"] = np.array(load_time)
        hf.attrs["fmin"] = fmin
        hf.attrs["fmax"] = fmax

        #with open(tablefile,"wb") as f:  
        #    pickle.dump(table_data, f)
            

    # show the times taken for each part of run
    print("Run time: ", time.time() - start_run)
    print("Av soaprun time: ", np.mean(av_soapruns))
    print("Av image generation time: ", np.mean(av_images))
    print("Av image save time: ", np.mean(av_save))
    print("Sum soaprun time: ", np.sum(av_soapruns))
    print("Sum image generation time: ", np.sum(av_images))
    print("Sum image save time: ", np.sum(av_save))





def read_data(filename):
    """
    Read the data for the cnn
    """
    with open(filename,"r") as f:
        # read image subtract offset and divide my standard deviation                                                                            
        image = pickle.load(f)
    image = image/image.max()

    global image_shape
    image_shape = image.shape
    reshape_image = image.reshape(image.shape[0],image.shape[1],1)
    return reshape_image

def run(model,test_data,device="cpu"):
    # set the evaluation mode                                                                                                                               
    model.eval()

    # test loss for the data                                                                                                                                
    test_loss = 0
    samples = {}
    # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing                                            
    with torch.no_grad():
        test_batch, test_labels = torch.Tensor(np.array([val[0] for val in test_data])).to(device), torch.Tensor(np.array([val[1] for val in test_data])).to(device)

        outputs = model.test(test_batch)

    return outputs,test_labels.cpu().numpy()

def run_vitmap_cnn(modelfname, vit_data, size = (156,89)):
    """
    run the cnn
    """
    #model = load_model(modelfname)
    model = torch.load(modelfname, map_location="cpu").to("cpu")
    
    vit_resize = resize(vit_data, size, anti_aliasing=True)

    vit_resize = vit_resize/vit_resize.max()

    vit_resize = vit_resize.reshape(1,vit_resize.shape[0],vit_resize.shape[1])
    model.eval()
    with torch.no_grad():
        prob = model.test(torch.Tensor([vit_resize,]))

    return prob[0][0]

def run_vitmapstat_cnn(modelfname, vit_data, vit_stat, size = (156,89)):
    """
    run the cnn
    """
    #model = load_model(modelfname)
    model = torch.load(modelfname, map_location="cpu").to("cpu")
    
    vit_resize = resize(vit_data, size, anti_aliasing=True)

    vit_resize = vit_resize/vit_resize.max()

    vit_resize = vit_resize.reshape(1,vit_resize.shape[0],vit_resize.shape[1])
    model.eval()
    with torch.no_grad():
        prob = model.test(torch.Tensor([vit_resize,]), torch.Tensor([vit_stat,]))

    return prob[0][0]

def run_spect_cnn(modelfname, H_data, L_data, size = (156,89), degfree = 96):
    """
    run the cnn
    """
    #model = load_model(modelfname)
    model = torch.load(modelfname, map_location="cpu").to("cpu")
    
    H_resize = resize(H_data, size, anti_aliasing=True)
    L_resize = resize(L_data, size, anti_aliasing=True)

    H_resize = (H_resize - degfree)/(2*degfree)
    L_resize = (L_resize - degfree)/(2*degfree)

    H_resize = H_resize.reshape(1,H_resize.shape[0],H_resize.shape[1])
    L_resize = L_resize.reshape(1,L_resize.shape[0],L_resize.shape[1])
    spect_dat = np.vstack([H_resize, L_resize])

    model.eval()
    with torch.no_grad():
        prob = model.test(torch.Tensor([spect_dat,]))

    return prob[0][0]

def run_all_cnn(modelfname, vit_data, H_data, L_data, vit_stat, size = (156,89), degfree = 96):
    """
    run the cnn
    """
    #model = load_model(modelfname)
    model = torch.load(modelfname, map_location="cpu").to("cpu")

    vit_resize = resize(vit_data, size, anti_aliasing=True)
    H_resize = resize(H_data, size, anti_aliasing=True)
    L_resize = resize(L_data, size, anti_aliasing=True)

    vit_resize = vit_resize/vit_resize.max()
    H_resize = (H_resize - degfree)/(2*degfree)
    L_resize = (L_resize - degfree)/(2*degfree)

    vit_resize = vit_resize.reshape(vit_resize.shape[0],vit_resize.shape[1],1)
    H_resize = H_resize.reshape(1,H_resize.shape[0],H_resize.shape[1])
    L_resize = L_resize.reshape(1,L_resize.shape[0],L_resize.shape[1])
    
    spect_dat = np.vstack([H_resize, L_resize])

    model.eval()
    with torch.no_grad():

        prob = model.test(torch.Tensor([vit_resize,]), torch.Tensor([spect_dat,]), torch.Tensor([vit_stat,]) )

    return prob[0][0]


def main():
    #sftpath = "/home/pulsar/public_html/fscan/H1/weekly/H1Fscan_coherence/H1Fscan_coherence/"
    #channel = "H1_GDS-CALIB_STRAIN"
    #outpath = "/home/joseph.bayley/projects/soap_scripts/line_finding/imgs_test/"
    
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-c', '--config-file', help='config file', type=str, required=False, default=None)

    parser.add_argument('-o', '--out-path', help='top level of output directories', type=str)
    parser.add_argument('-i', '--sft-path', help='path to sfts', type=str, required=False)
    parser.add_argument('-s', '--start-freq', help='start frequency for band', type=float, required=False)
    parser.add_argument('-e', '--end-freq', help='end frequency for band', type=float, required=False)
    parser.add_argument('-w', '--band-width', help='band_width', type=float, required=False, default = 0.1)
    parser.add_argument('--stride', help='stride, how many frequency bins to sum', type=int, required=False, default = 1)
    parser.add_argument('-r', '--obs-run', help='observing run', type=str, required=False)
    parser.add_argument('-l', '--lookup-dir', help='path to soap lookup tables', type=str, required=False)
    parser.add_argument('-sd', '--sub-dir', help='sub directory to save data', type=str, required=False)
    parser.add_argument('-cv', '--cnn-vitmap-model', help='filename of vitmap cnn model', type=str, required=False, default=None)
    parser.add_argument('-cs', '--cnn-spect-model', help='filename of spect cnn model', type=str, required=False, default=None)
    parser.add_argument('-cvs', '--cnn-vitmapstat-model', help='filename of vitmapstat cnn model', type=str, required=False, default=None)
    parser.add_argument('-ca', '--cnn-all-model', help='filename of vitmapstatspect cnn model', type=str, required=False, default=None)
    parser.add_argument('--force-overwrite', help='force overwrite tables and plots in output directory', action=argparse.BooleanOptionalAction)


    try:                                                     
        args = parser.parse_args()  
    except:  
        sys.exit(1)

    if args.cnn_vitmap_model is not None:

        #import tensorflow as tf
        #from keras import backend as K
        #from keras.models import load_model

        from skimage.transform import resize
        import torch

    if args.config_file is not None:
        cfg = SOAPConfig(args.config_file)
        #cfg = configparser.ConfigParser()
        #cfg.read(args.config_file)
    else:
        #outpath,minfreq,maxfreq,obs_run="O3",vitmapmodelfname=None, spectmodelfname = None, vitmapstatmodelfname = None, allmodelfname = None, sub_dir = "soap",
        cfg = {"output":{}, "data":{}, "input":{}, }

    if args.out_path:
        cfg["output"]["save_directory"] = args.out_path

    if args.sft_path:
        sftpath = args.sft_path.split(",")
        cfg["input"]["load_directory"] = sftpath

    run_soap_in_band(cfg, args.start_freq, args.end_freq, force_overwrite=args.force_overwrite)

if __name__ == "__main__":
    main()