#!/home/joseph.bayley/.virtualenvs/soap36/bin/python
import matplotlib
matplotlib.use("agg")
import numpy as np
import sys
import os
#import plot
import pickle
import time
import datetime
import soapcw as soap
from soapcw import cw
import argparse
from skimage.transform import resize
import skimage.measure
from scipy.signal import butter,filtfilt
import matplotlib.pyplot as plt
import copy
from soapcw.cnn import generate_train_data
from soapcw.cnn import generate_test_data
#import mdc_test
from soapcw.cnn import generate_gaussian_train_data


def find_sft_file(indir,bandmin,bandmax):
    """
    find the narrowbanded sft file in a directory which runs between band min and band max
    args
    --------------
    indir: string
        search directory
    bandmin: float
        minimum frequency
    bandmax: float
        maximum frequency

    returns
    ----------------
    sft file that matches this string
    sft file patter : H-<nsft>_<det>_<tsft>SFT_NB_F<freq>Hz0_W<width>Hz180-<sttime>-<span>.sft
    """
    # lists all files in directory
    hindir = os.path.join(indir,"H1")
    lindir = os.path.join(indir,"L1")

    hname = None
    lname = None

    # loops over files to find one with matching string
    for fname in os.listdir(hindir):
        #print("F{:0>4}Hz".format(int(bandmin)), fname)
        if "F{:0>4}Hz".format(int(bandmin)) in fname:
            hname = fname

    for fname in os.listdir(lindir):
        #if "F{}_{}.sft".format(bandmin,bandmax) in fname:
        if "F{:0>4}Hz".format(int(bandmin)) in fname:
            lname = fname

    if hname is None or lname is None:
        print("NO SFT FOUND {}, {}, {}, F{:0>4}Hz".format(indir, bandmin, bandmax, int(bandmin)))

    return os.path.join(hindir,hname),os.path.join(lindir,lname)


def return_outputs(datah,datal,fmin,fmax,snr,out_snr="./",h0=None,depth=None,size=(156,89),resize_image=True,test_plot=False,pars=None,test=False,epochs=None,minmaxpaths=None,inj_track=None, save_options = ["vit_imgs","H_imgs","L_imgs","stats","sum_stats","pars","diffs","paths","powers","plots"], lookup_dir = "/home/joseph.bayley/data/soap/lookup_tables/line_aware_optimised/", degfree = 96, brange = "", totshape = None):
    """
    save the outputs of the SOAP search and the spectrograms of the data
    
    """
    #save_options = ["stats","paths", "pars", "powers"]
    #save_options = ["stats","paths", "pars", "powers","vit_imgs","H_imgs","L_imgs"]
    # define all save directories for images, stats, pars, plots etc. out_snr only exists for training data
    """
    outdirs = {}
    for dirname in save_options:
        outdirs[dirname] = os.path.join(*[outpath,dirname,brange,out_snr])
    """
    #print(outdirs)

    # set location of lookup table directory
    #lookup_dir = "/home/joseph.bayley/data/soap/lookup_tables/lookup_both_opt_gauss_2/"
    
    # for line veto, choose optimum parameters from previous searches
    #lookup_1 = str(lookup_dir) + "/log_signoiseline_1det_{}degfree_4.0_5.0_0.0387.pkl".format(degfree,4.0,5.0,0.0387)
    #lookup_2 = str(lookup_dir) + "/log_signoiseline_2det_{}degfree_4.0_5.0_0.0387.pkl".format(degfree,4.0,5.0,0.0387)

    #print("SOAP pars:",fmin,degfree, np.mean(datah), np.mean(datal))
    lookup_1 = str(lookup_dir) + "/log_signoiseline_1det_{}degfree_{}_{}_{}.pkl".format(degfree,4.0,10.0,0.4)
    lookup_2 = str(lookup_dir) + "/log_signoiseline_2det_{}degfree_{}_{}_{}.pkl".format(degfree,4.0,10.0,0.4)

    # Gaussian lookup parmaeters
    #lookup_1 = str(lookup_dir) + "/signoiseline_1det_{}_{}_{}.txt".format(2.1,1.0,0.0)
    #lookup_2 = str(lookup_dir) + "/signoiseline_2det_{}_{}_{}.txt".format(2.1,1.0,0.0)

    # set the transition matrix as likely to be in the same bin in each detector and slightyl more probable to go straight (mostly so it goest straight through gaps)
    #tr = soap.tools.transition_matrix_2d_5(1.0000001,1.000001,1e400,1e400,log=True)
    tr = soap.tools.transition_matrix_2d(1.00000001,1e400,1e400)


    if "stats" in save_options:
        # run two detector search using the line aware statistic
        viterbi = soap.two_detector(tr,datah,datal,lookup_table_2det=lookup_2,lookup_table_1det=lookup_1)

    roll_track_sub = viterbi.vit_track - np.roll(viterbi.vit_track, 1)
    if np.any(np.abs(roll_track_sub[3:-3]) > 3):
        print("diff of track >1")
        print(viterbi.vit_track)
        sys.exit()
        

    # summed stat
    if "sum_stats" in save_options:
        sum_viterbi = soap.two_detector(tr,datah,datal)
    
    if inj_track is not None and "diffs" in save_options:
        diffs = np.array(viterbi.vit_track1) - np.array(inj_track)[:len(viterbi.vit_track1)]

    # get the normalised viterbi feather plots
    vit_data = viterbi.vitmap

    # get the normalised SFT power along the optimum track
    if "powers" in save_options:
        powers = soap.tools.track_power(viterbi.vit_track1,datah),soap.tools.track_power(viterbi.vit_track2,datal)

    #create the figure showing SFTs and tracks
    if "plots" in save_options:
        plotfig = make_figure(fmin,fmax,epochs[0],epochs[-1],datah,datal,[viterbi.vit_track1,viterbi.vit_track2],powers,vit_data,1800.,plot_track=True)

    # create the appropriate save directories
    """
    for key,j in outdirs.items():
        if not os.path.isdir(j):
            os.makedirs(j)
    """
    # if a size is not set then set the size of output image to 1/3 length in time and 1/2 in frequency
    if size is None:
        size = (vit_data.shape[0] / 3, vit_data.shape[1] / 2)
        
    # resize the normalised SFTs and the viterbi maps using skimages resize function which interpolates (also option for maxpooling)
    if resize_image:
            
        # viterbi map resize
        vit_resize      = resize(vit_data, size, anti_aliasing=True)
        #vit_resize      = skimage.measure.block_reduce(vit_data, (3,2), np.max)
        vit_resize_norm = vit_resize#/vit_resize.max()

        # H1 sfts resize
        deth_resize      = resize(datah, size, anti_aliasing=True)
        #deth_resize      = skimage.measure.block_reduce(datah, (3,2), np.max)
        deth_resize_norm = deth_resize#/deth_resize.max()

        # L1 sfts resize
        detl_resize      = resize(datal, size, anti_aliasing=True)
        #detl_resize      = skimage.measure.block_reduce(datal, (3,2), np.max)
        detl_resize_norm = detl_resize#/detl_resize.max()
    
    else:
        # in this case dont resize the image atall
        vit_resize_norm  = vit_data
        deth_resize_norm = datah
        detl_resize_norm = datal
    
    # if images from same data set are not the same size the raise error and exit
    if np.shape(vit_resize_norm) != np.shape(deth_resize_norm) or np.shape(vit_resize_norm) != np.shape(detl_resize_norm):
        print("Wrong Sizes")
        sys.exit()

    if totshape is None:
        totshape = (np.shape(vit_resize_norm), np.shape(vit_data))
    else:
        if totshape[1] != np.shape(vit_data):
            print("TOTALSHAPE DIFFERENT",totshape[1],np.shape(vit_data))
        if totshape[0] != np.shape(vit_resize_norm):
            print("RESIZESHAPE DIFFERENT",totshape[0],np.shape(vit_resize_norm))
            raise Exception("Shapes different between runs")


    return_dict = {}
    if "vit_imgs" in save_options:
        return_dict["vit_imgs"] = vit_resize_norm
        
    if "stats" in save_options:
        return_dict["stats"] = viterbi.max_end_prob
            
    if "sum_stats" in save_options:
        return_dict["sum_stats"] = sum_viterbi.max_end_prob
            
    if "H_imgs" in save_options:
        return_dict["H_imgs"] = deth_resize_norm
    if "L_imgs" in save_options:
        return_dict["L_imgs"] = detl_resize_norm

    if "paths" in save_options:
        return_dict["paths"] = [epochs,viterbi.vit_track]
    if "powers" in save_options:
        return_dict["powers"] = [epochs,powers[0],powers[1]]

    # save parametrs of injections if there are injections
    if pars is not None:
        return_dict["pars"] = pars

    return return_dict

def make_figure(flow,fhigh,tstart,tend,sum_obs1,sum_obs2,tracks,powers,vit_m,tsft,plot_track=True):
    fig, ax = plt.subplots(nrows=4,figsize = (14,16))
    im1 = ax[0].imshow(sum_obs1.T,aspect='auto',origin='lower',extent=[tstart,tend,flow,fhigh],label = "H1",cmap = plt.get_cmap("Greys"))
    im2 = ax[1].imshow(sum_obs2.T,aspect='auto',origin='lower',extent=[tstart,tend,flow,fhigh],label='L1',cmap = plt.get_cmap("Greys"))
    vmin,vmax = 0,1
    vim = ax[3].imshow(vit_m.T,aspect='auto',origin='lower',extent=[tstart,tend,flow,fhigh],cmap = plt.get_cmap('Greys'),label = "V")

    if plot_track:
        ax[0].plot(np.linspace(tstart,tend,len(tracks[0])),tracks[0]/tsft + flow,'C0',label = "Viterbi track")
        ax[1].plot(np.linspace(tstart,tend,len(tracks[1])),tracks[1]/tsft + flow,'C2',label = "Viterbi track")
    else:
        pass

    ax[2].plot(np.linspace(tstart,tend,len(powers[0])),powers[0],'C0',label = "H1")
    ax[2].plot(np.linspace(tstart,tend,len(powers[1])),powers[1],'C2',label = "L1")

    ax[0].set_xticklabels([])
    ax[1].set_xticklabels([])
    ax[3].set_xlabel("Time [s]",fontsize=20)
    leg0 = ax[0].legend(fancybox=True,fontsize=19,loc = 'upper right',borderpad = 0.2)
    leg1 = ax[1].legend(fancybox=True,fontsize=19,loc = 'upper right',borderpad = 0.2)
    leg2 = ax[2].legend(fancybox=True,fontsize=19,loc = 'upper right',borderpad = 0.2)
    leg3 = ax[3].legend(fancybox=True,fontsize=19,loc = 'upper right',borderpad = 0.2)

    ax[0].text(0.02, 0.85, 'H1',
               verticalalignment='bottom', horizontalalignment='left',
               transform=ax[0].transAxes,
               color='C0', fontsize=20,bbox={'facecolor':'white', 'alpha':0.5, 'pad':3,'boxstyle':'round,pad=0.3','ec':'none'})
    ax[1].text(0.02, 0.85, 'L1',
               verticalalignment='bottom', horizontalalignment='left',
               transform=ax[1].transAxes,
               color='C2', fontsize=20,bbox={'facecolor':'white', 'alpha':0.5, 'pad':3,'boxstyle':'round,pad=0.3','ec':'none'})
    ax[2].text(0.02, 0.85, 'Viterbi',
               verticalalignment='bottom', horizontalalignment='left',
               transform=ax[2].transAxes,
               color='black', fontsize=20,bbox={'facecolor':'white', 'alpha':0.5, 'pad':3,'boxstyle':'round,pad=0.3','ec':'none'})
    #divider = make_axes_locatable(ax[2])                                                                                                                                                      
    #cax = divider.append_axes("right", size="2%", pad=0.05,)                                                                                                                                  
    cax = fig.add_axes([0.91, 0.11, 0.01, 0.19])
    cbar = plt.colorbar(vim, cax=cax)
    cax1 = fig.add_axes([0.91, 0.69, 0.01, 0.19])
    cbar1 = plt.colorbar(im1, cax=cax1)
    cax2 = fig.add_axes([0.91, 0.496, 0.01, 0.19])
    cbar2 = plt.colorbar(im2, cax=cax2)

    cbar.ax.tick_params(labelsize=16)
    cbar1.ax.tick_params(labelsize=16)
    cbar2.ax.tick_params(labelsize=16)
    cbar.set_label(label=r"$\propto \log\left[O_{\rm{SGL}}\right]$", fontsize = 20)
    cbar1.set_label(label=r"SFT power", fontsize = 20)
    cbar2.set_label(label=r"SFT power", fontsize = 20)
    ax[0].tick_params(axis='both', which='major', labelsize=17)
    ax[1].tick_params(axis='both', which='major', labelsize=17)
    ax[2].tick_params(axis='both', which='major', labelsize=17)
    ax[3].tick_params(axis='both', which='major', labelsize=17)
    ax[0].yaxis.set_ticks(np.arange(flow+0.01, fhigh-0.01, 0.02))
    ax[1].yaxis.set_ticks(np.arange(flow+0.01, fhigh-0.01, 0.02))
    #ax[2].yaxis.set_ticks(np.arange(flow+0.01, fhigh-0.01, 0.02))                                                                                                                             
    ax[3].yaxis.set_ticks(np.arange(flow+0.01, fhigh-0.01, 0.02))
    ax[0].set_ylim([flow,fhigh])
    ax[1].set_ylim([flow,fhigh])
    #ax[2].set_ylim([flow,fhigh])                                                                                                                                                              
    ax[3].set_ylim([flow,fhigh])
    ax[0].set_xlim([tstart,tend])
    ax[1].set_xlim([tstart,tend])
    ax[2].set_xlim([tstart,tend])
    ax[3].set_xlim([tstart,tend])
    ax[0].set_ylabel("Frequency [Hz]",fontsize=20)
    ax[1].set_ylabel("Frequency [Hz]",fontsize=20)
    ax[2].set_ylabel("Normalised \n SFT Power",fontsize=20)
    ax[3].set_ylabel("Frequency [Hz]",fontsize=20)
    ax[0].get_yaxis().get_major_formatter().set_useOffset(False)
    ax[1].get_yaxis().get_major_formatter().set_useOffset(False)
    ax[3].get_yaxis().get_major_formatter().set_useOffset(False)
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


            
def sum_sft(data,nsfts=48):

    data_av1 = []
    fraction = []
    for i in np.linspace(0,len(data)-nsfts-1,len(data)-nsfts)[::nsfts]:
        end = i+nsfts
        chunk = data[int(i):int(end)]
        av = np.nansum(chunk.T,axis=1)
        data_av1.append(av)
                        
    return np.array(data_av1)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def main():
    if len(sys.argv) < 2 or sys.argv[1] == '-h':
        print(sys.argv)
        sys.exit(0)
    else:
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-v', '--verbose', help='display status', action='store_true')
        parser.add_argument("-c", "--config-file", help="config file contatining parameters")
        parser.add_argument('-l', '--band-min', help='lower frequency of band', type=float, required=True)
        parser.add_argument('-u', '--band-max', help='upper frequency of band', type=float, required=True)
        parser.add_argument('-w', '--band-width', help='width of indidual band', type=float, required=True)
        parser.add_argument('-dt', '--data-type', help='train, validation or test data', type=str, default = "train")
        parser.add_argument('-rt', '--run-type', help='run type, gauss: return gaussian noise, else:O1,O2,O3,(O4)',required=True)
        parser.add_argument('-sr', '--search', help='generate search data, if true generates data with no injections', type=str2bool, const=True, default = False,nargs='?')
        parser.add_argument('-nn', '--no-noise', help='do not regenerate noise bands', type=str2bool, const=True, default = False,nargs='?')
        parser.add_argument('-so', '--save-options', help='files to save', type=str, default = "vit_img stats pars H_imgs L_imgs vit_imgs",nargs="?")
        parser.add_argument('-np', '--nperband', help='numnber of examples per band gaussian noise', type=int, default = 20,nargs="?")

        try:                                                     
            args = parser.parse_args()  
        except:  
            sys.exit(1)

        from soapcw.soap_config_parser import SOAPConfig

        if args.config_file is not None:
            config = SOAPConfig(args.config_file)
        else:
            config = None


        if args.data_type == "test":
            save_dir = os.path.join(config["general"]["save_dir"], "test")
            if args.run_type != "gaussian":
                generate_test_data.loop_band_test_data(args.load_dir,save_dir,args.band_min,args.band_max,args.band_width,args.resize_image, num_repeat = args.nperband)
            elif args.run_type == "gaussian":
                gaussian_train.loop_band_train_augment(loadpath=args.load_dir,path=save_dir,bandmin=args.band_min,bandmax=args.band_max, band_width=args.band_width,resize_image=args.resize_image, nonoise=args.no_noise,snr_min=args.snr_min,snr_max=args.snr_max,nperband=1,test=True,save_options=args.save_options.split(" "))

        elif args.data_type in ["train", "validation"]:
            save_dir = os.path.join(config["general"]["save_dir"], args.data_type)

            print("runtype", args.run_type)
            if args.run_type == "gaussian" or args.run_type == "gauss":
                generate_gaussian_train_data.loop_band_train_augment(
                    config, 
                    path=save_dir,
                    bandmin=args.band_min,
                    bandmax=args.band_max, 
                    band_width=args.band_width, 
                    nonoise=args.no_noise,
                    snr_min=config["data"]["snrmin"],
                    snr_max=config["data"]["snrmax"],
                    nperband=args.nperband,
                    test=False,
                    save_options=config["data"]["save_options"]
                    )
            else:
                generate_train_data.loop_band_train_augment(
                    config, 
                    save_dir,
                    args.band_min,
                    args.band_max,
                    args.band_width,
                    config["data"]["resize_image"], 
                    gen_noise_only=config["data"]["gen_noise_only"],
                    snr_min=config["data"]["snrmin"],
                    snr_max=config["data"]["snrmax"],
                    save_options=config["data"]["save_options"]
                    )

        """
        elif args.mdc == True:
            mdc_test.loop_band_mdc(args.save_dir,args.load_dir,args.noise_dir,args.band_min,args.band_max,args.band_width,args.resize_image)
        elif args.search == True:
            generate_test_data.loop_band_test_data(args.load_dir,args.save_dir,args.band_min,args.band_max,args.band_width,args.resize_image,search = True)
        """

if __name__ == '__main__':       
    main()

