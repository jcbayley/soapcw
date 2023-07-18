from soapcw.cnn import cnn_data_gen
import numpy as np
from soapcw import cw
import time
import os
import datetime
import copy
import pickle

def loop_band_test_data(loadpath,path,bandmin,bandmax,band_width,resize_image=True,search=False, num_repeat = 1):

    if 40 <= bandmin <= 500: 
        stride = 1
        degfree = 96
        plotcut = stride*180
        width = 0.1
        brange = "band_40_500"
    elif 500 < bandmin <= 1000:
        stride = 2
        degfree = int(stride*96)
        plotcut = stride*180
        width = 0.2
        brange = "band_500_1000"
    elif 1000 < bandmin <= 1500:
        stride = 3
        degfree = int(stride*96)
        plotcut = stride*180
        width = 0.3
        brange = "band_1000_1500"
    elif 1500 < bandmin <= 2000:
        stride = 4
        degfree = int(stride*96)
        plotcut = stride*180
        width = 0.4
        brange = "band_1500_2000"

        
    bl,be = bandmin,bandmax

    tmin, tmax = None,None#931035615.,931035615.+1800*100

    vetolist = list(np.arange(bandmin,bandmax))
    
    hname,lname = cnn_data_gen.find_sft_file(loadpath,bandmin,bandmax)

    sfts = cw.LoadSFT("{};{}".format(hname,lname),norm=True,filled=True,vetolist = vetolist,save_rngmed=True)

    bin_width = 1./float(1800.)
    nbins = len(getattr(sfts,"H1").frequencies)
    sub_bins = np.round(width/bin_width).astype(int) 

    # set as list for fill 2 Hz band
    start_time = time.time()
    range_bands = np.round(np.arange(bandmin,bandmax-width,0.5*width),2)
    #range_index = np.arange(0,(bandmax-width-bandmin)*1800,90).astype(int)
    range_index = np.round(np.arange(0,nbins,width/(2*bin_width))).astype(int)[:-1]

    # set which bands are odd and even with and ondex of 1 or 0
    odd_even = np.zeros(len(range_bands))
    odd_even[1::2] = 1

    if not os.path.isdir(path):
        os.makedirs(path)
    erange = 15

    snrstart,snrend = 20,200

    noise_out_snr = "snr_0.0_0.0"
    signal_out_snr = "snr_{}_{}".format(snrstart,snrend)
    #save_options = ["stats","paths", "pars", "powers"]
    save_options = ["stats","paths", "pars", "powers","vit_imgs","H_imgs","L_imgs"]
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



    for i in range(len(range_bands)):
            
        k = range_bands[i]

        skip_band = False
        hardware_list = [(108.8,108.9),(147.45,147.55),(192.7,192.75)]
        for lin in hardware_list:
            if k > lin[0] and k < lin[1]:
                skip_band = True

        if skip_band:
            continue

        ms = int(range_index[i])
        me = int(ms+sub_bins)
        
        if odd_even[i] == 1:
            split = "odd"
        else:
            split = "even"

        #outpath = os.path.join(path,split)                     
            
        av_sh = (2/1800.)*np.nanmedian([sfts.H1.rng_med[:,ms:me],sfts.L1.rng_med[:,ms:me]])
        
        datah = copy.deepcopy(sfts.H1.norm_sft_power[:,ms:me])
        datal = copy.deepcopy(sfts.L1.norm_sft_power[:,ms:me])
        print("SFTLEN. H1:{}, L1:{}".format(sum(datah[:,0] != degfree),sum(datal[:,0] != degfree)))
        
        #print(k,k+width, snr )
        for nr in range(num_repeat):

            snr = int(np.random.rand(1)*(snrend-snrstart) + snrstart)
            print("SNR: ", snr)
            rnd = np.random.rand(1)

            if rnd > 0.5 or search:
                # if snr of 0 give downsamp data otherwise give sft for injections
                snr = 0
            
            outputs = run_and_inj_test(copy.copy(datah),copy.copy(datal),k,width,resize_image=resize_image,snr=snr,av_sh = av_sh,sfts=sfts, stride=stride, degfree=degfree, brange = brange)

            for key in save_options:
                if split == "even":
                    if snr == 0:
                        even_noise_save_outs[key].append(outputs[key])
                    else:
                        even_signal_save_outs[key].append(outputs[key])
                elif split == "odd":
                    if snr == 0:
                        odd_noise_save_outs[key].append(outputs[key])
                    else:
                        odd_signal_save_outs[key].append(outputs[key])

            
        del datah,datal


    for key in save_options:
        even_noise_filenames[key] = os.path.join(*[path,"even",key,brange,noise_out_snr,"freq_{}_{}.pkl".format(bandmin, len(even_noise_save_outs[key]))])
        even_signal_filenames[key] = os.path.join(*[path,"even",key,brange,signal_out_snr,"freq_{}_{}.pkl".format(bandmin, len(even_signal_save_outs[key]))])
        odd_noise_filenames[key] = os.path.join(*[path,"odd",key,brange,noise_out_snr,"freq_{}_{}.pkl".format(bandmin, len(odd_noise_save_outs[key]))])
        odd_signal_filenames[key] = os.path.join(*[path,"odd",key,brange,signal_out_snr,"freq_{}_{}.pkl".format(bandmin, len(odd_signal_save_outs[key]))])
    
        for fname in [even_noise_filenames[key], even_signal_filenames[key], odd_noise_filenames[key], odd_signal_filenames[key]]:
            if not os.path.isdir(os.path.dirname(fname)):
                os.makedirs(os.path.dirname(fname))

        print(even_noise_filenames[key])
        with open(even_noise_filenames[key],"wb") as f:
            pickle.dump(even_noise_save_outs[key], f)
        with open(odd_noise_filenames[key],"wb") as f:
            pickle.dump(odd_noise_save_outs[key], f)
        with open(even_signal_filenames[key],"wb") as f:
            pickle.dump(even_signal_save_outs[key], f)
        with open(odd_signal_filenames[key],"wb") as f:
            pickle.dump(odd_signal_save_outs[key], f)



    end_time = time.time()
    date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    with open(os.path.join(path,"timing.txt"),"a") as f:
        f.write("time-for-gen    {}    {} \n".format(end_time-start_time,date))

        
        
def run_and_inj_test(datah,datal,fmin,width,resize_image=False,snr = 0, av_sh = None,sfts=None, stride = 1, degfree = 96, brange = ""):
    
    
    # save noise realisation 1
    if snr == 0:
        nsft, tstart, tsft, flow, fhigh = len(sfts.H1.epochs), sfts.H1.epochs[0], 1800., fmin,fmin+width
        sig = cw.GenerateSignal()
        sig.alpha = 0
        sig.delta = 0
        sig.cosi = 0
        sig.phi0 = 0
        sig.psi = 0
        sig.f = [100,-1e-16]
        sig.snr = 0
        sig.fmin = flow
        sig.fmax = flow + width
        sig.tref = tstart

        sig.earth_ephem = "/home/joseph.bayley/repositories/lalsuite/lalpulsar/lib/earth00-40-DE430.dat.gz"
        sig.sun_ephem = "/home/joseph.bayley/repositories/lalsuite/lalpulsar/lib/sun00-40-DE430.dat.gz"
        data = sig.get_spectrogram(epochs = sfts.H1.epochs, nsft = len(datah),tref=sfts.H1.epochs[0],tsft=tsft,fmin=flow,fmax=fhigh,snr=0,noise_spect={"H1":datah,"L1":datal})
        #print(data.h0)
        data.sum_sfts()
        data.downsamp_frequency(stride=stride,data_type="summed_norm_sft_power",remove_original=False)
        H1path,L1path = (flow + width/2.,flow + width/2.),(flow + width/2.,flow + width/2.)
        save_pars = {"fmin":sig.fmin,"fmax":sig.fmax, "alpha":None, "delta":None, "cosi":None, "psi":None, "phi0":None, "f0":None, "fdot":None, "snr":0, "depth":np.inf, "tref":sig.tref, "av_sh": av_sh, "h0" : 0}

        noise_outputs = cnn_data_gen.return_outputs(out_snr="./",datah=data.H1.downsamp_summed_norm_sft_power,datal=data.L1.downsamp_summed_norm_sft_power,fmin=fmin,fmax=fmin+width,snr=0,depth=0,h0=0,resize_image=resize_image,test=True,epochs=data.H1.summed_epochs,minmaxpaths=(H1path,L1path),degfree=degfree, brange = brange, pars = save_pars)
        return noise_outputs
    else:
        # inject signal into band
        
        sig = cw.GenerateSignal()
        nsft, tstart, tsft, flow, fhigh = len(sfts.H1.epochs), sfts.H1.epochs[0], 1800., fmin,fmin+width
        
        #sigfreq = np.random.uniform(0,1)*(width)*0.5 + flow + (width)*0.2
        
        
        # freq, spindown^n, alpha, sin(delta), phi0, psi, cosi
        params_low =  [fmin + 0.03,      -9 ,0      ,-1,0      ,0      ,-1.0]
        params_high = [fmin + width-0.03,-16,2*np.pi,1 ,2*np.pi,np.pi/2,1.0]
        data_list_signal = []
        data_list_noise = []
        nsft = int(nsft)
        param_list = ["f","fd","Alpha","Delta","phi0","psi","cosi"]
        pars = {}
        for j in np.linspace(0,len(param_list)-1,len(param_list)).astype(int):
            pars[param_list[j]] = params_low[j] + np.random.uniform(0,1)*(params_high[j] - params_low[j])
            
        sig.alpha = pars["Alpha"]
        sig.delta = np.arcsin(pars["Delta"])
        sig.cosi = pars["cosi"]
        sig.phi0 = pars["phi0"]
        sig.psi = pars["psi"]
        sig.f = [pars["f"],-np.random.uniform(0,1)*(10**pars["fd"])]
        sig.earth_ephem = "/home/joseph.bayley/repositories/lalsuite/lalpulsar/lib/earth00-40-DE430.dat.gz"
        sig.sun_ephem = "/home/joseph.bayley/repositories/lalsuite/lalpulsar/lib/sun00-40-DE430.dat.gz"
        sig.snr = snr
        sig.fmin = flow
        sig.fmax = fhigh
        sig.tref = tstart



        #print(tstart, len(datah), tstart, tsft, flow,fhigh,snr,np.mean(datah),np.mean(datal))
        #print("FREQS", flow, fhigh)
        data = sig.get_spectrogram(epochs = sfts.H1.epochs, nsft = len(datah),tref=sfts.H1.epochs[0],tsft=tsft,fmin=flow,fmax=fhigh,snr=snr,noise_spect={"H1":datah,"L1":datal})
        #print(data.h0)
        data.sum_sfts()
        data.downsamp_frequency(stride=stride,data_type="summed_norm_sft_power",remove_original=False)

        H1pulsar_path = sig.get_pulsar_path(sfts.H1.epochs,sig.edat,"H1")
        L1pulsar_path = sig.get_pulsar_path(sfts.L1.epochs,sig.edat,"L1")

        H1paths = min(H1pulsar_path),max(H1pulsar_path)
        L1paths = min(L1pulsar_path),max(L1pulsar_path)

        depth = data.depth
        h0 = np.sqrt(av_sh)/depth
        save_pars = {"fmin":sig.fmin,"fmax":sig.fmax, "alpha":sig.alpha, "delta":sig.delta, "cosi":sig.cosi, "psi":sig.psi, "phi0":sig.phi0, "f0":sig.f[0], "fdot":sig.f[1], "snr":snr, "depth":depth, "tref":sig.tref, "av_sh": av_sh, "h0" : h0}
        

        sig_outputs = cnn_data_gen.return_outputs(out_snr="./",datah=data.H1.downsamp_summed_norm_sft_power,datal=data.L1.downsamp_summed_norm_sft_power,fmin=fmin,fmax=fmin+width,snr=snr, h0=h0,depth=depth,resize_image=resize_image,pars=save_pars,test=True,epochs=data.H1.summed_epochs,minmaxpaths=(H1paths,L1paths),degfree=degfree, brange = brange)
        
        return sig_outputs
        
