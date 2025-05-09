from soapcw.cnn import cnn_data_gen
import numpy as np
from soapcw import cw
import time
import os
import datetime
import copy
import pickle
import h5py
import logging
import sys

logging.basicConfig(level=logging.INFO, 
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S') 

def roll_in_time(indata,numbins = 100,sh=None, rng_med=None, use_complex=True):
    """Roll the data in time by a given number of bins, preserving the location of the gaps

    Args:
        indata (2d array): input data
        numbins (int, optional): number of time bins to roll by. Defaults to 100.
        sh (_type_, optional): the noise floor. Defaults to None.

    Returns:
        _type_: _description_
    """
    sfts_index = [i for i,j in enumerate(np.mean(indata,axis=1) != 2) if j]
    
    sfts_vals_rolled = np.roll(indata[sfts_index].T,numbins).T
    
    if sh is not None:
        sh_rolled = np.roll(sh[sfts_index].T,numbins).T
        sh_new = np.ones(len(sh))*np.nan
    
    if rng_med is not None:
        rng_med_rolled = np.roll(rng_med[sfts_index].T,numbins).T
        rng_med_new = np.ones(np.shape(rng_med))*np.nan
    
    if use_complex:
        data = np.ones(np.shape(indata)).astype("complex")*2
    else:
        data = np.ones(np.shape(indata))*2

    for i,j in enumerate(sfts_index):
        data[j,:] = sfts_vals_rolled[i]
        if sh is not None:
            sh_new[j] = sh_rolled[i]
        if rng_med is not None:
            rng_med_new[j] = rng_med_rolled[i]
        
    if sh is not None and rng_med is not None:
        return data, sh_new, rng_med_new
    else:
        return data
        
def flip_data_time(indata,sh = None, rng_med=None, use_complex=True):
    """Flip the data in time preserving the location of the gaps

    Args:
        indata (2d array): input data
        sh (_type_, optional): noisefloor. Defaults to None.

    Returns:
        _type_: _description_
    """
    sfts_index = [i for i,j in enumerate(np.mean(indata,axis=1) != 2) if j]
    
    sfts_vals_reverse = indata[sfts_index][::-1]
    
    if use_complex:
        data = np.ones(np.shape(indata)).astype("complex")*2
    else:
        data = np.ones(np.shape(indata))*2

    if sh is not None:
        sh_reverse = sh[sfts_index][::-1]
        sh_new = np.ones(len(sh))*np.nan

    if rng_med is not None:
        rng_med_reverse = rng_med[sfts_index][::-1]
        rng_med_new = np.ones(np.shape(rng_med))*np.nan
    
    for i,j in enumerate(sfts_index):
        data[j,:] = sfts_vals_reverse[i]
        if sh is not None:
            sh_new[j] = sh_reverse[i]
        if rng_med is not None:
            rng_med_new[j] = rng_med_reverse[i]
        
    if sh is not None and rng_med is not None:
        return data, sh_new, rng_med_new
    else:
        return data

def flip_data_freq(indata):
    """flip the data in frequency

    Args:
        indata (2d array): input spectrogram

    Returns:
        _type_: _description_
    """
    data = indata.T[::-1].T
        
    return data



def loop_band_train_augment(
    config,
    path,
    bandmin,
    bandmax,
    band_width,
    resize_image=True,
    gen_noise_only_bands=False,
    save_options = None):
    """Loop over the data in load directory and augment to create training data

    Args:
        path (_type_): _description_
        bandmin (_type_): _description_
        bandmax (_type_): _description_
        band_width (_type_): _description_
        resize_image (bool, optional): _description_. Defaults to True.
        gen_noise_only_bands (bool, optional): _description_. Defaults to False.
        snr_min (int, optional): _description_. Defaults to 50.
        snr_max (int, optional): _description_. Defaults to 150.
        save_options (_type_, optional): _description_. Defaults to None.
    """

    start_time = time.time()
    tsft = 1800.

    for i, (minband, maxband) in enumerate(zip(config["data"]["band_starts"], config["data"]["band_ends"])):
        if minband <= bandmin < maxband:
            stride = config["data"]["strides"][i]
            degfree = int(2 * config["data"]["n_summed_sfts"] * stride)
            width = config["data"]["band_widths"][i]
            nbin = int(width*tsft)
            plotcut = nbin
            brange = f"band_{int(minband):.1f}_{int(maxband):.1f}"
            break
        else:
            continue

    #print(f"bandmin: {bandmin}, bandmax: {bandmax}, width: {width}, stride: {stride}, degfree: {degfree}, brange: {brange}")

    # in 2Hz band set frequencies and indicies with 0.1 Hz bands not overlapping
    range_bands = np.round(np.arange(bandmin,bandmax-width,width),1)
    range_index = np.arange(0,(bandmax-width-bandmin)*int(tsft),nbin).astype(int)    
    
    # set which bands are odd and even with and ondex of 1 or 0
    odd_even = np.zeros(len(range_bands))
    odd_even[1::2] = 1

    bl,be = bandmin,bandmax

    if config["data"]["start_time"] is not None:
        tmin = config["data"]["start_time"]
    else:
        tmin = None
    if config["data"]["end_time"] is not None:
        tmax = config["data"]["end_time"]
    else:
        tmax = None

    #tmin, tmax = None,None#931035615.,931035615.+1800*100

    # set integer Hz as vetos, this will cut out a few bins surrounding
    vetolist = list(np.arange(bandmin,bandmax))

    # find the appropriate sft files
    hname,lname = cnn_data_gen.find_sft_file(config["narrowband"]["narrowband_sft_dir"],bandmin,bandmax)
    
    # load in the narrowbanded sfts, normalise them, fill in the gaps and save the running median
    logging.info(f"Loading SFTs ... {hname} {lname} in times {tmin} - {tmax}")
    sfts = cw.LoadSFT("{};{}".format(hname,lname),norm=True,filled=True,vetolist = vetolist,save_rngmed=True, tmin=tmin,tmax=tmax, remove_sft=False)


    if config["lookuptable"]["lookup_type"] == "amplitude":
        amp="amp"
    else:
        amp=""

    # make appropriate directoes
    if not os.path.isdir(path):
        os.makedirs(path)

    erange = 15

    noise_out_snr = "snr_0.0_0.0"
    signal_out_snr = f'snr_{config["cnn_data"]["snrmin"]:.1f}_{config["cnn_data"]["snrmax"]:.1f}'

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

    skip_list = []#config["data"]["hardware_injections"]

    even_noise_filenames = os.path.join(*[path,"even",brange,noise_out_snr,f"freq_{bandmin:.1f}_{bandmax:.1f}_{len(config['cnn_data']['shift_bins'])*len(range_bands)}.hdf5"])
    even_signal_filenames = os.path.join(*[path,"even",brange,signal_out_snr,f"freq_{bandmin:.1f}_{bandmax:.1f}_{len(config['cnn_data']['shift_bins'])*len(range_bands)}.hdf5"])
    odd_noise_filenames = os.path.join(*[path,"odd",brange,noise_out_snr,f"freq_{bandmin:.1f}_{bandmax:.1f}_{len(config['cnn_data']['shift_bins'])*len(range_bands)}.hdf5"])
    odd_signal_filenames = os.path.join(*[path,"odd",brange,signal_out_snr,f"freq_{bandmin:.1f}_{bandmax:.1f}_{len(config['cnn_data']['shift_bins'])*len(range_bands)}.hdf5"])

    if os.path.isfile(even_noise_filenames) and os.path.isfile(even_signal_filenames) and os.path.isfile(odd_noise_filenames) and os.path.isfile(odd_signal_filenames):
        logging.info("!!!!!! Data already exists, skipping and exiting... !!!!!!!!!!!")
        sys.exit(1)

    logging.info(f"iterating over Nbands: {len(range_bands)}")
    # main loop over all sub bands
    for i in range(len(range_bands)):
        
        # set initial band frequency
        k = range_bands[i]
        
        # band width is 0.1 Hz
        #width = 0.1

        # choose sub bands to skip, this list is hardware injections
        skip_band = False
        for lin in skip_list:
            if k > lin[0] and k < lin[1]:
                skip_band = True
                break

        if skip_band:
            continue

        #snr_start,snr_end = 80,150
        ms = int(range_index[i])
        me = int(ms+nbin)
        # put odd bands into one folder and even bands into another folder
        if odd_even[i] == 1:
            #outpath = os.path.join(path,"odd")
            split = "odd"
        else:
            #outpath = os.path.join(path,"even")
            split = "even"
            
        # shift data up and down in frequency by n bins  
        shifts = [ns_bin for ns_bin in config["cnn_data"]["shift_bins"]]  
        #shifts = [ms,ms + 30*stride,ms - 30*stride,ms + 60*stride, ms - 60*stride]

        # loop over all of the augmentation shifts
        for shft in shifts:
            #set band as index plut 180 bins i.e. 0.1 Hz
            
            cts = ms + shft
            cte = int(cts + nbin)
            #print(split, ms, shft, cts, cte, sfts.H1.nbins, k+shft/tsft, k, shft/tsft)
            if cts < 0 or cte >= sfts.H1.nbins:
                logging.info("Error index outside of bin range ................")
                continue
            
            # get the median of the running medians as an estimate of the noise floow in this band.
            av_sh = [(2/tsft)*np.nanmedian(sfts.H1.rng_med[:,cts:cte],axis=1),(2/tsft)*np.nanmedian(sfts.L1.rng_med[:,cts:cte],axis=1)]
                
            # set lower frequency
            flow = k + shft / tsft
            
            # amke copy of data so dont overwrite sft
            datah = copy.deepcopy(sfts.H1.sft[:,cts:cte])
            datal = copy.deepcopy(sfts.L1.sft[:,cts:cte])
            rng_med = {"H1":copy.deepcopy(sfts.H1.rng_med[:,cts:cte]), 
                        "L1":copy.deepcopy(sfts.L1.rng_med[:,cts:cte])}

            
            # run injections function with new data, this saves noise iamge and injects signal and saves signal images
            noise_outs1, signal_outs1 = run_and_inj(
                config,
                datah,datal,flow,width,resize_image=resize_image,av_sh=av_sh,
                gen_noise_only_bands=gen_noise_only_bands,
                save_options=  save_options, stride = stride, 
                degfree = degfree, brange = brange, sfts=sfts,
                rng_med=rng_med
                )
  
            del datah, datal
            
            # flip data in time
        
            datah,shH,rngH = flip_data_time(
                copy.deepcopy(sfts.H1.sft[:,cts:cte]),
                sh=av_sh[0],
                rng_med=copy.deepcopy(sfts.H1.rng_med[:,cts:cte]))
            datal,shL,rngL = flip_data_time(
                copy.deepcopy(sfts.L1.sft[:,cts:cte]),
                sh=av_sh[1],
                rng_med=copy.deepcopy(sfts.L1.rng_med[:,cts:cte]))

         
            noise_outs2, signal_outs2 = run_and_inj(
                config,
                datah,datal,flow,width,resize_image=resize_image,av_sh=[shH,shL],
                gen_noise_only_bands=gen_noise_only_bands,
                save_options=save_options, stride = stride, 
                degfree = degfree, brange = brange,sfts=sfts,
                rng_med={"H1":rngH, "L1":rngL}
                )

            del datah,datal
        
            # flip data in freq
        
            datah = flip_data_freq(copy.deepcopy(sfts.H1.sft[:,cts:cte]))
            datal = flip_data_freq(copy.deepcopy(sfts.L1.sft[:,cts:cte]))
            rng_H = flip_data_freq(copy.deepcopy(sfts.H1.rng_med[:,cts:cte]))
            rng_L = flip_data_freq(copy.deepcopy(sfts.L1.rng_med[:,cts:cte]))

            noise_outs3, signal_outs3 = run_and_inj(
                config,
                datah,datal,flow,width,resize_image=resize_image,av_sh=av_sh,
                gen_noise_only_bands=gen_noise_only_bands,
                save_options=save_options, stride = stride, 
                degfree = degfree, brange = brange,sfts=sfts,
                rng_med={"H1":rng_H, "L1":rng_L}
                )

            del datah,datal
            
            # roll data in time by 100 bins
        
            datah,shH, rngH = roll_in_time(copy.deepcopy(sfts.H1.sft[:,cts:cte]),numbins=100,sh=av_sh[0],rng_med=copy.deepcopy(sfts.H1.rng_med[:,cts:cte]))
            datal,shL, rngL = roll_in_time(copy.deepcopy(sfts.L1.sft[:,cts:cte]),numbins=100,sh=av_sh[1],rng_med=copy.deepcopy(sfts.L1.rng_med[:,cts:cte]))
            tinit = 0
            noise_outs4, signal_outs4 = run_and_inj(
                config,
                datah,datal,flow,width,resize_image=resize_image,av_sh=[shH,shL],
                gen_noise_only_bands=gen_noise_only_bands,
                save_options=save_options, stride = stride, 
                degfree = degfree, brange = brange,sfts=sfts,
                rng_med={"H1":rngH, "L1":rngL}
                )
            
            del datah,datal
            
            for key in save_options:
                if split == "even":
                    even_noise_save_outs[key].extend([noise_outs1[key],noise_outs2[key],noise_outs3[key],noise_outs4[key] ])
                    even_signal_save_outs[key].extend([signal_outs1[key],signal_outs2[key],signal_outs3[key],signal_outs4[key]])
                elif split == "odd":
                    odd_noise_save_outs[key].extend([noise_outs1[key],noise_outs2[key],noise_outs3[key],noise_outs4[key] ])
                    odd_signal_save_outs[key].extend([signal_outs1[key],signal_outs2[key],signal_outs3[key],signal_outs4[key]])
                else:
                    raise Exception("split not even or odd")
            
        

    logging.info("saving data to file .......")



    for fname, temp_data in zip([even_noise_filenames, even_signal_filenames, odd_noise_filenames, odd_signal_filenames], [even_noise_save_outs, even_signal_save_outs, odd_noise_save_outs, odd_signal_save_outs]):
        if not os.path.isdir(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
       # print(fname, save_options)
        parkeys = ['fmin', 'fmax', 'width', 'av_sh', 'tref','snr', 'h0', 'depth', 'f', 'fd', 'alpha', 'sindelta', 'phi0', 'psi', 'cosi']
        with h5py.File(fname,"w") as f:
            for key in save_options:
                if key == "pars":
                    # pars are stored as a dictionary so have to be saved differently to hdf5
                    """
                    try:
                        parkeys = list(temp_data[key][0].keys())
                    except:
                        print(np.shape(temp_data))
                        print("temp_datakeys", temp_data[key][0])
                        raise Exception(f"temp data has no data: {fname}")
                    """
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

    """
    for key in save_options:
        even_noise_filenames[key] = os.path.join(*[path,"even",key,brange,noise_out_snr,"freq_{}_{}.pkl".format(bandmin, len(even_noise_save_outs[key]))])
        even_signal_filenames[key] = os.path.join(*[path,"even",key,brange,signal_out_snr,"freq_{}_{}.pkl".format(bandmin, len(even_signal_save_outs[key]))])
        odd_noise_filenames[key] = os.path.join(*[path,"odd",key,brange,noise_out_snr,"freq_{}_{}.pkl".format(bandmin, len(odd_noise_save_outs[key]))])
        odd_signal_filenames[key] = os.path.join(*[path,"odd",key,brange,signal_out_snr,"freq_{}_{}.pkl".format(bandmin, len(odd_signal_save_outs[key]))])
    
        for fname in [even_noise_filenames[key], even_signal_filenames[key], odd_noise_filenames[key], odd_signal_filenames[key]]:
            if not os.path.isdir(os.path.dirname(fname)):
                os.makedirs(os.path.dirname(fname))

        with open(even_noise_filenames[key],"wb") as f:
            pickle.dump(even_noise_save_outs[key], f)
        with open(odd_noise_filenames[key],"wb") as f:
            pickle.dump(odd_noise_save_outs[key], f)
        with open(even_signal_filenames[key],"wb") as f:
            pickle.dump(even_signal_save_outs[key], f)
        with open(odd_signal_filenames[key],"wb") as f:
            pickle.dump(odd_signal_save_outs[key], f)
    """

    end_time = time.time()
    date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    with open(os.path.join(path,"timing.txt"),"a") as f:
        f.write("time-for-gen    {}    {} \n".format(end_time-start_time,date))




def run_and_inj(config, datah,datal,fmin,width,resize_image=False,av_sh=None,gen_noise_only_bands=False, save_options = None, stride = 1,degfree = 96, brange = "", sfts = None, rng_med=None):
    """_summary_

    Args:
        datah (_type_): input hanford detector data
        datal (_type_): input livingston detector data
        fmin (_type_): minimum frequency of band [Hz]
        width (_type_): band width [Hz]
        resize_image (bool, optional): _description_. Defaults to False.
        snrrange (tuple, optional): _description_. Defaults to (50,150).
        av_sh (_type_, optional): _description_. Defaults to None.
        gen_noise_only_bands (bool, optional): _description_. Defaults to False.
        save_options (_type_, optional): _description_. Defaults to None.
        stride (int, optional): _description_. Defaults to 1.
        degfree (int, optional): _description_. Defaults to 96.
        brange (str, optional): _description_. Defaults to "".
        sfts (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if np.shape(datah) != np.shape(datal):
        print("shapes do not match", "fmin ", fmin, stride )
        print("HL shape: ", np.shape(datah), np.shape(datal))
        return 0

    if av_sh is not None:
        Sn = {"H1":av_sh[0],"L1":av_sh[1]}
    #rng_med = {"H1":None, "L1":None}
    # hardcoded start times as nsft
    nsft, tstart, tsft, flow, fhigh = sfts.H1.nsft, sfts.H1.epochs[0], 1800., fmin,fmin+width
    fmax = fmin + width
    # not gen_noise_only_bands means that we also generate the noise bands as well as the injected bands
    if gen_noise_only_bands: 
        # save noise realisation 1
        out_snr_noise = "snr_0.0_0.0"
        sig = cw.GenerateSignal()
        sig.alpha = 0
        sig.delta = 0
        sig.cosi = 0
        sig.phi0 = 0
        sig.psi = 0
        sig.f = [0,0]
        sig.earth_ephem = config["cnn_data"]["earth_ephemeredis"]
        sig.sun_ephem = config["cnn_data"]["sun_ephemeredis"]
        sig.snr = 0
        sig.fmin = fmin
        sig.fmax = fmax
        sig.tref = tstart
        
        Sn = {"H1":av_sh[0],"L1":av_sh[1]}
        # just gaussian noise
        #data = sig.get_spectrogram(
        #    tstart = tstart, nsft = nsft,tref=tstart,tsft=tsft,fmin=fmin,fmax=fmax,snr=0,dets= ["H1","L1"],Sn=Sn
        #    )
        #use real data
        data = sig.get_spectrogram(
            tstart = tstart, nsft = len(datah),tref=tstart,tsft=tsft,
            fmin=fmin,fmax=fmax,snr=0,noise_sft={"H1":datah,"L1":datal}, real_data=True,
            rng_med = rng_med, sftpaths={"H1":sfts.H1.sftpath,"L1":sfts.L1.sftpath}
            )

        data.sum_sfts()
        data.downsamp_frequency(stride=stride,data_type="summed_norm_sft_power",remove_original=False)

        noise_outputs = cnn_data_gen.return_outputs(
            config,
            out_snr=out_snr_noise,
            datah=copy.deepcopy(data.H1.downsamp_summed_norm_sft_power),
            datal=copy.deepcopy(data.L1.downsamp_summed_norm_sft_power), 
            fmin=fmin,fmax=fmax,snr=0,depth=0,h0=0, 
            resize_image=resize_image, 
            save_options = save_options, 
            epochs = data.H1.summed_epochs, 
            degfree = degfree
            )

        noise_outputs["pars"] = {"snr":0}
        noise_outputs["pars"]["tref"] = tstart
        noise_outputs["pars"]["snr"] = 0
        noise_outputs["pars"]["h0"] = 0
        noise_outputs["pars"]["depth"] = np.inf
        noise_outputs["pars"]["av_sh"] = np.nanmedian(av_sh)
        noise_outputs["pars"]["width"] = width
        noise_outputs["pars"]["fmin"] = data.H1.fmin
        noise_outputs["pars"]["fmax"] = data.H1.fmax
        param_list = ["f","fd","alpha","sindelta","phi0","psi","cosi"]
        for param in param_list:
            noise_outputs["pars"][param] = np.nan
    else:
        noise_outputs = None
    # inject signal into band

    sig = cw.GenerateSignal()
    
    snrstart,snrend = config["cnn_data"]["snrmin"], config["cnn_data"]["snrmax"]
                    
    snr = int(np.random.rand(1)*(snrend-snrstart) + snrstart)
        
    # places f0 in and around the frequency band and within the defined ranges for other parmaeters
    #params_low =  [fmin + 0.025,      -1e-9 ,0      ,-1, 0      ,0      ,-1.0]
    #params_high = [fmin + width-0.025, -1e-16,  2*np.pi, 1,2*np.pi,np.pi/2,1.0]
    """
    params_low =  [
        fmin + 0.025,      
        config["cnn_data"]["fdot_range"][0],
        config["cnn_data"]["alpha_range"][0],
        config["cnn_data"]["sindelta_range"][0], 
        config["cnn_data"]["phi0_range"][0],
        config["cnn_data"]["psi_range"][0],
        config["cnn_data"]["cosi_range"][0]]
    params_high = [
        fmin + width-0.025, 
        config["cnn_data"]["fdot_range"][1],
        config["cnn_data"]["alpha_range"][1],
        config["cnn_data"]["sindelta_range"][1], 
        config["cnn_data"]["phi0_range"][1],
        config["cnn_data"]["psi_range"][1],
        config["cnn_data"]["cosi_range"][1]]
    
    """
    data_list_signal = []
    data_list_noise = []
    nsft = int(nsft)
    param_list = ["f","fd","alpha","sindelta","phi0","psi","cosi"]
    pars = {}
    for key in param_list:
        if key == "f":
            pars[key] = np.random.uniform(fmin + 0.027, fmin + width - 0.027)
        else:
            pars[key] = np.random.uniform(config["cnn_data"][f"{key}_range"][0], config["cnn_data"][f"{key}_range"][1])
    #for j in np.linspace(0,len(param_list)-1,len(param_list)).astype(int):
    #    pars[param_list[j]] = params_low[j] + np.random.uniform(0,1)*(params_high[j] - params_low[j])

    #pars["fd"] = np.random.uniform(-1,1)*10**pars["fd"]
    #sunephem,earthephem = "/home/joseph.bayley/.virtualenvs/soap/share/lalpulsar/earth00-19-DE200.dat.gz","/home/joseph.bayley/.virtualenvs/soap/share/lalpulsar/sun00-19-DE200.dat.gz"

    sig.alpha = pars["alpha"]
    sig.delta = np.arcsin(pars["sindelta"])
    sig.cosi = pars["cosi"]
    sig.phi0 = pars["phi0"]
    sig.psi = pars["psi"]
    sig.f = [pars["f"],pars["fd"]]
    sig.earth_ephem = config["cnn_data"]["earth_ephemeredis"]
    sig.sun_ephem = config["cnn_data"]["sun_ephemeredis"]
    sig.snr = snr
    sig.fmin = fmin
    sig.fmax = fmax
    sig.tref = tstart

            
    data = sig.get_spectrogram(
        tstart = tstart, 
        nsft = len(datah),
        tref=tstart,
        tsft=tsft,
        fmin=flow,
        fmax=fhigh,
        snr=snr,
        noise_sft={"H1":datah,"L1":datal},
        real_data=True,
        rng_med=rng_med,
        sftpaths={"H1":sfts.H1.sftpath,"L1":sfts.L1.sftpath})

    logging.info("Summing SFTs ...")
    data.sum_sfts()
    logging.info("Downsampling SFTs ...")
    data.downsamp_frequency(stride=stride,data_type="summed_norm_sft_power",remove_original=False)

    epochs = data.H1.summed_epochs

    h0 = np.sqrt(np.nanmedian(av_sh))/data.depth

    inj_track = (sig.get_pulsar_path(epochs,"H1") - sig.fmin)*data.tsft

    pars["tref"] = tstart
    pars["snr"] = snr
    pars["av_sh"] = np.nanmedian(av_sh)
    pars["h0"] = h0
    pars["depth"] = data.depth
    pars["fmin"] = fmin
    pars["fmax"] = fmax
    pars["width"] = width

    out_snr_inj = "snr_{}_{}".format(snrstart,snrend)
    logging.info(f"f0: {pars['f']}, fmin: {fmin}, fmax: {fmax}, snr: {snr}")
    signal_outs = cnn_data_gen.return_outputs(
        config, 
        out_snr=out_snr_inj,
        datah=copy.deepcopy(data.H1.downsamp_summed_norm_sft_power),
        datal=copy.deepcopy(data.L1.downsamp_summed_norm_sft_power), 
        fmin=fmin,
        fmax=fmax,
        snr=snr,
        depth=data.depth,
        h0=h0,
        resize_image=resize_image,
        pars=pars,
        inj_track=inj_track,
        epochs=epochs, 
        save_options=save_options, 
        degfree = degfree)

    return noise_outputs, signal_outs
