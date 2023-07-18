try:
    from .import gen_lookup
except:
    from .import gen_lookup_python as gen_lookup
    #print("using python integration for line aware stat (install boost and gsl C++ libraries for faster runtime -- see documentation)")
import numpy as np
import sys
import json
import pickle as pickle
import argparse
import os

def save_lookup_amp(p1,p2,ratio,outdir, ndet=2, pow_range = (1,400,500), frac_range = (0.1,1,10)):
    """
    save the lookup table for two detectors with the line aware statistic with consitistent amplitude
    (uses json to save file)
    Args
    --------------
    p1 : float
        width of prior of signal model
    p2 : float
        width of prior of line model
    ratio : float
        ratio of line to noise models
    outdir: string
        directory to save lookup table file
    pow_range: tuple
        ranges for the spectrogram power (lower, upper, number), default (1,400,500)
    frac_range: tuple
        ranges for the ratios of sensitivity and duty cycle (lower, upper, number), default (0.1,1,10)
    """
    minimum,maximum,num = pow_range
    minn,maxn,numn = frac_range
    #ch_arr_app = gen_data.gen_lookup_noise(np.linspace(minimum,maximum,num),np.linspace(minimum,maximum,num),np.linspace(minn,maxn,numn),int_type="chi2",approx=False,pvs=p1,pvl=p2,beta=ratio)
    filename = outdir+"/ch2_signoiseline_{}det_{}_{}_{}.json".format(ndet,p1,p2,ratio)
    if os.path.isfile(filename):
        print(("File {} exists".format(filename)))
    else:
        ch_arr_app = gen_lookup.LineAwareAmpStatistic(np.linspace(minimum,maximum,num),fraction=np.linspace(minn,maxn,numn), ndet=ndet,signal_prior_width=p1,line_prior_width=p2,noise_line_model_ratio=ratio)
        with open(filename,'w+') as f:
            save_data = [[minimum,maximum,num,minn,maxn,numn],np.log(ch_arr_app[0]).tolist()]

            json.dump(save_data,f)

def save_lookup(p1,p2,ratio,outdir,ndet=2,pow_range = (1,400,500), k=2, N=48):
    """
    save the lookup table for two detectors with the line aware statistic
    
    Args
    --------------
    p1 : float
        width of prior of signal model
    p2 : float
        width of prior of line model
    ratio : float
        ratio of line to noise models
    outdir: string
        directory to save lookup table file
    pow_range: tuple
        ranges for the spectrogram power (lower, upper, number), default (1,400,500)

    """

    minimum,maximum,num = pow_range

    if os.path.isfile(outdir+"/signoiseline_{}det_{}_{}_{}.txt".format(ndet, p1,p2,ratio)):
        pass
    else:
        if ndet == 1:
            ch_arr_app = gen_lookup.LineAwareStatistic(np.linspace(minimum,maximum,num),
                                                        ndet=ndet,
                                                        signal_prior_width=p1,
                                                        line_prior_width=p2,
                                                        noise_line_model_ratio=ratio)
        if ndet == 2:
            powers = np.linspace(minimum,maximum,num)
            ch_arr_app = gen_lookup.LineAwareStatistic(powers=powers,
                                                        ndet=ndet,
                                                        k = k,
                                                        N = N,
                                                        signal_prior_width=p1,
                                                        line_prior_width=p2,
                                                        noise_line_model_ratio=ratio)
        with open(outdir+"/signoiseline_{}det_{}_{}_{}.txt".format(ndet, p1,p2,ratio),'wb') as f:
            header = "{} {} {}".format(minimum,maximum,num)
            np.savetxt(f,np.log(ch_arr_app.signoiseline),header = header)



def resave_files(p1,p2,ratio,output):
    """
    resave text files into pickle format
    """
    if os.path.isfile(output+"/txt/ch2_signoiseline_{}_{}_{}.txt".format(p1,p2,ratio)):
        with open(output+"/txt/ch2_signoiseline_{}_{}_{}.txt".format(p1,p2,ratio),'rb') as f:
            save_array = pickle.load(f)
        if os.path.isdir(output+"/pkl/"):
            pass
        else:
            os.mkdir(output+"/pkl/")
        with open(output+"/pkl/ch2_signoiseline_{}_{}_{}.pkl".format(p1,p2,ratio),'wb') as f:
            pickle.dump(save_array,f,protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog = 'SOAP lookup table generation',
                    description = 'generates lookup tables for SOAP',)

    parser.add_argument('--amp-stat',action='store_true') 
    parser.add_argument('-s', '--signal-probability', required=True, type=float) 
    parser.add_argument('-l', '--line-probability', required=True, type=float) 
    parser.add_argument('-n', '--noise-line-ratio', required=True, type=float) 
    parser.add_argument('-ndet', '--ndet', default=2, required=False, type=int) 
    parser.add_argument('-k', default=2, required=False, type=int) 
    parser.add_argument('-N', '--num-sfts', default=48, required=False, type=int) 
    parser.add_argument('-o', '--save-dir', required=True, type=str) 
    
    parser.add_argument('-pmin', '--pow-min', default=1, required=False, type=float) 
    parser.add_argument('-pmax', '--pow-max', default=400, required=False, type=float) 
    parser.add_argument('-np', '--n-powers', default=500, required=False, type=int) 

    parser.add_argument('-fmin', '--frac-min', default=0.1, required=False, type=float) 
    parser.add_argument('-fmax', '--frac-max', default=1, required=False, type=float) 
    parser.add_argument('-nf', '--n-fracs', default=10, required=False, type=int) 

    args = parser.parse_args()

    if not args.amp_stat:
        save_lookup(args.signal_probability,
                    args.line_probability,
                    args.noise_line_ratio,
                    args.save_dir,
                    k = args.k,
                    N = args.num_sfts,
                    ndet=args.ndet,
                    pow_range = (args.pow_min,args.pow_max,args.n_powers))
    else:
        save_lookup_amp(args.signal_probability,
                        args.line_probability,
                        args.noise_line_ratio,
                        args.save_dir, 
                        ndet = args.ndet, 
                        pow_range = (args.pow_min,args.pow_max,args.n_powers), 
                        frac_range = (args.frac_min,args.frac_max,args.n_fracs))
