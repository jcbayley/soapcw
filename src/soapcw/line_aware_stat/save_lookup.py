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

def save_lookup_amp(
    p1,
    p2,
    ratio,
    outdir, 
    ndet=2, 
    k=2,
    N=48,
    pow_range = (1,400,500), 
    frac_range = (0.1,1,10)):
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
    ch_arr_app = gen_lookup.LineAwareAmpStatistic(
            np.linspace(minimum,maximum,num),
            fractions=np.linspace(minn,maxn,numn), 
            ndet=ndet,
            k=k,
            N=N,
            signal_prior_width=p1,
            line_prior_width=p2,
            noise_line_model_ratio=ratio)

    ch_arr_app.save_lookup(outdir,log=True, stat_type = "signoiseline")


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

    powers = np.linspace(minimum,maximum,num)
    ch_arr_app = gen_lookup.LineAwareStatistic(powers=powers,
                                                ndet=ndet,
                                                k = k,
                                                N = N,
                                                signal_prior_width=p1,
                                                line_prior_width=p2,
                                                noise_line_model_ratio=ratio)

    ch_arr_app.save_lookup(outdir,log=True, stat_type = "signoiseline")

        #with open(outdir+"/signoiseline_{}det_{}_{}_{}.txt".format(ndet, p1,p2,ratio),'wb') as f:
        #    header = "{} {} {}".format(minimum,maximum,num)
        #    np.savetxt(f,np.log(ch_arr_app.signoiseline),header = header)



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
    parser.add_argument('-s', '--signal-prior-width', required=True, type=float) 
    parser.add_argument('-l', '--line-prior-width', required=True, type=float) 
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

    parser.add_argument('-A', '--make-all', action="store_true") 

    args = parser.parse_args()

    if not args.amp_stat:
        if args.make_all:
            for det in [1,2]:
                for nsft, mpower in [(48, 150),(96, 250),(144, 360),(192,470)]:
                    save_lookup(args.signal_prior_width,
                        args.line_prior_width,
                        args.noise_line_ratio,
                        args.save_dir,
                        k = args.k,
                        N = nsft,
                        ndet=det,
                        pow_range = (args.pow_min,mpower,args.n_powers))
        else:
            save_lookup(args.signal_prior_width,
                    args.line_prior_width,
                    args.noise_line_ratio,
                    args.save_dir,
                    k = args.k,
                    N = args.num_sfts,
                    ndet=args.ndet,
                    pow_range = (args.pow_min,args.pow_max,args.n_powers))
    else:
        if args.make_all:
            for det in [1,2]:
                for nsft, mpower in [(48, 150),(96, 250),(144, 360),(192,470)]:
                    save_lookup_amp(args.signal_prior_width,
                        args.line_prior_width,
                        args.noise_line_ratio,
                        args.save_dir, 
                        k=args.k,
                        N=nsft,
                        ndet = args.ndet, 
                        pow_range = (args.pow_min,mpower,args.n_powers), 
                        frac_range = (args.frac_min,args.frac_max,args.n_fracs))

        else:
            save_lookup_amp(args.signal_prior_width,
                        args.line_prior_width,
                        args.noise_line_ratio,
                        args.save_dir, 
                        k=args.k,
                        N=args.num_sfts,
                        ndet = args.ndet, 
                        pow_range = (args.pow_min,args.pow_max,args.n_powers), 
                        frac_range = (args.frac_min,args.frac_max,args.n_fracs))
