import soapcw_pipeline
import soapcw
import argparse
import numpy as np
import sys
import os
import time

def narrowband_for_followup(sft_filelists, output_directory, frequencylist_fname):

    sftlist = []
    for sftfile in sft_filelists:
        with open(sftfile,"r") as f:
            sftlist.extend(f.readlines())

    sttime = [float(os.path.basename(name).split("-")[-2]) for name in sftlist]

    print("numsfts: {}".format(len(sftlist)))
    tmin = np.min(sttime) 
    tmax = np.max(sttime) + 1800 # start of last sft plus length (in this case we are only using 1800s sfts
    #tmax = min(sttime) + 48*10*1800 # smaller range for testing
    print(f"minmaxtime, {tmin} --> {tmax}")
    
    # get the sfts in the specified time range 
    sftlist = soapcw_pipeline.run_full_soap_astro.get_sfts_in_range(tmin,tmax,sftlist)


    with open(frequencylist_fname, "r") as f:
        for val in f.readlines():
            valsplit = val.split(" ")
            freqmin, freqmax = float(valsplit[0]), float(valsplit[1])
            width = freqmax - freqmin
            freqmin = freqmin - 0.5
            if width < 1:
                width = 1
                freqmax = freqmin + width + 0.01
            freqmax = freqmax + 0.5
            width = 1.0
            print(f"Running: {freqmin}, {freqmax}, {width}")

            width = freqmax - freqmin

            output_sub_directory = os.path.join(output_directory, f"F_{freqmin:.2f}_{freqmax:.2f}_{width:.2f}")
            
            soapcw.narrowband_sfts.split_sfts_list(
                sorted(sftlist),
                output_sub_directory,
                freqmin,
                freqmax,
                width,
                0
            )
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s1", "--sfts_1", help="file containing list of sfts")
    parser.add_argument("-s2", "--sfts_2", help="file containing list of sfts")
    parser.add_argument("-o", "--output-directory", help="directory to put narrowbanded sfts")
    parser.add_argument("-f", "--frequency-list", help="list of frequencies to narrowband")
    try:                                                     
        args = parser.parse_args()  
    except:  
        sys.exit(1)


    sft_filelists = [args.sfts_1, args.sfts_2]
    narrowband_for_followup(sft_filelists, args.output_directory, args.frequency_list)

if __name__ == '__main__':
    main()