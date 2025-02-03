#!/usr/bin/env python

import sys
import os
import getopt
import time
import configparser 
from soapcw.soap_config_parser import SOAPConfig
import random
import numpy as np
import argparse
import logging
import subprocess


def create_dirs(dirs):
    for i in dirs:
        if not os.path.isdir(i):
            try: 
                os.makedirs(i)
            except:
                print("Could not create directory {}".format(i),file=sys.stderr)
                sys.exit(1)
    print("All directories exist")

def run_command_line(cl):
    """Run a string commandline as a subprocess, check for errors and return output."""

    logging.info('Executing: ' + cl)
    try:
        out = subprocess.check_output(cl,                       # what to run
                                      stderr=subprocess.STDOUT, # catch errors
                                      shell=True,               # proper environment etc
                                      universal_newlines=True   # properly display linebreaks in error/output printing
                                     )
    except subprocess.CalledProcessError as e:
        logging.error('Execution failed:')
        logging.error(e.output)
        print("-------failed {}".format(cl))
        out = None
        #raise
    os.system('\n')

    return(out)

def get_sft_filelist(file_list):
    """
    look through sftpaths to find all sft files
    """
    with open(file_list,"rb") as f:
        sftlist = f.readlines()

    decode_list = [fn.decode("utf-8").strip("\n") for fn in sftlist]

    # sort the loaded list by their start times
    start_times = [float(f.split("-")[-2]) for f in decode_list]
    sorted_list = np.array(sorted(zip(start_times,decode_list), key=lambda x : x[0]))

    return sorted_list[:,1]

def get_filelist(out_path, sftdir, obs_run, detector_str, overwrite=False):
    """ Creates a file with a list of all sfts in given directory"""
    filelist_fname = os.path.join(out_path, f"{obs_run}_{detector_str}_sft_list.txt")
    if not overwrite:
        if os.path.isfile(filelist_fname):
            raise Exception("File already exists: Please delete this file if you want to rewrite")
    filelist_runstr = f"find {sftdir} -name '*.sft' -type f > {filelist_fname}"
    print(f"Creating filelist: {filelist_runstr}")
    run_command_line(filelist_runstr)
    return filelist_fname

def split_sfts_list(
    detector, 
    filelist_fname, 
    output_dir,
    fband_start,
    fband_end,
    fband_width,
    fband_overlap,
    ):
    """Splits each of the sfts using lalapps split SFTs"""

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    sft_list = get_sft_filelist(filelist_fname)

    #runstr = "find {} -name *.sft| sort -n -k4 -t '-'| xargs lalapps_splitSFTs -d {} -fs {} -fe {} -fb {} -fx {} -n {} -as 0 -- ".format(sft_path, detector, band_start, band_end, band_width, band_overlap, out_path)

    for fname in sft_list:
        try:
            runstr = "lalpulsar_splitSFTs -fs {} -fe {} -fb {} -fx {} -n {} -as 0 -- {}".format(
                fband_start, 
                fband_end, 
                fband_width, 
                fband_overlap, 
                output_dir, 
                fname)
            run_command_line(runstr)
        except:
            print(f"Split failed for SFT {fname}")


def write_subfile(config_file, params, output_dir, detector, filelist_fname, band_min, band_max):
    """Write a condor submit file to resubmit this file running the scripts to split the sfts"""
    #dirs = [params["dirs"]["output_dir"]]
    #for i in ["condor","condor/err","condor/log","condor/out"]:
    #    dirs.append(os.path.join(params["dirs"]["output_dir"],i))
        
    #create_dirs(dirs)

    condor_dir = os.path.join(params["general"]["root_dir"],"condor_narrowband")

    if not os.path.isdir(params["general"]["root_dir"]):
        os.makedirs(params["general"]["root_dir"])

    if not condor_dir:
        os.makedirs(condor_dir)

    for fl in ["err","log","out"]:
        if not os.path.isdir(os.path.join(condor_dir,fl)):
            os.makedirs(os.path.join(condor_dir,fl))

    comment = "narrowband_{}_{}_{}_{}.sub".format(detector,params["data"]["obs_run"],band_min,band_max)
    sub_filename = os.path.join(*[condor_dir,comment])

    execute = os.path.join(os.path.split(sys.executable)[0],params["scripts"]["narrow_exec"])

    with open(sub_filename,'w') as f:
        f.write('# filename: {}\n'.format(sub_filename))
        f.write('universe = vanilla\n')
        f.write('executable = {}\n'.format(execute))
        #f.write('enviroment = ""\n')
        f.write('getenv  = True\n')
        f.write('log = {}/log/{}_$(cluster).log\n'.format(condor_dir,comment))
        f.write('error = {}/err/{}_$(cluster).err\n'.format(condor_dir,comment))
        f.write('output = {}/out/{}_$(cluster).out\n'.format(condor_dir,comment))
        f.write('request_disk = 1000 \n')
        f.write(f'arguments = --config-file {config_file} --sft-filelist {filelist_fname} --output-dir {output_dir}\n')
        f.write(f'accounting_group = {params["condor"]["accounting_group"]}\n')
        f.write('queue\n')
    
    print(time.time(), f"generated subfile - {sub_filename}")


    
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-c', '--config-file', help='path to config file', type=str, required=True)
    parser.add_argument('-dag', '--make-dag', help='make the dag file for each detector', action="store_true")
    parser.add_argument('-mf', '--make-filelist', help='make the dag file for each detector', action="store_true")
    parser.add_argument('-d', '--detector', help='which detector to run on', type=str,  required=False)
    parser.add_argument('-s', '--sft-filelist', help='filelist of sfts', type=str,  required=False)
    parser.add_argument('-o', '--output-dir', help='output_directory for narrowbanded sfts', type=str,  required=False)
    parser.add_argument('-ov', '--overwrite', help='overwrite filelist', action="store_true")


    args = parser.parse_args()  

    config_file = os.path.abspath(args.config_file)
    cp = SOAPConfig(config_file)

    if args.make_filelist == True:
        for i, detector in enumerate(cp["data"]["detectors"]):
            input_sft_dir = cp["input"]["sft_dirs"][i]
            output_sft_dir = os.path.join(cp["narrowband"]["narrowband_sft_dir"])
            filelist_fname = get_filelist(output_sft_dir, input_sft_dir, cp['data']['obs_run'], detector, overwrite=args.overwrite)
        print("Made filelists")

    if args.make_dag == True:
        print("Making dag files ........")
        for i,detector in enumerate(cp["data"]["detectors"]):
            output_dir = os.path.join(cp["narrowband"]["narrowband_sft_dir"], detector)
            filelist_fname = os.path.join(cp["narrowband"]["narrowband_sft_dir"], f"{cp['data']['obs_run']}_{detector}_sft_list.txt")
            #filelist_fname = get_filelist(cp["dirs"]["output_dir"], sftpath, data_type)

            write_subfile(config_file, cp, output_dir, detector, filelist_fname, cp["narrowband"]["band_min"], cp["narrowband"]["band_max"])
        print("Made dag files")

    else:
        split_sfts_list(args.detector, args.sft_filelist, args.output_dir, cp["narrowband"]["band_min"], cp["narrowband"]["band_max"], cp["narrowband"]["band_width"], cp["narrowband"]["band_overlap"])


if __name__ == '__main__':
    main()
