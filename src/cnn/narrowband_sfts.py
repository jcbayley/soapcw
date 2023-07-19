#!/usr/bin/env python

import sys
import os
import getopt
import time
import configparser 
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

    return [fn.decode("utf-8").strip("\n") for fn in sftlist]

def get_filelist(out_path, sftdir, data_type):
    """ Creates a file with a list of all sfts in given directory"""
    filelist_fname = os.path.join(out_path, f"{data_type}_sft_list.txt")
    if os.path.isfile(filelist_fname):
        raise Exception("File already exists: Please delete this file if you want to rewrite")
    filelist_runstr = f"find {sftdir} -name '*.sft' -type f > {filelist_fname}"
    run_command_line(filelist_runstr)
    return filelist_fname

def split_sfts_list(config, detector, filelist_fname, output_dir):
    """Splits each of the sfts using lalapps split SFTs"""

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    sft_list = get_sft_filelist(filelist_fname)

    #runstr = "find {} -name *.sft| sort -n -k4 -t '-'| xargs lalapps_splitSFTs -d {} -fs {} -fe {} -fb {} -fx {} -n {} -as 0 -- ".format(sft_path, detector, band_start, band_end, band_width, band_overlap, out_path)
    
    for fname in sft_list:
        try:
            runstr = "lalapps_splitSFTs -d {} -fs {} -fe {} -fb {} -fx {} -n {} -as 0 -- {}".format(detector, config["params"]["band_start"], config["params"]["band_end"], config["params"]["band_width"], config["params"]["band_overlap"], output_dir, fname)
            run_command_line(runstr)
        except:
            print(f"Split failed for SFT {fname}")


def write_subfile(config_file, params, output_dir, detector, filelist_fname):
    """Write a condor submit file to resubmit this file running the scripts to split the sfts"""
    dirs = [params["dirs"]["output_dir"]]
    for i in ["condor","condor/err","condor/log","condor/out"]:
        dirs.append(os.path.join(params["dirs"]["output_dir"],i))
        
    create_dirs(dirs)

    comment = "narrowband_{}_{}_{}_{}.sub".format(detector,params["params"]["obsrun"],params["params"]["band_start"],params["params"]["band_end"])
    sub_filename = os.path.join(*[params["dirs"]["output_dir"],"condor",comment])

    with open(sub_filename,'w') as f:
        f.write('# filename: {}\n'.format(sub_filename))
        f.write('universe = vanilla\n')
        f.write('executable = {}\n'.format(params["code"]["narrow_exec"]))
        #f.write('enviroment = ""\n')
        f.write('getenv  = True\n')
        f.write('log = {}/condor/log/{}_$(cluster).log\n'.format(params["dirs"]["output_dir"],comment))
        f.write('error = {}/condor/err/{}_$(cluster).err\n'.format(params["dirs"]["output_dir"],comment))
        f.write('output = {}/condor/out/{}_$(cluster).out\n'.format(params["dirs"]["output_dir"],comment))
        f.write('request_disk = 1000 \n')
        f.write(f'arguments = --config-file {config_file} --sft-filelist {filelist_fname} --output-dir {output_dir}\n')
        f.write(f'accounting_group = {params["params"]["accounting_group"]}\n')
        f.write('queue\n')
    
    print(time.time(), f"generated subfile - {sub_filename}")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-f', '--config-file', help='path to config file', type=str, required=True)
    parser.add_argument('-dag', '--make-dag', help='make the dag file for each detector', action="store_true")
    parser.add_argument('-d', '--detector', help='which detector to run on', type=str,  required=False)
    parser.add_argument('-s', '--sft-filelist', help='filelist of sfts', type=str,  required=False)
    parser.add_argument('-o', '--output-dir', help='output_directory for narrowbanded sfts', type=str,  required=False)


    args = parser.parse_args()  

    cp = configparser.ConfigParser()
    config_file = os.path.abspath(args.config_file)
    cp.read(config_file)

    if args.make_dag == True:
        for detector in cp["params"]["detectors"].split(","):
            detector = detector.strip()
            data_type = f'{detector}{cp["dirs"]["data_type"]}'
            sftpath = os.path.join(cp["dirs"]["sft_dir"], data_type)
            output_dir = os.path.join(cp["dirs"]["output_dir"], data_type)

            filelist_fname = get_filelist(cp["dirs"]["output_dir"], sftpath, data_type)

            write_subfile(config_file, cp, output_dir, detector, filelist_fname)

    else:
        split_sfts_list(cp, args.detector, args.sft_filelist, args.output_dir)

    
