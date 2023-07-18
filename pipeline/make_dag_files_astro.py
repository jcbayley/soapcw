#!/usr/bin/python
import sys
import os
import getopt
import time
import configparser 
import argparse
import random
import numpy as np
from astropy.time import Time
import json
from .soap_config_parser import SOAPConfig


def write_plot_subfile(sub_filename,config,config_file,cdirs,comment,verbose=True):

    with open(sub_filename,'w') as f:
        f.write('# filename: {}\n'.format(sub_filename))
        f.write('universe = vanilla\n')
        #f.write('executable = {}\n'.format(config["scripts"]["search_exec"]))
        # finds the location of the python executable and replaces with the correct soap exec
        execute = os.path.join(os.path.split(sys.executable)[0],config["scripts"]["search_exec"])
        f.write('executable = {}\n'.format(execute))
        f.write('getenv  = True\n')
        f.write('RequestMemory = {} \n'.format(config["condor"]["memory"]))
        f.write(f'request_disk={config["condor"]["request_disk"]}\n')
        f.write('log = {}/{}_$(cluster).log\n'.format(cdirs["log_dir"],comment))
        f.write('error = {}/{}_$(cluster).err\n'.format(cdirs["err_dir"],comment))
        f.write('output = {}/{}_$(cluster).out\n'.format(cdirs["output_dir"],comment))
        args = f'arguments = --config-file {config_file} -s $(bandstart) -e $(bandend) -r {config["data"]["obs_run"]} -l {config["lookuptable"]["lookup_dir"]} -w {config["condor"]["band_load_size"]} -sd {config["output"]["sub_directory"]}'

        f.write(args + "\n")
        #f.write('accounting_group = aluk.dev.s6.cw.viterbi\n')
        f.write(f'accounting_group = {config["condor"]["accounting_group"]}\n')
        f.write('queue\n')
    if verbose:
        print(time.time(), "generated subfile")
                
def write_html_subfile(sub_filename,config, config_file, cdirs, comment, verbose=True):

    with open(sub_filename,'w') as f:
        f.write('# filename: {}\n'.format(sub_filename))
        f.write('universe = vanilla\n')
        f.write('executable = {}\n'.format(config["scripts"]["html_exec"]))
        #f.write('enviroment = ""\n')
        f.write('getenv  = True\n')
        f.write('RequestMemory = {} \n'.format(config["condor"]["memory"]))
        f.write(f'request_disk={config["condor"]["request_disk"]}\n')
        f.write('log = {}/{}_$(cluster).log\n'.format(cdirs["log_dir"],comment))
        f.write('error = {}/{}_$(cluster).err\n'.format(cdirs["err_dir"],comment))
        f.write('output = {}/{}_$(cluster).out\n'.format(cdirs["output_dir"],comment))
        
        args = f'arguments = -c {config_file}'
        f.write(args + "\n")

        #f.write('accounting_group = aluk.dev.s6.cw.viterbi\n')
        f.write(f'accounting_group = {config["condor"]["accounting_group"]}\n')
        f.write('queue\n')
    if verbose:
        print(time.time(), "generated subfile")


def write_dagfile(config_file, config, verbose = False):
    """ Write a dag file containing all link to submit file for all sub bands"""

    condor_dirs = {}
    condor_dirs["condor_dir"] = os.path.join(config["general"]["root_dir"], "condor")

    condor_dirs["log_dir"] = os.path.join(condor_dirs["condor_dir"],"logs")
    condor_dirs["err_dir"] = os.path.join(condor_dirs["condor_dir"],"err")
    condor_dirs["output_dir"] = os.path.join(condor_dirs["condor_dir"],"output")
    
    condor_dirs["save_directory"] = os.path.join(config["output"]["save_directory"], config["data"]["obs_run"], config["output"]["sub_directory"])
    
    for key, val in condor_dirs.items():
        if not os.path.isdir(val):
            os.makedirs(val)

    ##
    ## Make the sub file for SOAP run and summary page generation
    ##

    bs1 = config["data"]["band_starts"]
    be1 = config["data"]["band_ends"]

    run_sub_comment = "soap_astro_F{}_{}_{}".format(bs1[0],be1[-1],config["data"]["obs_run"])
    html_sub_comment = "soap_html_F{}_{}_{}".format(bs1[0],be1[-1],config["data"]["obs_run"])

    run_sub_filename = "{}/{}.sub".format(condor_dirs["condor_dir"],run_sub_comment)
    html_sub_filename = "{}/{}.sub".format(condor_dirs["condor_dir"],html_sub_comment)

    write_plot_subfile(run_sub_filename,config,config_file,condor_dirs,run_sub_comment,verbose=verbose)
    write_html_subfile(html_sub_filename,config,config_file,condor_dirs,html_sub_comment,verbose=verbose)

    ##
    ## Make list of band starts from condor split
    ##
    band_starts = []
    band_ends = []
    for i, bs in enumerate(bs1):
        t_bss = np.arange(bs1[i],
            be1[i],
            config["condor"]["band_load_size"]
            )[:-1]
        t_bes = t_bss + config["condor"]["band_load_size"] + config["data"]["band_widths"][i]

        band_starts.extend(list(t_bss))
        band_ends.extend(list(t_bes))

    dag_filename = "{}/{}.dag".format(condor_dirs["condor_dir"],run_sub_comment)

    jobids = []
    with open(dag_filename,'w') as f:
        seeds = []
        for i in range(len(band_starts)):
            seeds.append(random.randint(1,1e9))

        for i in range(len(band_starts)):
            comment = "F_{}".format(int(band_starts[i]))
            jobid = "{}_{}_{}".format(comment,i,seeds[i])
            jobids.append(jobid)
            job_string = "JOB {} {}\n".format(jobid,run_sub_filename)
            retry_string = "RETRY {} 2\n".format(jobid)
            vars_string = 'VARS {} bandstart="{}" bandend="{}"\n'.format(jobid,band_starts[i],band_ends[i])
            f.write(job_string)
            f.write(retry_string)
            f.write(vars_string)
            """
            if i == 0:
                order_string = None
            else:
                prev_job_ids = ""
                prev_uid = seeds[i-1]
                prev_comment = "F_{}".format(band_starts[i-1])
                prev_jobid = " {}_{}_{}".format(prev_comment,i-1,prev_uid)
                order_string = "PARENT {} CHILD {} \n".format(prev_jobid,jobid)
                #f.write(order_string)
            """
        
        comment = "F_{}".format("html")
        jobid = "{}_{}".format(comment,1)
        job_string = "JOB {} {}\n".format(jobid,html_sub_filename)
        retry_string = "RETRY {} 1\n".format(jobid)
        #vars_string = 'VARS \n'
        #prev_jobid = "{}".format([i jobids])
        order_string = "PARENT {} CHILD {} \n".format(" ".join(jobids),jobid)
        f.write(job_string)
        f.write(retry_string)
        #f.write(vars_string)
        f.write(order_string)
    if verbose:
        print(time.time(), "generated dag file")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", help="configuration file for soap")
    try:                                                     
        args = parser.parse_args()  
    except:  
        sys.exit(1)

    config_file = os.path.abspath(args.config_file)

    config = SOAPConfig(args.config_file)
    #config = configparser.ConfigParser()
    #config.read(config_file)

    write_dagfile(config_file, config)

if __name__ == '__main__':
    main()


    
