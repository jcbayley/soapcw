#!/home/joseph.bayley/.virtualenvs/soap27/bin/python

import sys
import os
import getopt
import time
import configparser 
import random
import numpy as np
import argparse

def create_dirs(dirs):
    for i in dirs:
        if not os.path.isdir(i):
            try: 
                os.makedirs(i)
            except:
                print("Could not create directory {}".format(i))
                sys.exit(1)
    print("All directories exist")


def write_subfile(sub_filename,config_file,config,datatype,comment,dirs, verbose=False):
    log_dir,err_dir,output_dir,condor_dir,save_dir = dirs

    with open(sub_filename,'w') as f:
        f.write('# filename: {}\n'.format(sub_filename))
        f.write('universe = vanilla\n')
        execute = os.path.join(os.path.split(sys.executable)[0],config["code"]["search_exec"])
        f.write('executable = {}\n'.format(execute))
        #f.write('enviroment = ""\n')
        f.write('getenv  = True\n')
        f.write("RequestMemory = 50000 \n")
        f.write("request_disk = 10000 \n")
        f.write('log = {}/{}_$(cluster).log\n'.format(log_dir,comment))
        f.write('error = {}/{}_$(cluster).err\n'.format(err_dir,comment))
        f.write('output = {}/{}_$(cluster).out\n'.format(output_dir,comment))
        args = "arguments = -c {} ".format(config_file)

        args += f' -l $(bandstart) -u $(bandend) -w $(bandwidth) --data-type {datatype} -rt {config["data"]["run"]}'
        
        if config["data"]["run"] == "gauss" and datatype == "train":
            args += f' -np {config["data"]["nperband"]}'
        else:
            args += f' -np 1'

        f.write(args + "\n")
        #f.write('accounting_group = aluk.dev.s6.cw.viterbi\n')
        f.write(f'accounting_group = {config["condor"]["accounting_group"]}\n')
        f.write('queue\n')
    if verbose:
        print(time.time(), "generated subfile")


def write_dagfile(config, datatype="train"):

    condor_dir = "{}/condor".format(config["condor"]["root_dir"])
    output_dir = "{}/condor/output".format(config["condor"]["root_dir"])
    save_dir = "{}".format(config["general"]["save_dir"])
    log_dir = "{}/condor/logs".format(config["condor"]["root_dir"])
    err_dir = "{}/condor/err".format(config["condor"]["root_dir"])
    dirs = [log_dir,err_dir,output_dir,condor_dir,save_dir]
    create_dirs(dirs)

    k = 0

    band_list = []
    for i, bs in enumerate(config["data"]["band_starts"]):
        bandstart = np.arange(config["data"]["band_starts"][i], config["data"]["band_ends"][i], config["condor"]["data_load_size"] - 0.1)[:-1]
        bandend = bandstart + config["condor"]["data_load_size"]
        bandwidths =  [config["data"]["band_widths"][i]] * len(bandstart)
        tlist = np.array([bandstart, bandend, bandwidths]).T
        band_list.append(tlist)

    print(config["data"]["run"])

    sub_comment = f'{config["data"]["run"]}_{datatype}_split'
    
    sub_filename = "{}/{}.sub".format(condor_dir,sub_comment)
    write_subfile(sub_filename,config.config_file,config,datatype,sub_comment,dirs,verbose=False)
    dag_filename = "{}/{}.dag".format(condor_dir,sub_comment)

    with open(dag_filename,'w') as f:
        seeds = []
        for l_band in band_list:
            for sband in l_band:
                seeds.append(random.randint(1,1e9))
        for i,l_band in enumerate(band_list):
            for j,sband in enumerate(l_band):
                comment = "F_{}".format(sband[0])
                uid = seeds[i*j]
                jobid = "{}_{}_{}".format(comment,i*j,uid)
                job_string = "JOB {} {}\n".format(jobid,sub_filename)
                retry_string = "RETRY {} 10\n".format(jobid)
                vars_string = f'VARS {jobid} bandstart="{sband[0]}" bandend="{sband[1]}" bandwidth="{sband[2]}"\n'
                f.write(job_string)
                f.write(retry_string)
                f.write(vars_string)

    
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', help='display status', action='store_true')
    parser.add_argument("-c", "--config-file", help="config file contatining parameters")
    #parser.add_argument("-d", "--data-type", help="datatype (train, val, test)", default="train")

    try:                                                     
        args = parser.parse_args()  
    except:  
        sys.exit(1)

    from soapcw.soap_config_parser import SOAPConfig

    config = SOAPConfig(args.config_file)

    for k, typename in enumerate(config["data"]["type"]):
        write_dagfile(config, typename)

if __name__ == '__main__':
    main()


    
