import os
import shutil
import h5py
import json
import numpy as np
from collections import OrderedDict
import importlib.resources as pkg_resources
import importlib_resources
import pandas as pd
import matplotlib.pyplot as plt

def make_directory_structure(root_dir):

    my_resources = importlib_resources.files("soapcw_pipeline")
    cssfile = (my_resources / "css"/ "general.css")
    javascriptfile = (my_resources / "scripts"/ "table_scripts.js")
    javascriptfiletoplist = (my_resources / "scripts"/ "table_scripts_toplist.js")
    
    if not os.path.isdir(os.path.join(root_dir, "css")):
        os.makedirs(os.path.join(root_dir, "css"))

    if not os.path.isdir(os.path.join(root_dir, "scripts")):
        os.makedirs(os.path.join(root_dir, "scripts"))

    shutil.copy(cssfile, os.path.join(root_dir, "css/"))
    shutil.copy(javascriptfile, os.path.join(root_dir, "scripts/"))
    shutil.copy(javascriptfiletoplist, os.path.join(root_dir, "scripts/"))
    """
    shutil.copy("../css/general.css", os.path.join(root_dir, "css/"))
    shutil.copy("../scripts/table_scripts.js", os.path.join(root_dir, "scripts/"))
    """

def create_page_usage_page():
    # Read the original HTML file
    my_resources = importlib_resources.files("soapcw_pipeline")
    original_file = (my_resources / "html"/ "usage.html")
    with open(original_file) as source:
        content = source.read()

    return content


def create_home_page():
    # Read the original HTML file
    my_resources = importlib_resources.files("soapcw_pipeline")
    original_file = (my_resources / "html"/ "homepage.html")
    with open(original_file) as source:
        content = source.read()

    return content

def create_astro_page(run_headings, sub_headings):

    # Read the original HTML file
    my_resources = importlib_resources.files("soapcw_pipeline")
    original_file = (my_resources / "html"/ "astropage.html")
    with open(original_file) as source:
        content = source.read()

    # Find the starting point of the dynamic content
    content = content.replace("!!!REPLACE_run_headings_REPLACE!!!", run_headings)
    content = content.replace("!!!REPLACE_sub_headings_REPLACE!!!", sub_headings)

    return content

def create_line_page(run_headings, sub_headings):

    # Read the original HTML file
    my_resources = importlib_resources.files("soapcw_pipeline")
    original_file = (my_resources / "html"/ "linepage.html")
    with open(original_file) as source:
        content = source.read()

    # Find the starting point of the dynamic content
    content = content.replace("!!!REPLACE_run_headings_REPLACE!!!", run_headings)
    content = content.replace("!!!REPLACE_sub_headings_REPLACE!!!", sub_headings)

    return content



def create_run_page(run_headings, obs_run="run", toplist=''):
    # Read the original HTML file
    my_resources = importlib_resources.files("soapcw_pipeline")
    original_file = (my_resources / "html"/ "runpage.html")
    with open(original_file) as source:
        content = source.read()

    # Find the starting point of the dynamic content
    content = content.replace("!!!REPLACE_obs_run_REPLACE!!!", obs_run)
    content = content.replace("!!!REPLACE_run_headings_REPLACE!!!", run_headings)

    return content


def read_line_files_old(linefile,det=None):
    """
    open line list file and save information on line
    returns
    --------
    linelist: list
        list of lines []
    """
    data = []
    with open(linefile,"r") as f: 
        i = 0
        for line in f.readlines(): 
            if i == 0:
                i +=1
                continue
            if not line.startswith("%"): 
                lnsplit = line.split("\t")
                lnsave = []
                for ln in lnsplit:
                    try:
                        lnsave.append(float(ln))
                    except:
                        lnsave.append("{} {} {} \n".format(det,lnsplit[0],ln))
                data.append(lnsave)
                del lnsplit,lnsave

    return data

def get_line_info_old(linedata, flow, fhigh):
    info = ""
    # if line files have been loaded include information on known lines
    if linedata is not None:
        for line in linedata:
            # if type of line is not a comb
            if line[1] == 0:
                lowfreq = line[0] - line[5]
                highfreq = line[0] + line[6]

                if flow < lowfreq < fhigh or flow < highfreq < fhigh or flow < line[0] < fhigh:
                    info += line[7]
            # if type of line is a comb include the initial frequency, and first harmonic
            elif line[1] == 1:
                spacing = line[0]
                fharm = line[3]
                lharm = line[4]
                offset = line[2]

                ranges = np.arange(fharm,lharm)*spacing + offset
                for comb in ranges:
                    if flow < comb < fhigh:
                        info += line[7]

    if info == "" or info == np.nan or info == "nan" or info == "NaN":
        pass
    else:
        info = "lines:" + info
    return info

def get_line_info(linedata, flow, fhigh):
    info = ""
    # if line files have been loaded include information on known lines
    if linedata is not None:

        lines = linedata.loc[linedata["Type (0:line; 1:comb; 2:comb with scaling width)"] == 0]
        lines = lines.loc[
            (lines["Frequency or frequency spacing [Hz]"] + lines[" Left width [Hz]" ] < fhigh) & 
            (lines["Frequency or frequency spacing [Hz]"] - lines[" Right width [Hz]" ] > flow)]

        # the spaces at the front of comments is important for the column name
        for index, line in lines.iterrows():
            info += line[" Comments"]

        combs = linedata.loc[linedata["Type (0:line; 1:comb; 2:comb with scaling width)"] == 1]

        for index, comb in combs.iterrows():
            spacing = comb["Frequency or frequency spacing [Hz]"]
            first = comb["First visible harmonic"]
            last = comb[" Last visible harmonic"]
            offset = comb["Frequency offset [Hz]"]

            comb_freqs = np.arange(first, last, spacing) + offset

            if np.any((comb_freqs < fhigh) & (comb_freqs > flow)):
                info += comb[" Comments"]

    #print(info)
    if info == "" or info == np.nan or info == "nan" or info == "NaN":
        pass
    else:
        info = "lines:" + info
    return info

def get_hwinj_info(hwinjtable, flow, fhigh):
    """ """

    hwinjs = hwinjtable.loc[
            ((hwinjtable["f0 (epoch start)"] < fhigh) & 
            (hwinjtable["f0 (epoch start)"] > flow)) | 
            ((hwinjtable["f0 (epoch start)"] > fhigh) & 
            (hwinjtable["f0 (epoch stop)"] < fhigh))| 
            ((hwinjtable["f0 (epoch start)"] > flow) & 
            (hwinjtable["f0 (epoch stop)"] < flow))]

    info = ""
    for index, line in hwinjs.iterrows():
        info += f"<a href='https://ldas-jobs.ligo.caltech.edu/~keith.riles/cw/injections/preO3/preO3_injection_params.html'> hwinj: {line['Pulsar']}</a>"
    
    return info

def make_json_from_hdf5(root_dir, linepaths=None, table_order=None, hwinjfile=None, freqbands = [20,500,1000,1500,2000]):
    """Loads in all hdf5 files and writes them into json format that can be loaded by javascript into summary pages"""
    hdf5dir = os.path.join(root_dir, "data")

    json_data = []

    if linepaths is not None:
        linedataframes = []
        for linefile in linepaths:
            linedataframes.append(pd.read_csv(linefile))
        linedata = pd.concat(linedataframes, axis=0, ignore_index=True)

    if hwinjfile is not None:
        if hwinjfile.endswith("html"):
            hwinjdata = pd.read_html(hwinjfile, header=0)[0]


    """
    linedata = None
    if linepaths is not None:
        if type(linepaths) == str:
            linedata = read_line_files(linepaths)
        else:
            linedata = []
            for linefile in linepaths:
                if "H1" in linefile:
                    det = "H1"
                if "L1" in linefile:
                    det = "L1"
                linedata.extend(read_line_files(linefile,det = det))
    """

    for fname in os.listdir(hdf5dir):
        with h5py.File(os.path.join(hdf5dir, fname), "r", track_order=True) as f:
            for i in range(len(f[list(f.keys())[0]])):
                temp_data = OrderedDict()
                for key in table_order:
                    if key not in list(f.keys()): continue
                    # convert the plot path to the location on the server (works only for LIGO servers at the moment)
                    if key == "plot_path":
                        path = f[key][i].decode()
                        if "/soap_2/" in path:
                            path = path.replace("/soap_2/","/soap/")
                        path = path.replace("/home/", "https://ldas-jobs.ligo.caltech.edu/~").replace("/public_html","")
                        temp_data[key] = path
                    else:
                        temp_data[key] = np.round(f[key][i],2)
                    #if i > 50:
                    #    sys.exit()
                    info = ""
                    if hwinjfile is not None:
                        info += get_hwinj_info(hwinjdata, f["fmin"][i], f["fmax"][i])
                    if linepaths is not None:
                        info += get_line_info(linedata, f["fmin"][i], f["fmax"][i])
                    temp_data.update({"info":info})
                    temp_data.move_to_end("info")
                json_data.append(temp_data)
       
    
    # sort the table so the highest lineaware statitics shjow first
    try:
        sorted_json_data = sorted(json_data, key=lambda d: d["lineaware_stat"])
    except:
        sorted_json_data = sorted(json_data, key=lambda d: d["fmin"])

    with open(os.path.join(root_dir, "table.json"), "w") as f:
        json.dump(sorted_json_data, f)

    # find the top statistics in each band
    def filter_json_by_key_range(json_data, key, min_value, max_value):
        return [d for d in json_data if min_value <= d.get(key, float('inf')) < max_value]


    toplistjson = []
    for bind in range(len(freqbands) - 1):
        f0,f1 = freqbands[bind], freqbands[bind+1]
        # filter json to in band
        filtered_json_data = filter_json_by_key_range(json_data, "fmin", f0, f1)
        # take top 2% of stats in band
        fraction_keep = 0.97
        if "lineaware_stat" in filtered_json_data[0]:
            sorted_filtered_json_data = sorted(filtered_json_data, key=lambda d: d["lineaware_stat"])[int(fraction_keep*len(filtered_json_data)):]
        else:
            sorted_filtered_json_data = sorted(filtered_json_data, key=lambda d: d["H1_viterbistat"])[int(fraction_keep*len(filtered_json_data)):]

        print(f0, f1, len(sorted_filtered_json_data), len(sorted_filtered_json_data), len(filtered_json_data))
        toplistjson.extend(sorted_filtered_json_data)

    print(len(toplistjson))
    # also save separate list of just the top statistics
    with open(os.path.join(root_dir, "table_toplist.json"), "w") as f:
        json.dump(toplistjson, f)

def get_public_dir(root_dir):

    username = root_dir.split("/")[2]
    public = root_dir.split("/public_html/")[1]
    public_html = f"https://ldas-jobs.ligo.caltech.edu/~{username}/{public}"
    return public_html

def get_html_string(root_dir, linepaths=None, table_order=None, force_overwrite=False, hwinjfile=None):

    public_dir = get_public_dir(root_dir)
    print("pbdir: ", public_dir)

    run_headings = ""
    sub_headings = ""

    obsruns = sorted(os.listdir(root_dir))

    for head in obsruns:
        if os.path.isdir(os.path.join(root_dir, head)):
            i = 0
            for subhead in os.listdir(os.path.join(root_dir, head)):
                subdir = os.path.join(root_dir, head, subhead)
                if os.path.isdir(subdir):
                    if i == 0:
                        run_headings += f'<a href="{public_dir}/{head}/{subhead}/{subhead}.html">{head}</a>'
                        i += 1
                    else:
                        continue

    for head in obsruns:
        if os.path.isdir(os.path.join(root_dir, head)):
            sub_headings += f"<h1> {head} </h1> <ul>"
            for subhead in os.listdir(os.path.join(root_dir, head)):
                subdir = os.path.join(root_dir, head, subhead)
                if os.path.isdir(subdir):
                    sub_headings += f'<l1><a href="{public_dir}/{head}/{subhead}/{subhead}.html"> {subhead} </a></li> </br>'

                    if os.path.exists(os.path.join(subdir, "table.json")):
                        if force_overwrite:
                            try:
                                make_json_from_hdf5(subdir, linepaths, table_order, hwinjfile=hwinjfile)
                            except Exception as e:
                                print(f"WARNING: Cannot recreate json table")
                                print(e)
                        else:
                            print(f"WARNING: No new updates to {subhead}, {subdir}")
                    else:
                        print(f"Creating json for {subhead} -- {subdir}")
                        make_json_from_hdf5(subdir, linepaths, table_order, hwinjfile=hwinjfile)

                    try:
                        make_summary_histogram(subdir)
                    except Exception as e:
                        print(f"Could not make histogram for {subdir}")
                        print(e)

                    run_html = create_run_page(run_headings, obs_run=head)
                    with open(os.path.join(subdir, f"{subhead}.html"), "w") as f:
                        f.write(run_html)

                    run_html = create_run_page(run_headings, obs_run=head, toplist='_toplist')
                    with open(os.path.join(subdir, f"{subhead}_toplist.html"), "w") as f:
                        f.write(run_html)
            sub_headings += "</ul>"

    return run_headings, sub_headings

def get_html_string_week(root_dir, linepath=None, table_order=None):

    run_headings = ""
    sub_headings = ""
    if os.path.exists(root_dir):
        # head is the observing run
        for head in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, head)):
                run_headings += f'<a href="./{head}/{head}.html">{head}</a>'
                sub_headings += f"<h1> {head} </h1> <ul>"
                for subhead in os.listdir(os.path.join(root_dir, head)):
                    subdir = os.path.join(root_dir, head, subhead)
                    if os.path.isdir(subdir):
                        for weekrun in os.listdir(subdir):
                            weekdir = os.path.join(line_dir, head, subhead, weekrun)
                            if "week" in weekrun:
                                sub_headings += f'<l1><a href="./{head}/{subhead}/{weekrun}/{subhead}.html"> {subhead} {weekrun}</a></li> </br>'

                                if os.path.exists(os.path.join(weekdir, "table.json")):
                                    try:
                                        make_json_from_hdf5(weekdir, linepaths, table_order)
                                    except:
                                        print(f"WARNING: Cannot recreate json table, no new updates to {subhead}, {weekdir}")
                                else:
                                    make_json_from_hdf5(weekdir, linepaths, table_order)

                                run_html = create_run_page(line_run_headings, obs_run=head)
                                with open(os.path.join(weekdir, f"{subhead}.html"), "w") as f:
                                    f.write(run_html)
                            else:
                                sub_headings += f'<l1><a href="./{head}/{subhead}/{subhead}.html"> {subhead} </a></li> </br>'

                                if os.path.exists(os.path.join(subdir, "table.json")):
                                    try:
                                        make_json_from_hdf5(subdir, linepaths, table_order)
                                    except:
                                        print(f"WARNING: Cannot recreate json table, no new updates to {subhead}, {subdir}")
                                else:
                                    make_json_from_hdf5(subdir, linepaths, table_order)

                                run_html = create_run_page(line_run_headings, obs_run=head)
                                with open(os.path.join(subdir, f"{subhead}.html"), "w") as f:
                                    f.write(run_html)
                sub_headings += "</ul>"

    return run_headings, sub_headings

def make_summary_histogram(root_dir):
    """create a summary histogram of all statistics and save in root directory

    Args:
        root_dir (_type_): _description_
    """
    json_filename = os.path.join(root_dir, "table.json")

    with open(json_filename,"r") as f:
        data = json.load(f)

    band_ranges = [(20,500),(500,1000), (1000,1500), (1500,2000)]
    stats = {}
    for band in band_ranges:
        stats[band[0]] = np.array([(td["lineaware_stat"], td["fmin"]) for td in data if band[0] < td["fmin"] < band[1]])

    fig, ax = plt.subplots(nrows = 4, figsize = (4,10))
    for i, st in enumerate(band_ranges):
        hst = ax[i].hist(np.array(sorted(stats[st[0]], key=lambda x: x[0]))[10:-10, 0], bins = 100, label = f"{str(st)} Hz")
        ax[i].legend(fontsize="17")
        ax[i].set_ylabel("count", fontsize="17")
        #ax[i].set_xlabel(f"Viterbi statistic [{st[0]} - {st[1]} Hz]", fontsize="17")
        ax[i].set_xlabel(f"Line-aware statistic", fontsize="17")
        #ax[i].set_yscale("log")
    fig.tight_layout()

    fig.savefig(os.path.join(root_dir, "summary_histogram.png"))

def write_pages(cfg, root_dir, linepaths, table_order, force_overwrite=False, hwinjfile=None, obs_run="run"):
    """ Generate and write the html pages with the inputs from the directory structure"""
    
    make_directory_structure(root_dir)

    astro_dir = os.path.join(root_dir, "astrophysical")

    run_headings,sub_headings = get_html_string(astro_dir, linepaths=linepaths, table_order=table_order, force_overwrite=force_overwrite, hwinjfile=hwinjfile)

    line_dir = os.path.join(root_dir, "lines")

    #line_run_headings, line_sub_headings = get_html_string_week(line_dir, linepath=linepaths, table_order=table_order)
    line_run_headings,line_sub_headings = get_html_string(line_dir, linepaths=linepaths, table_order=table_order, force_overwrite=force_overwrite, hwinjfile=hwinjfile)

    # create pages
    home_html = create_home_page()
    usage_html = create_page_usage_page()
    astro_html = create_astro_page(run_headings, sub_headings)
    line_html = create_line_page(line_run_headings, line_sub_headings)


    # write pages
    with open(os.path.join(root_dir, "index.html"), "w") as f:
        f.write(home_html)

    with open(os.path.join(root_dir, "page_usage.html"), "w") as f:
        f.write(usage_html)

    with open(os.path.join(astro_dir, "astropage.html"), "w") as f:
        f.write(astro_html)

    with open(os.path.join(line_dir, "linepage.html"), "w") as f:
        f.write(line_html)

 

def main():
    import argparse
    from .soap_config_parser import SOAPConfig
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-c', '--config-file', help='config file', type=str, required=True, default=None)
    parser.add_argument('-o', '--out-path', help='top level of output directories', type=str)
    parser.add_argument('--force-overwrite', help='force overwrite tables', action=argparse.BooleanOptionalAction)
                                                   
    args = parser.parse_args()  

    if args.config_file is not None:
        if not os.path.isfile(args.config_file):
            raise FileNotFoundError(f"No File: {args.config_file}")
        cfg = SOAPConfig(args.config_file)

    else:
        #outpath,minfreq,maxfreq,obs_run="O3",vitmapmodelfname=None, spectmodelfname = None, vitmapstatmodelfname = None, allmodelfname = None, sub_dir = "soap",
        cfg = {"output":{}, "data":{}, "input":{}, }

    if args.out_path:
        cfg["output"]["save_directory"] = args.out_path

    if "lines_h1" in cfg["input"].keys():
        linepaths = [cfg["input"]["lines_h1"], cfg["input"]["lines_l1"]]
    else:
        linepaths = None
    if "hardware_injections" in cfg["input"].keys():
        hwinjfile = cfg["input"]["hardware_injections"]
    else:
        hwinjfile = None

    table_order = ["fmin", "fmax", "lineaware_stat", "H1_viterbistat", "L1_viterbistat", "CNN_vitmap_stat", "CNN_spect_stat", "CNN_vitmapstat_stat", "CNN_vitmapspect", "CNN_all_stat", "plot_path"]

    write_pages(cfg, os.path.dirname(os.path.normpath(cfg["output"]["save_directory"])), linepaths, table_order, force_overwrite=args.force_overwrite, hwinjfile=hwinjfile)

if __name__ == "__main__":
    main()
