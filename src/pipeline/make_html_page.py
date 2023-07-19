import os
import shutil
import h5py
import json
import numpy as np
from collections import OrderedDict
import importlib.resources as pkg_resources
import importlib_resources
import pandas as pd

def make_directory_structure(root_dir):

    my_resources = importlib_resources.files("pipeline")
    cssfile = (my_resources / "css"/ "general.css")
    javascriptfile = (my_resources / "scripts"/ "table_scripts.js")
    
    if not os.path.isdir(os.path.join(root_dir, "css")):
        os.makedirs(os.path.join(root_dir, "css"))

    if not os.path.isdir(os.path.join(root_dir, "scripts")):
        os.makedirs(os.path.join(root_dir, "scripts"))

    shutil.copy(cssfile, os.path.join(root_dir, "css/"))
    shutil.copy(javascriptfile, os.path.join(root_dir, "scripts/"))
    """
    shutil.copy("../css/general.css", os.path.join(root_dir, "css/"))
    shutil.copy("../scripts/table_scripts.js", os.path.join(root_dir, "scripts/"))
    """

def create_page_usage_page():
    html = ''' <!DOCTYPE html>
    <html lang="en">
    <head>
    <title>Usage</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="css/general.css" media="screen" type="text/css">
    </head>
    <body>

    <div class="navbar">
    <a href="index.html" class="active">Home</a>
    <a href="astrophysical/astropage.html">Astrophysical Search</a>
    <a href="lines/linepage.html">Line Search</a>
    <a href=""https://git.ligo.org/joseph.bayley/soapcw"" class="right">Code</a>
    <a href="page_usage.html" class="right">About/Usage</a>
    </div>

    <div class="row">
    <div class="side">
        <h2>SOAP</h2>
        <div><img style="width:50%;margin-right:auto;margin-left:auto;display:block" src="https://git.ligo.org/joseph.bayley/soapcw/-/raw/master/logo/drawing.png" alt="SOAP logo"/></div>
        <p>The SOAP search is a rapid search for continuous gravitational waves using the Viterbi algorithm</p>
        <h3>Links</h3>
        <p>GitLab: <a href="https://git.ligo.org/joseph.bayley/soapcw">https://git.ligo.org/joseph.bayley/soapcw</a></p>
        <p><a href="https://joseph.bayley.docs.ligo.org/soapcw/index.html">Documentation</a></p>

        <p><a href="https://journals.aps.org/prd/abstract/10.1103/PhysRevD.100.023006">Methods Paper</a></p>
        <p><a href="https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.083024">CNN followup paper</a></p>
        <p><a href="https://journals.aps.org/prd/abstract/10.1103/PhysRevD.106.083022">Parameter estimation paper</a></p>
    </div>
    <div class="main">
        <h2>How to use these pages</h2>
        The run pages are built around a table of each of the frequency bands that are run on and an image showing the output spectrograms and viterbi tracks along with other information.
    
        To filter which frequency bands to use the box to the top left gives a set of options

        <img style="width:30%;margin-right:auto;margin-left:auto;display:block" src="https://git.ligo.org/joseph.bayley/soapcw/-/raw/master/pipeline/images/options.png" />
        <ul>
        <li> In the first box one can sort the table based on one of the selected headings in ascending or descending order. </li>
        <li> In the second box there is a choice of a number of variables, checking these will include them in the table. </li>
        <li> The third box filters the frequency ranges, input the ranges and clikc filter to change the frequency range </li>
        <li> The fourth box has two buttons (last page, next page), these move between the table pages forwards and backwards </li>
        <ul>

        <img style="width:30%;margin-right:auto;margin-left:auto;display:block" src="https://git.ligo.org/joseph.bayley/soapcw/-/raw/master/pipeline/images/table.png" />

        The table then shows a list of different frequency bands defined by the fmin and fmax.
        By default it shows the lineaware stat, which is the line aware statistic output from the SOAP search, and the info column. 
        The info column contains extra information about which is known about that band, i.e. info from line investigations or hw injections

        By clicking on one of these cells it shows a plot and a table containing all of the information about this band.
        The table of information looks like this:

        <img style="width:100%;margin-right:auto;margin-left:auto;display:block" src="https://git.ligo.org/joseph.bayley/soapcw/-/raw/master/pipeline/images/band_summary.png" />

        Which contains all table columns. The hide track button is used to toggle the viterbi track in the plot. 
        The previous and next buttons cycle through the table in order.

        <img style="width:60%;margin-right:auto;margin-left:auto;display:block" src="https://git.ligo.org/joseph.bayley/soapcw/-/raw/master/pipeline/images/plot.png" />

        The plot then contains a number of panels showing information about the band.
        <ul>
        <li> The first panel shows the time-frequency spectrogram from L1 with the viterbi track overlayed </li>
        <li> The second panel shows the time-frequency spectrogram from H1 with the viterbi track overlayed </li>
        <li> The third panel shows the viterbi map from the combination of both detectors, this is related to the probability that a signal is present in any particular time-frequency bin </li>
        <li> The fourth panel shows the normalised SFT power along the red Viterbi track </li>
        <li> The fifth panel shows the mean noise floor across the frequency band as a function of time. </li>
        <ul>
    </div>
    </div>

    <div class="footer">
        <p>If you have any problems with this page please submit an issue to: https://git.ligo.org/joseph.bayley/soapcw/-/issues or contact: joseph.bayley@glasgow.ac.uk </p>

    </div>

    </body>
    </html>'''
    return html


def create_home_page():


    html = ''' <!DOCTYPE html>
    <html lang="en">
    <head>
    <title>SOAP</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="css/general.css" media="screen" type="text/css">
    </head>
    <body>

    <div class="navbar">
    <a href="index.html" class="active">Home</a>
    <a href="astrophysical/astropage.html">Astrophysical Search</a>
    <a href="lines/linepage.html">Line Search</a>
    <a href="https://git.ligo.org/joseph.bayley/soapcw" class="right">Code</a>
    <a href="page_usage.html" class="right">About/Usage</a>
    </div>

    <div class="row">
    <div class="side">
        <h2>SOAP</h2>
        <div><img style="width:50%;margin-right:auto;margin-left:auto;display:block" src="https://git.ligo.org/joseph.bayley/soapcw/-/raw/master/logo/drawing.png" alt="SOAP logo"/></div>
        <p>The SOAP search is a rapid search for continuous gravitational waves using the Viterbi algorithm</p>
        <h3>Links</h3>
        <p>GitLab: <a href="https://git.ligo.org/joseph.bayley/soapcw">https://git.ligo.org/joseph.bayley/soapcw</a></p>
        <p><a href="https://joseph.bayley.docs.ligo.org/soapcw/index.html">Documentation</a></p>

        <p><a href="https://journals.aps.org/prd/abstract/10.1103/PhysRevD.100.023006">Methods Paper</a></p>
        <p><a href="https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.083024">CNN followup paper</a></p>
        <p><a href="https://journals.aps.org/prd/abstract/10.1103/PhysRevD.106.083022">Parameter estimation paper</a></p>
    </div>
    <div class="main">
        <h2>General info about SOAP</h2>
        <img style="width:70%" src="https://git.ligo.org/joseph.bayley/soapcw/-/raw/master/pipeline/images/vitmap_ex.png" />
        <br>
        <h2>Astrophysical search</h2>
        <p>The astrophysical search runs using sets of 1800s SFTs summed over one day and combining each detector using the <a href="https://joseph.bayley.docs.ligo.org/soapcw/bayesianlineaware.html">line aware statistic</a></p>
        <h2>Line search</h2>
        <p>The line search also runs on sets of 1800s SFTs summed over one day, however are the search is run separately for each detector to identify instrumental artefacts within the data.</p>


    </div>
    </div>

    <div class="footer">
        <p>If you have any problems with this page please submit an issue to: https://git.ligo.org/joseph.bayley/soapcw/-/issues or contact: joseph.bayley@glasgow.ac.uk </p>

    </div>

    </body>
    </html>'''
    return html

def create_astro_page(run_headings, sub_headings):

    html = f'''<!DOCTYPE html>
            <html lang="en">
            <head>
            <title>Astro Page</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link rel="stylesheet" href="../css/general.css" media="screen" type="text/css">
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js"></script>
            </head>
            <body>


            <div class="navbar">
            <a href="../index.html" class="active">Home</a>
            <a href="../astrophysical/astropage.html">Astrophysical Search</a>
            <a href="../lines/linepage.html">Line Search</a>

            <a></a>
            {run_headings}

            <a href="https://git.ligo.org/joseph.bayley/soapcw" class="right">Code</a>
            <a href="../page_usage.html" class="right">About/Usage</a>
            </div>


            <div class="row">
            <div class="side">
                <h2>About</h2>
                <div><img style="width:50%;margin-right:auto;margin-left:auto;display:block" src="https://git.ligo.org/joseph.bayley/soapcw/-/raw/master/logo/drawing.png" alt="SOAP logo"/></div>
                    <p>The SOAP search is a rapid search for continuous gravitational waves using the Viterbi algorithm</p>
                    <h3>Links</h3>
                    <p>GitLab: <a href="https://git.ligo.org/joseph.bayley/soapcw">https://git.ligo.org/joseph.bayley/soapcw</a></p>
                    <p><a href="https://joseph.bayley.docs.ligo.org/soapcw/index.html">Documentation</a></p>

                    <p><a href="https://journals.aps.org/prd/abstract/10.1103/PhysRevD.100.023006">Methods Paper</a></p>
                    <p><a href="https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.083024">CNN followup paper</a></p>
                    <p><a href="https://journals.aps.org/prd/abstract/10.1103/PhysRevD.106.083022">Parameter estimation paper</a></p>
            </div>
            <div class="main">
                <h2>List of runs with information</h2>
                <p>For each of the observing runs there were multiple runs of SOAP with different settings, the recommended one to use is at the top of the list.</p>
                {sub_headings}
            </div>
            </div>

            <div class="footer">
                <p>If you have any problems with this page please submit an issue to: https://git.ligo.org/joseph.bayley/soapcw/-/issues or contact: joseph.bayley@glasgow.ac.uk </p>
            </div>

            </body>

            </html>'''

    return html

def create_line_page(run_headings, sub_headings):

    html = f'''<!DOCTYPE html>
            <html lang="en">
            <head>
            <title>Line Page</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link rel="stylesheet" href="../css/general.css" media="screen" type="text/css">
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js"></script>
            </head>
            <body>


            <div class="navbar">
            <a href="../index.html" class="active">Home</a>
            <a href="../astrophysical/astropage.html">Astrophysical Search</a>
            <a href="../lines/linepage.html">Line Search</a>

            <a></a>
            {run_headings}

            <a href="https://git.ligo.org/joseph.bayley/soapcw" class="right">Code</a>
            <a href="../page_usage.html" class="right">About/Usage</a>
            </div>


            <div class="row">
            <div class="side">
                <h2>About</h2>
                <div><img style="width:50%;margin-right:auto;margin-left:auto;display:block" src="https://git.ligo.org/joseph.bayley/soapcw/-/raw/master/logo/drawing.png" alt="SOAP logo"/></div>
                    <p>The SOAP search is a rapid search for continuous gravitational waves using the Viterbi algorithm</p>
                    <h3>Links</h3>
                    <p>GitLab: <a href="https://git.ligo.org/joseph.bayley/soapcw">https://git.ligo.org/joseph.bayley/soapcw</a></p>
                    <p><a href="https://joseph.bayley.docs.ligo.org/soapcw/index.html">Documentation</a></p>

                    <p><a href="https://journals.aps.org/prd/abstract/10.1103/PhysRevD.100.023006">Methods Paper</a></p>
                    <p><a href="https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.083024">CNN followup paper</a></p>
                    <p><a href="https://journals.aps.org/prd/abstract/10.1103/PhysRevD.106.083022">Parameter estimation paper</a></p>
            </div>
            <div class="main">
                <h2>List of runs with information</h2>
                <p>SOAP is run separately for each detector, without the Baysian line aware statistic, using just the summed power as a statistic.</p>
                {sub_headings}
            </div>
            </div>

            <div class="footer">
                <p>If you have any problems with this page please submit an issue to: https://git.ligo.org/joseph.bayley/soapcw/-/issues or contact: joseph.bayley@glasgow.ac.uk </p>
            </div>

            </body>

            </html>'''

    return html



def create_run_page(run_headings, obs_run="run"):


    html = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <title>{obs_run}</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="../../../css/general.css" media="screen" type="text/css">

    <link href="https://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css" rel="stylesheet">   
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js"></script>
    <link rel="stylesheet" href="https://cdn.datatables.net/1.10.2/css/jquery.dataTables.min.css"></style>
    <script type="text/javascript" src="https://cdn.datatables.net/1.10.2/js/jquery.dataTables.min.js"></script>
    <script type="text/javascript" src="https://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js"></script>


    </head>

    <body>

    <div class="navbar">
    <a href="../../../index.html" class="active">Home</a>
    <a href="../../../astrophysical/astropage.html">Astrophysical Search</a>
    <a href="../../../lines/linepage.html">Line Search</a>
    <a></a>
    {run_headings}
    <a href="https://git.ligo.org/joseph.bayley/soapcw" class="right">Code</a>
    <a href="../../../page_usage.html" class="right">About/Usage</a>
    </div>

    <div class="row">
    <div class="side"> 

        <div style="display:flex">
            <button type="button" class="btn btn-cl" style="width:100%; margin:3px" id="filterdivbutton" onclick="filterDivs(event)"> Hide filters</button>
        </div>
        <div class="box" id="sortdiv">
        Sort table by: 
        <select name="Sortby" id="sortby" key="fmin" onchange=reloadPage(event)></select>
        <select name="Sortorder" id="sortorder" key="descending" onchange=reloadPage(event)>
            <option key="ascending">Ascending</option>
            <option key="descending">Descending</option>
        </select>
        </div>

        <div class="box" id="checkboxdiv">
        Include column in table:
        <form id="checkboxform">
        </form>
        </div>

        <div class="box" id="filterdiv">
        Filter frequencies: 
        </br>
        Min: <input type="text" id="minfreq" name="minfreq" size="4" maxlength="6" value="0"> Hz ----
        Max: <input type="text" id="maxfreq" name="maxfreq" size="4" maxlength="6" value="2000"> Hz 
        </br>
        <button type="button" style="margin:3px" class="btn btn-cl py-3" id="filtertable" onclick="reloadPage(event)"> Filter</button>
        <button type="button" class="btn btn-cl py-3" id="filtertablereset" onclick="resetTable(event)"> Reset</button>
        </div>
        <div class="box" id="filterlinediv">
        Filter Info: 
        </br>
        <input type="checkbox" id="onlyhwinjs" name="onlyhwinjs" value="1" onclick=reloadPage(event)> Only Hw injs</input>
        <input type="checkbox" id="onlyknownlines" name="onlyknownlines" value="1" onclick=reloadPage(event)> Only known lines</input>
        </br>
        <input type="checkbox" id="hidehwinjs" name="hidehwinjs" value="1" onclick=reloadPage(event)> Hide Hw injs</input>
        <input type="checkbox" id="hideknownlines" name="hideknownlines" value="1" onclick=reloadPage(event)> Hide known lines</input>
        </div>
        <div class="box" id="showalldiv">
        <button type="button" class="btn btn-cl py-3" id="lastpagebutton" value=0 onclick="lastTablePage(event)"> Last page</button>
        <button type="button" class="btn btn-cl py-3" id="nextpagebutton" value=0 onclick="nextTablePage(event)"> Next page</button>
        </div>

        <table cellpadding="4" border="1" class="table table-bordered table-striped" id="inTable" > 
        </table>
    </div>

    <div class="main">
        <div> 
        <button type="button" style="margin:3px" class="btn btn-cl" id="trackbutton" value="0" onclick="showHideTrack(event)">Hide track</button> 
        <button type="button" style="margin:3px" class="btn btn-cl py-3" id="prevbutton" value="0" onclick="previousImage(event)">Previous</button>
        <button type="button" style="margin:3px" class="btn btn-cl py-3" id="nextbutton" value="0" onclick="nextImage(event)">Next</button> 
        
        <table cellpadding="4" border="1" id="infoTable" class="table table-striped table-bordered"> 
        </table>
        </div>

        <div class="imagewrap"> 
        <img src="" id="image" value="0"></img> 
        </div>
    </div>
    </div>

    <div class="footer">
    <p>If you have any problems with this page please submit an issue to: https://git.ligo.org/joseph.bayley/soapcw/-/issues or contact: joseph.bayley@glasgow.ac.uk </p>
    </div>

    <script type="text/javascript" src="../../../scripts/table_scripts.js"></script>

    </body>


    </html>
    '''

    # button to be added to show all of the frequency bands
    """ <div class="box" id="showalldiv">
        <button type="button" class="btn btn-cl py-3" id="showallbutton" onclick="loadAllBands(event)"> Load all bands</button>
        </div>"""

    return html

def create_line_run_page(run_headings, obs_run="run"):


    html = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <title>{obs_run}</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="../../../../css/general.css" media="screen" type="text/css">

    <link href="https://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css" rel="stylesheet">   
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js"></script>
    <link rel="stylesheet" href="https://cdn.datatables.net/1.10.2/css/jquery.dataTables.min.css"></style>
    <script type="text/javascript" src="https://cdn.datatables.net/1.10.2/js/jquery.dataTables.min.js"></script>
    <script type="text/javascript" src="https://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js"></script>


    </head>

    <body>

    <div class="navbar">
    <a href="../../../index.html" class="active">Home</a>
    <a href="../../../astrophysical/astropage.html">Astrophysical Search</a>
    <a href="../../../lines/linepage.html">Line Search</a>
    <a></a>
    {run_headings}
    <a href="https://git.ligo.org/joseph.bayley/soapcw" class="right">Code</a>
    <a href="../../../page_usage.html" class="right">About/Usage</a>
    </div>

    <div class="row">
    <div class="side"> 

        <div style="display:flex">
            <button type="button" class="btn btn-cl" style="width:100%; margin:3px" id="filterdivbutton" onclick="filterDivs(event)"> Hide filters</button>
        </div>
        <div class="box" id="sortdiv">
        Sort table by: 
        <select name="Sortby" id="sortby" key="fmin" onchange=reloadPage(event)></select>
        <select name="Sortorder" id="sortorder" key="descending" onchange=reloadPage(event)>
            <option key="ascending">Ascending</option>
            <option key="descending">Descending</option>
        </select>
        </div>

        <div class="box" id="checkboxdiv">
        Include column in table:
        <form id="checkboxform">
        </form>
        </div>

        <div class="box" id="filterdiv">
        Filter frequencies: 
        </br>
        Min: <input type="text" id="minfreq" name="minfreq" size="4" maxlength="6" value="0"> Hz ----
        Max: <input type="text" id="maxfreq" name="maxfreq" size="4" maxlength="6" value="2000"> Hz 
        </br>
        <button type="button" style="margin:3px" class="btn btn-cl py-3" id="filtertable" onclick="reloadPage(event)"> Filter</button>
        <button type="button" class="btn btn-cl py-3" id="filtertablereset" onclick="resetTable(event)"> Reset</button>
        </div>
        <div class="box" id="showalldiv">
        <button type="button" class="btn btn-cl py-3" id="lastpagebutton" value=0 onclick="lastTablePage(event)"> Last page</button>
        <button type="button" class="btn btn-cl py-3" id="nextpagebutton" value=0 onclick="nextTablePage(event)"> Next page</button>
        </div>

        <table cellpadding="4" border="1" class="table table-bordered table-striped" id="inTable" > 
        </table>
    </div>

    <div class="main">
        <div> 
        <button type="button" style="margin:3px" class="btn btn-cl" id="trackbutton" value="0" onclick="showHideTrack(event)">Hide track</button> 
        <button type="button" style="margin:3px" class="btn btn-cl py-3" id="prevbutton" value="0" onclick="previousImage(event)">Previous</button>
        <button type="button" style="margin:3px" class="btn btn-cl py-3" id="nextbutton" value="0" onclick="nextImage(event)">Next</button> 
        
        <table cellpadding="4" border="1" id="infoTable" class="table table-striped table-bordered"> 
        </table>
        </div>

        <div class="imagewrap"> 
        <img src="" id="image" value="0"></img> 
        </div>
    </div>
    </div>

    <div class="footer">
    <p>If you have any problems with this page please submit an issue to: https://git.ligo.org/joseph.bayley/soapcw/-/issues or contact: joseph.bayley@glasgow.ac.uk </p>
    </div>

    <script type="text/javascript" src="../../../scripts/table_scripts.js"></script>

    </body>


    </html>
    '''

    # button to be added to show all of the frequency bands
    """ <div class="box" id="showalldiv">
        <button type="button" class="btn btn-cl py-3" id="showallbutton" onclick="loadAllBands(event)"> Load all bands</button>
        </div>"""

    return html


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
            (hwinjtable["f0 (epoch start)"] < fhigh) & 
            (hwinjtable["f0 (epoch start)"] > flow)]

    info = ""
    for index, line in hwinjs.iterrows():
        info += f"<a href='https://ldas-jobs.ligo.caltech.edu/~keith.riles/cw/injections/preO3/preO3_injection_params.html'> hwinj: {line['Pulsar']}</a>"
    
    return info

def make_json_from_hdf5(root_dir, linepaths=None, table_order=None, hwinjfile=None):
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

    # also save separate list of just the top statistics
    with open(os.path.join(root_dir, "table_toplist.json"), "w") as f:
        json.dump(sorted_json_data[:int(0.1*len(sorted_json_data))], f)

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
                            except:
                                print(f"WARNING: Cannot recreate json table")
                        else:
                            print(f"WARNING: No new updates to {subhead}, {subdir}")
                    else:
                        make_json_from_hdf5(subdir, linepaths, table_order, hwinjfile=hwinjfile)

                    run_html = create_run_page(run_headings, obs_run=head)
                    with open(os.path.join(subdir, f"{subhead}.html"), "w") as f:
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

                                run_html = create_line_run_page(line_run_headings, obs_run=head)
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

                                run_html = create_line_run_page(line_run_headings, obs_run=head)
                                with open(os.path.join(subdir, f"{subhead}.html"), "w") as f:
                                    f.write(run_html)
                sub_headings += "</ul>"

    return run_headings, sub_headings

def write_pages(root_dir, linepaths, table_order, force_overwrite=False, hwinjfile=None, obs_run="run"):
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

    parser.add_argument('-c', '--config-file', help='config file', type=str, required=False, default=None)
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

    table_order = ["fmin", "fmax", "lineaware_stat", "H1_viterbistat", "L1_viterbistat", "CNN_vitmap_stat", "CNN_spect_stat", "CNN_vitmapstat_stat", "CNN_all_stat", "plot_path"]

    write_pages(os.path.dirname(os.path.normpath(cfg["output"]["save_directory"])), linepaths, table_order, force_overwrite=args.force_overwrite, hwinjfile=hwinjfile)

if __name__ == "__main__":
    main()