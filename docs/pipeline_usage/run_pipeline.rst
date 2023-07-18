.. highlight:: shell

============
Pipeline usage
============

SOAP is usually run as a pipeline on the LIGO detectors, 
both searching for astrophysical signals and as a detector characterisation tool.
These are referenced as the astrophysical searches and the line searches.

These tools were designed to run on the LIGO clusters with access to LIGO data, however may still work elsewhere.

Astrophysical search
------------

Once installed there are two scripts thats can be used to run the pipeline, first the astrophysical search:

.. code-block:: console

    $ soapcw-run-soap-astro

is the command which has a number of options. The majority of options are stored in a configuration files which might look something like below. 
Note that currently this configuration file is slightly different from that of the machine learning config used in the cnn and Neville despite having similar inputs. 

.. code-block:: console

    [general]
    root_dir = /home/joseph.bayley/repositories/soap/soap_pipeline/summary_pages/astrophysical/O3/
    temp_dir = /home/joseph.bayley/projects/soap_temp/

    [condor]
    memory = 5000
    request_disk = 5000
    accounting_group = ligo.dev.o4.cw.explore.test
    n_jobs = 100
    band_load_size = 8.0

    [input]
    load_directory = [/hdfs/frames/O3/pulsar/sfts/tukeywin/L1_C01_Gated_1800s/, /hdfs/frames/O3/pulsar/sfts/tukeywin/H1_C01_Gated_1800s/]
    hard_inj = /home/joseph.bayley/projects/soap_summary_pages/data/o3/
    lines_h1 = /home/joseph.bayley/projects/soap_summary_pages/data/o3/O3H1lines.txt
    lines_l1 = /home/joseph.bayley/projects/soap_summary_pages/data/o3/O3L1lines.txt

    [data]
    band_starts = [40,500,1000, 1500]
    band_ends   = [500,1000,1500,2000]
    band_widths = [0.1,0.2,0.3,0.4]
    strides     = [1,2,3,4]
    obs_run     = O3
    n_summed_sfts = 48

    [lookuptable]
    type = power
    lookup_dir = /home/joseph.bayley/data/soap/lookup_tables/line_aware_optimised/
    snr_width_line = 4
    snr_width_signal = 10
    prob_line = 0.4

    [transitionmatrix]
    left_right_prob = 1.000000001
    det1_prob = 1e400
    det2_prob = 1e400

    [cnn]
    vitmapmodel_path = none
    spectmodel_path = none
    vitmapstatmodel_path = none
    allmodel_path = none

    [output]
    save_directory = /home/joseph.bayley/public_html/soap/astrophysical/
    sub_directory = soap_C01_gated_suppressed_line_doublestep

    [scripts]
    search_exec=soapcw-run-soap-astro
    html_exec=soapcw-make-html-pages

There are a number of available options for this in the command line, which will overwrite the config file, however, the important ones are:

.. code-block:: console
    '-s', '--start-freq' 
    '-e', '--end-freq'
    '-w', '--band-width'
    '--stride'

These set the narrow band widths with band width and define between which frequencies to run the search. 
The stride refers to how much overlap there will be between the bands (astrophysical searches usually overlap by 1/2 thge bandwidth)

Generally this is a lot of data to run over, so it is run using condor to manage the job submission. The submit and dag files can be created from the config file using this scripts

.. code-block:: console
    soapcw-make-dag-files-astro -c config_file.ini

Then the appropriate dag file can be submitted.


Line search 
-----------

The line search is set up in a similar way to the astro search however uses a slightly different config file and job dag file creation.

.. code-block:: console

    $ soapcw-run-soap-lines


.. code-block:: console

    $ soapcw-make-dag-files-lines -c config_file.ini

The ini file may look something like

.. code-block:: console
    [general]
    # root directory for the run and condor files
    root_dir = /path/to/run/files
    temp_dir = /path/to/temp/dir

    [condor]
    # memory and disk requirements to set via condor
    memory = 5000
    request_disk = 5000
    # the accounting group for the job to run under
    accounting_group = accounting.group
    # size of the frequency band to split the jobs into 
    band_load_size = 8.0

    [input]
    # directory where the detector SFTs are stored
    load_directory = [/path/to/H1/sfts, /path/to/L1/sfts]
    # path to the hardware injection file, stored as a pandas table
    hard_inj = /path/to/hardware/injection/file.
    lines_h1 = /path/to/h1/lines/file.
    lines_l1 = /path/to/l1/lines/file.

    [data]
    # start and end frequencies 
    # this can be split up into multiple bands wach which have a different bandwidth
    # and stride (summing of frequency bins)
    band_starts = [20,500,1000, 1500]
    band_ends   = [500,1000,1500,2000]
    band_widths = [0.1,0.2,0.3,0.4]
    strides     = [1,2,3,4]
    # which observing run for labeling
    obs_run     = O4
    # number of sfts to sum over (default 1 day)
    n_summed_sfts = 48

    [lookuptable]
    # which type of lookup table to use 
    # power finds the best coincident SNR between detectors
    # amplitude finds the best coincident Amplitude between detectors accounting for duty cycle etc (default power)
    lookup_type = power
    # path to where the lookup tables were generated
    lookup_dir = /path/to/lookup/tables
    # prior width of the line SNR distribution
    snr_width_line = 4
    # prior width of the signal SNR distribution
    snr_width_signal = 10
    # ratio of probabilities of the line and noise model
    prob_line = 0.4

    [transitionmatrix]
    # probabilty of jumping up or down in frequency
    left_right_prob = 1.000000001
    # probability of detector 1 being a bin away from geocenter 
    det1_prob = 1e400
    # probability of detector 2 being a bin away from geocenter 
    det2_prob = 1e400

    [cnn]
    # this assumes already pretrained models (all can be set to none if not needed (default))
    vitmapmodel_path = /path/to/vitmapmodel.pt
    spectmodel_path = none
    allmodel_path = none
    vitmapmodel_path = none
    spectmodel_path = none
    vitmapstatmodel_path = none
    allmodel_path = none

    [output]
    # directory to save outputs and make the html pages
    save_directory = /home/joseph.bayley/public_html/soap/astrophysical/
    # sub directory to put the results in if run on the same detector
    sub_directory = soap_C00_gated_suppressed_line_doublestep
    # overwrite the existin results in this directory (default false)
    overwrite_files = False

    [scripts]
    # executables to run the search with and to generate the html pages
    search_exec=soapcw-run-soap-astro
    html_exec=soapcw-make-html-pages


Making html pages
------------------
If the pages are not made automatically, they can be manually made using the html page script

.. code-block:: console
    $ soapcw-make-html-pages -c config.ini




.. _Github repo: https://git.ligo.org/joseph.bayley/soapcw
