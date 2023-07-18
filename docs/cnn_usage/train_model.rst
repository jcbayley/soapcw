============
CNN usage
============

SOAP is usually run as a pipeline on the LIGO detectors, 
both searching for astrophysical signals and as a detector characterisation tool.

The machine learning part of this method, mostly using CNNs to reduce the impact of instrumental artefacts, and how to use it is described below.

These tools were designed to run on the LIGO clusters with access to LIGO data, however may still work elsewhere.

Training a model
------------

Once the data has been generated, its time to train a model on this data. 
There are a number of ways to do this, however, we will use the same configuration file as on the data_generation page.

.. code-block:: console
    [general]
    save_dir = /home/user/data/soap/cnn/o3/test_o3/             # directory to output run files to
    sft_dirs = /dir/to/raw/H1/sfts/, /dir/to/raw/L1/sfts/       # directory to the raw LIGO SFTs
    narrowband_sft_dir = /dir/to/output/narrowband/sfts         # directory to output the narrowbanded SFTs

    [condor]
    accounting_group = accounting.tag                           # accounting tag for ligo cluster
    data_load_size = 4.1                                        # size of frequency band to assign to each job
    root_dir = /root/dir/for/condor/files                       # directory for the condor sub/dag files to go

    [data]
    band_starts = [20]                                          # start frequency of each larger band
    band_ends   = [500]                                         # end frequency of each larger band
    band_widths = [0.1]                                         # band width for each larger band
    strides     = [1]                                           # number of frequency bins to average for each larger band
    resize_image = false                                        # interpolate output spectrograms and vitmaps to different size
    run = o3                                                    # run label [gauss, o1, o2, o3, o4 ...]
    type = train                                                # type of data to generate (only train is needed due to odd/even bands)
    snrmin= 40                                                  # start SNR for injected signals
    snrmax= 200                                                 # end SNR for injected signals
    n_summed_sfts = 48                                          # Number of SFTs to sum over (48 default as 1 day for 1800s SFTs)
    save_options=[                                              # which data products to save 
        "vit_imgs",                                             # viterbi maps
        "H_imgs",                                               # H1 spectrograms
        "L_imgs",                                               # L1 spectrograms
        "stats",                                                # viterbi statistic
        "pars",                                                 # signal/data parameters
        "paths",                                                # viterbi tracks
        "powers"]                                               # Spectrogram power along track
    nperband = 10                                               # how many times to repeat injections per sub-band (only for gauss data type)
    gen_noise_only = True                                       # with real data, generate noise only as well as injected bands
    tstart = 1238166018                                         # start time of observation
    tend = 1269363618                                           # end time of observation

    [lookuptable]
    type = power                                                # lookuptable type (power, amplitude)                   
    lookup_dir = /path/to/save/lookuptable                      # where to save lookup tables
    snr_width_line = 4                                          # width on SNR prior for line model
    snr_width_signal = 10                                       # width on SNR prior for signal model
    prob_line = 0.4                                             # ratio of probability of line to noise model

    [model]
    model_type = "vitmapspectrogram"                            # which data products to use (viterbi maps and spectrograms)
    save_dir = /path/to/save/model/outputs                      # where to save model outputs
    learning_rate = 1e-4                                        # learning rate of adam optimiser
    img_dim = (180, 362)                                        # size of the default input image (spectrogram)
    conv_layers = [(32, 8, 2, 1),(32, 8, 2, 1),]                # convolutional layers (nfilters, filtersize, n_maxpool, stride)
    avg_pool_size = 5                                           # nxn grid size to average pool to after convolutions
    fc_layers = [64,32,2]                                       # list of fully connected layers 
    n_epochs=100                                                # epochs to train for
    n_train_multi_size=30                                       # when using avg_pool_size, train on many different size inputs (this is number of different sizes)
    save_interval=2                                             # how many epochs to save model after
    band_types = even, odd                                      # which models to train (one for even and one for odd)

    [code]
    search_exec=soapcw-cnn-make-data                            # script to run when making data (does not need to be changed)


.. code-block:: console
    $ soapcw-cnn-train-model -c config.ini 