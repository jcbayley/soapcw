.. highlight:: shell

============
Neville model
============

Creating a Model
-----------

A CVAE model can be created by calling the function below. The example below creates a model that predicts 4 parameters. this is the standard setup to predice the four doppler parameters of a CW source, also this takes in 362 time steps at 1 day long steps.  

.. code-block:: python
   
	neville_model = soapcw.neville.CVAE(
	      input_dim=362,                         # input length of timeseries
	      par_dim=4,                             # number of parameters to estimate (4 doppler paramaeters)
	      latent_dim=16,                         # size of the latent dimension
	      fc_layers = [128,128,128],             # fully connected MLP layers
	      conv_layers = [(4,4,2,1), (4,4,1,1)],  # convolutional layers (num filters, filter size, maxpool size, stride)
	      device="cpu",                          # which device to put the model on
	      fdim=0,                                # number of extra dimensions to add after convolutional layers (default is one reffering to base frequency of band)
	      inchannels = 1,                        # number of channels input
	      dist_type = "gaussian"                 # type of distribution to use in the latent space (default gaussian)
	)

	
Loading a Model
-----------

In this demonstration I will show how to load a pretrained model into Neville using a config file and some pretrained weights.


.. code-block:: python

	neville_model, neville_config = soapcw.neville.load_models.load_model_from_config(
	                                           config_file,     # path to configuration file - example below
						   neville_file,    # path to the model weights
						   device=device)

.. code-block:: console
		
		[general]
		save_dir = /path/to/save/directory
		sft_dirs = /path/to/H1/sfts, /path/to/L1/sfts
		narrowband_sft_dir = /path/to/narrowbanded/sfts/

		[condor]
		accounting_group = accounting.group.
		data_load_size = 4.1                              # size of frequency band to load for a single job
		root_dir = /path/to/condor/files                  # root directory to save the condor files

		[data]	
		band_starts = [20]                                # list of start frequencies for band
		band_ends   = [500]                               # list of end frequencies for band
		band_widths = [0.1]                               # list of widths of bands
		strides     = [1]                                 # list of strides to use for each band
		resize_image = false                              # resize the spectrogram/vitmap image
		run = o3                                          # observing run name
		type = train                                      # type of data to save
		snrmin= 40                                        # minimum SNR for injections
		snrmax= 200                                       # maximum SNR for injections
		n_summed_sfts = 48                                # number of SFTs to sum
		save_options=["pars","paths"]                     # only need the paths and parameters for training neville
		nperband = 10                                     # number of times to repeat injections in a band
		gen_noise_only = True                             # generate bands without any injections in
		tstart = 1238166018                               # start time of search
		tend = 1269363618                                 # end time of search
		
		[lookuptable]
		type = power                                      # type of lookup statistic to use power or amplitude
		lookup_dir = /path/to/lookup/tables               # path to load lookup table from
		snr_width_line = 4                                # width of prior of SNR for line model
		snr_width_signal = 10                             # width of prior of SNR for signal model
		prob_line = 0.4                                   # ratio of model probabilities for line and noise
		
		[model]
		model_type = "vitmapspectrogram"                  # which data types to use
		save_dir = /directory/to/save/model/to
		learning_rate = 1e-4                              # learning rate for cnn training
		img_dim = (180, 362)                              # size of input img 
		conv_layers = [(32, 8, 2, 1),]                    # convolutional layersin model (num filter, filter size, max pool size, stride)
		avg_pool_size = [10,2]                            # size of addaptive pooling layer
		fc_layers = [64,32,2]                             # fully connected mlp layers, 
		n_epochs=100                                      # number of epochs to train cnn for
		n_channels = 3                                    # number of channels as input to cnn
		n_train_multi_size=30                             # number of repeated training steps on subset of data
		save_interval=2                                   # interval of epochs to save cnn model
		band_types = even, odd                            # which band types to run one (odd or even)
		
		[neville]
		save_dir = /path/to/save/neville/model
		learning_rate = 1e-4                              # learning rate for neville training
		input_dim = 362                                   # input dimension for cvae model
		num_predict_params = 4                            # number of parameters to predict (4 for doppler parameters)
		latent_dim = 16                                   # size of cvae latent dimension
		conv_layers = [(4, 4, 1, 4), (4, 4, 1, 4)]        # convolutional layers in model (num filters, filter size, max pool size, stride)
		fc_layers = [128, 128, 128, 128]                  # fully connected mlp layers
		fdim=1                                            # dimensions to add after convolutional layer, default it 1 to add fmin of band 
		n_channels=1                                      # number of channels input to cvae
		dist_type = "gaussian"                            # distribution to use in latent dimension of cvae
		
		[code]
		search_exec=soapcw-cnn-make-data                  # exec to create data


Using the Model
--------------
For testing the model has a test function that will generate samples from the posterior distribution

.. code-block:: python

	neville_model.eval()  # put the model in eval mode
	with torch.no_grad(): # dont compute gradients for this
    	samps = neville_model.test(
			tracks_input,             # torch tensor of input tracks (ntracks, nchannels, nsamples) 
			freqs=fmins,              # torch tensor of base frequency for sub-bands corresponding to tracks
			num_samples=5000,         # number of posterior samples to generate
			transform_func=None,      # function to transform parameters, see below
			return_latent = False,    # whether to return latent space samples also
			par=None                  # injected parameters so latent space from q dsitribution can be returned
			)

The output of this function are:
 - samps[0] = Normalised doppler parameters posterior samples 
 - samps[1] = Transformed doppler parameters posterior samples (only if a transform function has beed specified)
 - samps[2] = Track element posterior samples (i.e. samples from binomial distribution corresponding to probability that signal is consistent with that track element)

In return latent argument is used the outputs are:
 - samps[0] = Normalised doppler parameters posterior samples 
 - samps[1] = Transformed doppler parameters posterior samples (only if a transform function has beed specified)
 - samps[2] = samples from the latent space of the r encoder
 - samps[3] = samples from the latent space of the q encoder
 - samps[4] = Track element posterior samples (i.e. samples from binomial distribution corresponding to probability that signal is consistent with that track element)


If using the transform function it should take in the samples and an index and output the transformed samples

 .. code-block:: python
	
	def transform(samps, i):
		lon = lon*2*np.pi
		lat = lat*np.pi/2
		f = f + 100.0
		fdot = fdot*1e-9
		return lon, lat, f, fdot