============================
Complete SOAP Pipeline Guide
============================

This guide provides a comprehensive walkthrough for running the complete SOAP (Search for Outstanding Astrophysical Phenomena) pipeline for gravitational wave continuous wave searches. The pipeline includes data preparation, statistical analysis, machine learning model training, and result visualization.

.. contents:: Table of Contents
   :local:
   :depth: 3

Overview
========

The SOAP pipeline consists of several interconnected stages:

1. **Data Preparation**: Narrowbanding SFTs and generating training data
2. **Statistical Setup**: Creating line-aware lookup tables
3. **Core Search**: Running the main SOAP algorithm on data
4. **Machine Learning**: Generating CNN training data and training models
5. **Visualization**: Creating output plots and sensitivity curves

Prerequisites
=============

Required Software
-----------------

- Python 3.9-3.12
- LALSuite (gravitational wave data analysis framework)
- HTCondor (for distributed computing)
- Access to LIGO SFT data

Installation
------------

.. code-block:: bash

   # Set up the environment and install SOAP in development mode
   uv sync

   # Verify installation
   uv run soapcw-run-soap-astro --help

Pipeline Stages
===============

Stage 1: Data Preparation - Narrowbanding SFTs
-----------------------------------------------

The first step is to prepare narrowband SFTs (Short Fourier Transforms) from the full-bandwidth data.

**Command:**

.. code-block:: bash

   soapcw-narrowband-sfts \
     --input-dir /path/to/full/bandwidth/sfts \
     --output-dir /path/to/narrowband/sfts \
     --freq-start 50.0 \
     --freq-end 2000.0 \
     --bandwidth 0.1 \
     --detector H1,L1

**Parameters:**

- ``--input-dir``: Directory containing full-bandwidth SFT files
- ``--output-dir``: Directory to save narrowband SFTs
- ``--freq-start/--freq-end``: Frequency range to process (Hz)
- ``--bandwidth``: Width of each narrowband (Hz)
- ``--detector``: Comma-separated list of detectors (H1, L1, V1)

**Output:** Narrowband SFT files organized by frequency band and detector.

Stage 2: Generate Line-Aware Statistics
----------------------------------------

Create lookup tables for line-aware statistical analysis to distinguish between astrophysical signals and instrumental lines.

**Command:**

.. code-block:: bash

   soapcw-make-line-aware-statistics \
     --output-dir /path/to/lookup/tables \
     --snr-width-line 4.0 \
     --snr-width-signal 10.0 \
     --prob-line 0.4 \
     --lookup-type power

**Parameters:**

- ``--output-dir``: Directory to save lookup tables
- ``--snr-width-line``: Prior width of line SNR distribution
- ``--snr-width-signal``: Prior width of signal SNR distribution
- ``--prob-line``: Prior probability ratio of line vs noise model
- ``--lookup-type``: Type of lookup table (``power`` or ``amplitude``)

**Output:** Binary lookup table files for statistical analysis.

Stage 3: Run Main SOAP Algorithm
---------------------------------

Execute the core Viterbi-based continuous wave search algorithm.

Configuration File Setup
~~~~~~~~~~~~~~~~~~~~~~~~~

Create a configuration file (e.g., ``search_config.ini``):

.. code-block:: ini

   [general]
   root_dir = /path/to/run/output
   temp_dir = /path/to/temp/directory

   [condor]
   memory = 8000
   request_disk = 10000
   accounting_group = ligo.dev.o4.cw.explore.test
   n_jobs = 100
   band_load_size = 8.0

   [input]
   load_directory = [/path/to/H1/sfts, /path/to/L1/sfts]
   hard_inj = /path/to/hardware/injections.h5
   lines_h1 = /path/to/H1_lines.txt
   lines_l1 = /path/to/L1_lines.txt

   [data]
   band_starts = [50, 500, 1000, 1500]
   band_ends = [500, 1000, 1500, 2000]
   band_widths = [0.1, 0.2, 0.3, 0.4]
   strides = [1, 2, 3, 4]
   obs_run = O4
   n_summed_sfts = 48

   [lookuptable]
   lookup_type = power
   lookup_dir = /path/to/lookup/tables
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
   save_directory = /path/to/results
   sub_directory = soap_search_run1

Astrophysical Search
~~~~~~~~~~~~~~~~~~~~

For searching for astrophysical continuous wave signals:

.. code-block:: bash

   # Generate DAG files for distributed computing
   soapcw-make-dag-files-astro -c search_config.ini

   # Submit the DAG file (created in root_dir)
   condor_submit_dag soap_astro_search.dag

   # Or run directly (for smaller searches)
   soapcw-run-soap-astro \
     --config search_config.ini \
     --start-freq 50.0 \
     --end-freq 100.0 \
     --band-width 0.1 \
     --stride 1

Line Search
~~~~~~~~~~~

For detector characterization and line searches:

.. code-block:: bash

   # Generate DAG files
   soapcw-make-dag-files-lines -c search_config.ini

   # Submit the DAG file
   condor_submit_dag soap_line_search.dag

   # Or run directly
   soapcw-run-soap-lines \
     --config search_config.ini \
     --start-freq 50.0 \
     --end-freq 100.0

**Output:** Search results including candidate lists, Viterbi maps, and statistical data.

Stage 4: Generate CNN Training Data
------------------------------------

Create training datasets for convolutional neural network models.

**Command:**

.. code-block:: bash

   # Generate training data DAG for distributed processing
   soapcw-cnn-make-data-dag \
     --config cnn_config.ini \
     --output-dir /path/to/training/data

   # Or generate data directly
   soapcw-cnn-make-data \
     --sft-dir /path/to/narrowband/sfts \
     --output-dir /path/to/training/data \
     --freq-start 50.0 \
     --freq-end 2000.0 \
     --n-samples 10000 \
     --signal-snr-range 10,100

**Training Data Types:**

- **Viterbi Maps**: 2D frequency-time tracking maps
- **Spectrograms**: Power spectral density data
- **Statistics**: Line-aware statistical features
- **Combined**: Multi-modal training data

**Output:** HDF5 files containing training/validation/test datasets.

Stage 5: Train CNN Models
--------------------------

Train machine learning models for signal detection and classification.

CNN Configuration
~~~~~~~~~~~~~~~~~

Create CNN configuration file (``cnn_config.ini``):

.. code-block:: ini

   [general]
   output_dir = /path/to/models
   data_dir = /path/to/training/data

   [training]
   batch_size = 32
   epochs = 100
   learning_rate = 0.001
   validation_split = 0.2

   [model]
   model_type = vitmapmodel  # vitmapmodel, spectmodel, vitmapstatmodel, allmodel
   input_shape = [128, 128, 1]

   [data_generation]
   n_train_samples = 50000
   n_val_samples = 10000
   signal_snr_range = [10, 100]
   noise_floor = 1.0

Train Models
~~~~~~~~~~~~

.. code-block:: bash

   # Train Viterbi map model
   soapcw-cnn-train-model \
     --config cnn_config.ini \
     --model-type vitmapmodel \
     --output-path /path/to/models/vitmapmodel.pt

   # Train spectrogram model
   soapcw-cnn-train-model \
     --config cnn_config.ini \
     --model-type spectmodel \
     --output-path /path/to/models/spectmodel.pt

   # Train combined statistics model
   soapcw-cnn-train-model \
     --config cnn_config.ini \
     --model-type vitmapstatmodel \
     --output-path /path/to/models/vitmapstatmodel.pt

**Model Types:**

- ``vitmapmodel``: Processes Viterbi tracking maps
- ``spectmodel``: Processes power spectrograms
- ``vitmapstatmodel``: Combines Viterbi maps with statistics
- ``allmodel``: Multi-modal model using all data types

**Output:** Trained PyTorch model files (``.pt``) for inference.

Stage 6: Re-run Search with Trained Models
-------------------------------------------

Integrate trained CNN models into the search pipeline for enhanced detection.

Update your configuration file:

.. code-block:: ini

   [cnn]
   vitmapmodel_path = /path/to/models/vitmapmodel.pt
   spectmodel_path = /path/to/models/spectmodel.pt
   vitmapstatmodel_path = /path/to/models/vitmapstatmodel.pt
   allmodel_path = /path/to/models/allmodel.pt

Then re-run the search:

.. code-block:: bash

   soapcw-run-soap-astro --config search_config.ini

Stage 7: Generate Results and Visualizations
---------------------------------------------

Create HTML pages with plots, sensitivity curves, and candidate summaries.

**Command:**

.. code-block:: bash

   soapcw-make-html-pages --config search_config.ini

**Generated Outputs:**

1. **HTML Summary Pages**: Interactive web pages with search results
2. **Candidate Lists**: Top gravitational wave candidates with SNR rankings
3. **Viterbi Maps**: 2D visualizations of frequency tracking
4. **Sensitivity Curves**: Upper limits on gravitational wave strain
5. **Band Summaries**: Statistical analysis across frequency bands
6. **Detection Statistics**: ROC curves and detection efficiency plots

**File Locations:**

Results are saved to the directory specified in ``[output] save_directory``:

::

   results/
   ├── index.html              # Main summary page
   ├── candidates/             # Individual candidate pages
   ├── plots/                  # PNG/PDF plot files
   ├── data/                   # Raw result data files
   └── sensitivity/            # Sensitivity curve data

Example Complete Workflow
==========================

Here's a complete example workflow for a typical SOAP search:

.. code-block:: bash

   # 1. Prepare narrowband SFTs
   soapcw-narrowband-sfts \
     --input-dir /hdfs/frames/O4/pulsar/sfts/C01/ \
     --output-dir /path/to/narrowband/sfts \
     --freq-start 50.0 --freq-end 2000.0 \
     --bandwidth 0.1 --detector H1,L1

   # 2. Generate line-aware statistics
   soapcw-make-line-aware-statistics \
     --output-dir /path/to/lookup/tables \
     --snr-width-line 4.0 --snr-width-signal 10.0

   # 3. Generate CNN training data
   soapcw-cnn-make-data \
     --sft-dir /path/to/narrowband/sfts \
     --output-dir /path/to/training/data \
     --freq-start 50.0 --freq-end 100.0 \
     --n-samples 10000

   # 4. Train CNN models
   soapcw-cnn-train-model \
     --config cnn_config.ini \
     --model-type vitmapmodel \
     --output-path /path/to/models/vitmapmodel.pt

   # 5. Run main search with trained models
   soapcw-make-dag-files-astro -c search_config.ini
   condor_submit_dag soap_astro_search.dag

   # 6. Generate HTML results
   soapcw-make-html-pages --config search_config.ini

Performance Optimization
=========================

For Large-Scale Searches
------------------------

- Use HTCondor DAG files for distributed processing
- Set appropriate ``band_load_size`` to balance memory and compute time
- Use multiple frequency bands with different bandwidths for efficiency
- Enable CNN models only for final candidate selection

Memory Management
-----------------

- Adjust ``n_summed_sfts`` based on available memory
- Use smaller ``band_width`` values for lower memory usage
- Set appropriate Condor memory requirements

Troubleshooting
===============

Common Issues
-------------

**SFT Loading Errors**
  - Verify SFT file paths and permissions
  - Check detector names match SFT file naming convention
  - Ensure sufficient disk space for temporary files

**Memory Issues**
  - Reduce ``band_load_size`` or ``n_summed_sfts``
  - Increase Condor memory allocation
  - Use narrower frequency bands

**CNN Training Failures**
  - Verify training data format and completeness
  - Check GPU availability and CUDA compatibility
  - Adjust batch size for available memory

**HTCondor Job Failures**
  - Check accounting group permissions
  - Verify file paths are accessible from compute nodes
  - Review Condor log files for specific errors

Getting Help
============

- Check the `SOAP documentation <https://soapcw.readthedocs.io/>`_
- Review example configuration files in ``src/soapcw_pipeline/config_files/``
- Examine Jupyter notebook tutorials in ``docs/usage/``
- Contact the development team for LIGO-specific deployment issues

This completes the comprehensive SOAP pipeline guide. Each stage builds upon the previous ones to create a complete gravitational wave continuous wave search and analysis workflow.
