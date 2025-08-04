import soapcw

single_detector = True
multi_detector = False

if single_detector:
    channel_path = "/home/pulsar/public_html/fscan/1800s/<IFO>_DMT-ANALYSIS_READY/<day/week/month>/<epoch>/<channel>/"

    # Create a transition matrix (i.e. probability of moving by +/- 1 frequency bin)
    # input is ratio of probability of jumping up or down a bin to staying in the same bin 
    # transition_matrix = [0.33,0.33,0.33]
    transition_matrix = soapcw.transition_matrix_1d(1.0) 

    # load in the SFTs from the channel -- will convert into power spectra for SOAP input
    # options to sum over N SFTs or average over frequency bins
    data = soapcw.cw.LoadData(channel_path)

    # access data using data.{detector_name}.normalised_power

    soap_output = soapcw.single_detector(transition_matrix, data.H1.normalised_power)

    soap_output.write_to_hdf("output.hdf5")

if multi_detector:
    H1_channel_path = "/home/pulsar/public_html/fscan/1800s/<IFO>_DMT-ANALYSIS_READY/<day/week/month>/<epoch>/<channel>/"
    L1_channel_path = "/home/pulsar/public_html/fscan/1800s/<IFO>_DMT-ANALYSIS_READY/<day/week/month>/<epoch>/<channel>/"

    channel_paths = f"{H1_channel_path};{L1_channel_path}"
    # Create a transition matrix (i.e. probability of moving by +/- 1 frequency bin)
    # input is ratio of probability of jumping up or down a bin to staying in the same bin 
    # transition_matrix = [0.33,0.33,0.33]
    transition_matrix = soapcw.transition_matrix_2d(1.0, 1e400,1e400) 

    # load in the SFTs from the channel -- will convert into power spectra for SOAP input
    # options to sum over N SFTs or average over frequency bins
    data = soapcw.cw.LoadData(channel_paths)

    # access data using data.{detector_name}.normalised_power

    soap_output = soapcw.two_detector(transition_matrix, data.H1.normalised_power, data.L1.normalised_power)

    soap_output.write_to_hdf("output.hdf5")