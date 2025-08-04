import pandas as pd
import h5py
import numpy as np
import emcee

# Define a sliding window function
def sliding_window_fraction(df, window_size=None, nsteps=40, threshold=None, key = "snrs"):
    las_fractions = []
    cnn_fractions = []
    las_fractions_error = []
    cnn_fractions_error = []
    
    zeroval = 0
    if key == "h0":
        df["logh0"] = np.log10(df["h0"])
        key = "logh0"
    if key == "logh0":
        zeroval = -np.inf
    if key == "depth":
        zeroval = np.inf    

    non_zero_snrs = df[df[key] != zeroval][key]
    if window_size is None:
        temp_steps = np.linspace(non_zero_snrs.min(), df[key].max(), nsteps)
        window_size = 10*np.abs(temp_steps[-2] - temp_steps[-1])

    minsnr = non_zero_snrs.min() + window_size
    maxsnr = df[key].max() - window_size
    snr_mids = np.linspace(minsnr, maxsnr, num=nsteps)
    for start in snr_mids:
        window = df[(df[key] >= start - window_size/2) & (df[key] < start + window_size/2)]
        if len(window) == 0:
            las_fraction = 0
            cnn_fraction = 0
            las_frac_err = 0
            cnn_frac_err = 0
        else:
            las_fraction = sum(window['las_detection'])/len(window["las_detection"])
            cnn_fraction = sum(window['cns_detection'])/len(window["cns_detection"])
            las_frac_err = np.sqrt(las_fraction * (1 - las_fraction) / len(window["las_detection"]))
            cnn_frac_err = np.sqrt(cnn_fraction * (1 - cnn_fraction) / len(window["cns_detection"]))
    
        if las_fraction is None:
            las_fraction = np.nan
        if cnn_fraction is None:
            cnn_fraction = np.nan
        las_fractions.append(las_fraction)
        cnn_fractions.append(cnn_fraction)
        las_fractions_error.append(las_frac_err)
        cnn_fractions_error.append(cnn_frac_err)
    if key == "logh0":
        snr_mids = snr_mids
    return snr_mids, las_fractions, cnn_fractions, las_fractions_error, cnn_fractions_error

def get_detection(df, key="las", quant=0.99, threshold=None):
    if threshold is None:
        threshold = df.loc[df['snrs'] == 0, key].quantile(quant)
    #threshold_cnn = df.loc[df['snrs'] == 0, 'cns'].quantile(0.99)
    df[f"{key}_detection"] = df[key] > threshold
    #df["cnn_detection"] = df["cns"] > threshold_cnn
    return df

def get_efficiency_at_val(eff_curves, eff=0.95, ind=1):
    final_eff = np.nan
    for i in range(len(eff_curves[0])):
        if eff_curves[ind][i] >= eff:
            final_eff = eff_curves[0][i]
            break
    return final_eff

def read_cnn_results_file(filename):
    """
    Read a file and return its contents as a list of lines.
    """
    with h5py.File(filename, "r") as f:
        df = pd.DataFrame({
            "las":  f["lineaware_statistic"][:], 
            "snrs": f["snr"][:], 
            "h0": f["h0"][:],
            "logh0": f["logh0"][:],
            "fmins": f["fmin"][:], 
            "depth": f["depth"][:],
            "cns":f["cnn_statistic"][:,1]})

    return df

def read_cnn_odd_even_results(even_filename, odd_filename):
    """
    Read odd and even results from files and return them as dataframes.
    """
    even_df = read_cnn_results_file(even_filename)
    odd_df = read_cnn_results_file(odd_filename)

    return even_df, odd_df


def join_and_group_by_frequency(df1, df2, n_bins=20, quant=0.99):
    """
    Join two dataframes and group by frequency.
    """

    # Define frequency range bins
    frequency_bins = np.round(np.arange(df1["fmins"].min(), df1['fmins'].max(), n_bins))

    dftot = pd.concat([df1, df2], ignore_index=True)
    falsealarm = dftot[dftot["snrs"] == 0]["las"].quantile(quant)
    df1 = get_detection(df1, "las", threshold=falsealarm)
    df2 = get_detection(df2, "las", threshold=falsealarm)

    ndfs = []
    for dataf in [df1, df2]:
        # Create a new column for frequency bins

        dataf['frequency_bin'] = pd.cut(dataf['fmins'], bins=frequency_bins, right=False)
        # Group the data by frequency bins
        grouped = dataf.groupby('frequency_bin')
        ndfs.append(grouped.apply(lambda group: get_detection(group, "cns", quant=quant)).reset_index(drop=True))


    
    dftot = pd.concat(ndfs, ignore_index=True)

    tot_grouped = dftot.groupby('frequency_bin')    

    #tot_grouped = tot_grouped.apply(lambda group: get_detection(group, key="las")).reset_index(drop=True).groupby('frequency_bin')

    return tot_grouped, frequency_bins

def get_efficiency_curves(grouped_dataset, window_size_snr=10, window_size_h0=0.1,window_size_depth=3,efficiency=0.95, nsteps=50):

    las_snr_at_95_fraction = []
    cnn_snr_at_95_fraction = []
    las_h0_at_95_fraction = []
    cnn_h0_at_95_fraction = []
    las_depth_at_95_fraction = []
    cnn_depth_at_95_fraction = []
    eff_curves_snr = {}
    eff_curves_h0 = {}
    for frequency_bin, group in grouped_dataset:
        fbin = float(str(frequency_bin).split('[')[1].split(',')[0])
        # Apply sliding window on SNR for each frequency bin
        snr_mids, las_s_fractions, cnn_s_fractions, las_s_fractions_error, cnn_s_fractions_error = sliding_window_fraction(group, window_size_snr, nsteps, key="snrs")
        h0_mids, h0_las_s_fractions, h0_cnn_s_fractions, h0_las_s_fractions_error, h0_cnn_s_fractions_error = sliding_window_fraction(group, window_size_h0, nsteps, key="logh0")
        depth_mids, depth_las_s_fractions, depth_cnn_s_fractions, depth_las_s_fractions_error, depth_cnn_s_fractions_error = sliding_window_fraction(group, window_size_depth, nsteps, key="depth")
        eff_curves_snr[fbin] = (snr_mids, las_s_fractions, cnn_s_fractions)
        eff_curves_h0[fbin] = (h0_mids, h0_las_s_fractions, h0_cnn_s_fractions)
        eff_curves_depth = (depth_mids, depth_las_s_fractions, depth_cnn_s_fractions)
        
        group = group.sort_values(by="snrs")
        las_snr_at_95_fraction.append((fbin, get_efficiency_at_val(eff_curves_snr[fbin], eff=efficiency, ind=1)))
        cnn_snr_at_95_fraction.append((fbin, get_efficiency_at_val(eff_curves_snr[fbin], eff=efficiency, ind=2)))

        las_h0_at_95_fraction.append((fbin, get_efficiency_at_val(eff_curves_h0[fbin], eff=efficiency, ind=1)))
        cnn_h0_at_95_fraction.append((fbin, get_efficiency_at_val(eff_curves_h0[fbin], eff=efficiency, ind=2)))

        las_depth_at_95_fraction.append((fbin, get_efficiency_at_val(eff_curves_depth, eff=efficiency, ind=1)))
        cnn_depth_at_95_fraction.append((fbin, get_efficiency_at_val(eff_curves_depth, eff=efficiency, ind=2)))
        

    las_snr_at_95_fraction = np.array(las_snr_at_95_fraction).astype(float)
    cnn_snr_at_95_fraction = np.array(cnn_snr_at_95_fraction).astype(float)
    las_h0_at_95_fraction = np.array(las_h0_at_95_fraction).astype(float)
    cnn_h0_at_95_fraction = np.array(cnn_h0_at_95_fraction).astype(float)
    las_depth_at_95_fraction = np.array(las_depth_at_95_fraction).astype(float)
    cnn_depth_at_95_fraction = np.array(cnn_depth_at_95_fraction).astype(float)

    las_snr_at_95_fraction[las_snr_at_95_fraction[:, 1] == None, 1] = np.nan
    cnn_snr_at_95_fraction[cnn_snr_at_95_fraction[:, 1] == None, 1] = np.nan
    las_h0_at_95_fraction[las_h0_at_95_fraction[:, 1] == None, 1] = np.nan
    cnn_h0_at_95_fraction[cnn_h0_at_95_fraction[:, 1] == None, 1] = np.nan
    las_depth_at_95_fraction[las_depth_at_95_fraction[:, 1] == None, 1] = np.nan
    cnn_depth_at_95_fraction[cnn_depth_at_95_fraction[:, 1] == None, 1] = np.nan

    return (eff_curves_snr, las_snr_at_95_fraction, cnn_snr_at_95_fraction), (eff_curves_h0, las_h0_at_95_fraction, cnn_h0_at_95_fraction), (eff_curves_depth, las_depth_at_95_fraction, cnn_depth_at_95_fraction)

# Fit the sigmoid function to the data
def sigmoid(x, a, b):
    return 1. / (1 + np.exp(-a * (x - b)))

def inv_sigmoid(y, a, b):
    return b - np.log(1/y - 1)/a

def fit_sigmoid(xdata, ydata, ranges, p0, quantiles=(0.05, 0.5, 0.95), nwalkers=32, ndim=2, log_probability=None, nsamples=2000):

    if len(xdata) > 1:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(xdata, ydata, ranges))
        sampler.run_mcmc(p0, nsamples, progress=True)

        samples = sampler.get_chain(discard=int(nsamples*0.5), thin=20, flat=True)

        resamp_x = np.linspace(np.min(xdata), np.max(xdata), 100)
        # Compute the sigmoid for all parameter samples
        sigmoids = [sigmoid(resamp_x, *params) for params in samples[::20]]

        output_quantiles = np.quantile(sigmoids, quantiles, axis=0)
        #output_quantiles = None

        eff_outputs = np.array([[resamp_x[i], *output_quantiles[:,i]] for i in range(len(resamp_x))])

        inv_sigmoids = [inv_sigmoid(0.95, *params) for params in samples]

        distr95 = np.quantile(inv_sigmoids, quantiles, axis=0)
        return samples, eff_outputs, distr95
    else:
        return np.nan, np.nan, np.array([np.nan, np.nan, np.nan])


def fit_sigmoid_efficiency_curves(grouped_dataset, efficiency=0.95, prior_ranges=None, sampler_params={"nwalkers":32, "ndim":2, "nsamples":2000}):


    # setup emcee run to run mcmc on sigmoid to detection data
    
    def log_prior(params, ranges=[(0,10), (10, 300)]):
        a, b = params
        if ranges[0][0] < a < ranges[0][1] and ranges[1][0] < b < ranges[1][1]:
            return 0.0
        return -np.inf
    
    def log_likelihood(params, x, y):
        a, b = params
        model = sigmoid(x, a, b)
        return np.nansum(y * np.log(model) + (1 - y)*(1 - model))

    def log_probability(params, x, y, ranges=[(0,10), (10, 300)]):
        lp = log_prior(params, ranges)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(params, x, y)


    nwalkers = sampler_params["nwalkers"]
    ndim = sampler_params["ndim"]
    nsamples = sampler_params["nsamples"]
    if prior_ranges is not None:
        snr_ranges = prior_ranges["snr"]
        h0_ranges = prior_ranges["h0"]
    else:
        snr_ranges = [(0.01, 5), (10, 300)]
        h0_ranges = [(0.01, 5), (-30, -10)]
    quantiles=(0.05, 0.5, 0.95)

    las_snr_at_95_fraction = []
    cnn_snr_at_95_fraction = []
    las_h0_at_95_fraction = []
    cnn_h0_at_95_fraction = []

    las_snr_efficiencies = []
    cnn_snr_efficiencies = []
    las_h0_efficiencies = []
    cnn_h0_efficiencies = []

    las_snr_samples = []
    cnn_snr_samples = []
    las_h0_samples = []
    cnn_h0_samples = []

    for frequency_bin, group in grouped_dataset:
        fbin = float(str(frequency_bin).split('[')[1].split(',')[0])
        snr_data = np.array(group["snrs"])
        h0_data = np.array(group["logh0"])
        ydata_las = np.array(group["las_detection"])
        ydata_cns = np.array(group["cns_detection"])

        snr_p0 = np.random.rand(nwalkers, ndim) * np.array([snr_ranges[0][1] - snr_ranges[0][0], snr_ranges[1][1] - snr_ranges[1][0]]) + np.array([snr_ranges[0][0], snr_ranges[1][0]])
        h0_p0 = np.random.rand(nwalkers, ndim) * np.array([h0_ranges[0][1] - h0_ranges[0][0], h0_ranges[1][1] - h0_ranges[1][0]]) + np.array([h0_ranges[0][0], h0_ranges[1][0]])

        # Calculate the median and 5th, 95th quantiles
        snr_las_samples, snr_las_qs, snr_las_95 = fit_sigmoid(snr_data, ydata_las, snr_ranges, snr_p0, quantiles=quantiles, nwalkers=nwalkers, ndim=ndim, log_probability=log_probability, nsamples=nsamples)
        snr_cns_samples, snr_cns_qs, snr_cns_95 = fit_sigmoid(snr_data, ydata_cns, snr_ranges, snr_p0, quantiles=quantiles, nwalkers=nwalkers, ndim=ndim, log_probability=log_probability, nsamples=nsamples)
        h0_las_samples, h0_las_qs, h0_las_95 = fit_sigmoid(h0_data, ydata_las, h0_ranges, h0_p0, quantiles=quantiles, nwalkers=nwalkers, ndim=ndim, log_probability=log_probability, nsamples=nsamples)
        h0_cns_samples, h0_cns_qs, h0_cns_95 = fit_sigmoid(h0_data, ydata_cns, h0_ranges, h0_p0, quantiles=quantiles, nwalkers=nwalkers, ndim=ndim, log_probability=log_probability, nsamples=nsamples)

        las_snr_at_95_fraction.append((fbin, *snr_las_95))
        cnn_snr_at_95_fraction.append((fbin, *snr_cns_95))
        las_h0_at_95_fraction.append((fbin, *h0_las_95))
        cnn_h0_at_95_fraction.append((fbin, *h0_cns_95))

        las_snr_efficiencies.append((fbin, snr_las_qs))
        cnn_snr_efficiencies.append((fbin, snr_cns_qs))
        las_h0_efficiencies.append((fbin, h0_las_qs))
        cnn_h0_efficiencies.append((fbin, h0_cns_qs))

        las_snr_samples.append((fbin, snr_las_samples))
        cnn_snr_samples.append((fbin, snr_cns_samples))
        las_h0_samples.append((fbin, h0_las_samples))
        cnn_h0_samples.append((fbin, h0_cns_samples))
    
    return {"snr_las":np.array(las_snr_at_95_fraction),
            "snr_cnn": np.array(cnn_snr_at_95_fraction), 
            "h0_las":np.array(las_h0_at_95_fraction), 
            "h0_cnn":np.array(cnn_h0_at_95_fraction),
            "snr_las_effs": las_snr_efficiencies,
            "snr_cnn_effs": cnn_snr_efficiencies,
            "h0_las_effs": las_h0_efficiencies,
            "h0_cnn_effs": cnn_h0_efficiencies,
            "snr_las_samples": las_snr_samples,
            "snr_cnn_samples": cnn_snr_samples,
            "h0_las_samples": las_h0_samples,
            "h0_cnn_samples": cnn_h0_samples,
            }

    