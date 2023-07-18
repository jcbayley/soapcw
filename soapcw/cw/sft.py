import lal
import numpy as np
import lalpulsar

class SFT:
    
    
    def __init__(self,nsft=None,tsft=None,delta_f=None,fmin=None,fmax=None,det_name=None,tstart=None):
        """
        Class to contain SFTs
        """
        self.sft = None
        self.norm_sft_power = None
        self.tstart = tstart
        self.nsft = nsft
        self.tsft = tsft
        self.epochs = None
        self.delta_f = delta_f
        self.det_name = det_name
        self.fmin = fmin
        self.fmax = fmax
        
        if self.delta_f is None and self.tsft is not None:
            self.delta_f = 1./self.tsft
            
            
    def sum_sft(self,sum_type="norm_sft_power",gap_val = 2,nsfts=48,remove_original=False):
        '''
        takes input sfts with tsft=1800s and returns sum of 48 sfts, i.e summing over a day
        and returns fraction of real data in each sum.
        args
        -------
        gap_val: double
            value used when there are gaps in the data (for calculating fraction)
        returns
        -----------
        data_av: array
            data summed over every day
        fraction: array
            fraction of each day which contained real data
        '''
        data = getattr(self,sum_type)
        data_av1 = []
        dutycycle = []
        sum_epochs = []
        sum_rng_med = []
        
        for i in np.arange(0,self.nsft,nsfts).astype(int):
            end = i+nsfts
            if end >= self.nsft:
                data = np.vstack([data,gap_val*np.ones((end - self.nsft, len(data[0])))])
            chunk = data[int(i):int(end)]
            av = np.nansum(chunk.T,axis=1)
            if np.isnan(gap_val):
                fr = 1. - len(chunk[:,0][np.isnan(chunk[:,0])])/float(nsfts)
            else:
                fr = 1. - len(chunk[:,0][chunk[:,0] == gap_val])/float(nsfts)
                        
            data_av1.append(av)
            dutycycle.append(fr)
            if hasattr(self, "rng_med"):
                sum_rng_med.append(np.nanmedian(self.rng_med[int(i):int(end)]))

            if self.epochs is not None:
                sum_epochs.append(self.epochs[int(i)] + 0.5*nsfts*self.tsft)

        setattr(self,"summed_{}".format(sum_type),np.array(data_av1))
        setattr(self,"summed_dutycycle",np.array(dutycycle))
        setattr(self,"summed_epochs",np.array(sum_epochs))
        if hasattr(self,"rng_med"):
            setattr(self, "summed_rng_med", np.array(sum_rng_med))

        
        if remove_original:
            delattr(self,"sft")

    def downsamp_frequency(self,data_type="norm_sft_power",stride=2,remove_original=False):
        '''
        downsample the frequency by taking the mean of n bins
        args
        -------
        stride: int
            number of bins to downsample by, default: 2, takes the mean of two bins
        '''
        data = getattr(self,data_type)

        if stride > self.nsft:
            raise Exception("Stride greater than length of SFT")
        if not hasattr(self,"nbins"):
            self.nbins = len(data[0])
            
        bins = np.arange(self.nbins)[::stride]
        if hasattr(self,"frequencies"):
            freqs = self.frequencies[bins] + self.delta_f
            setattr(self,"downsamp_frequencies",freqs)
            
        down_freq = np.array([np.sum(data[:,i:i+stride],axis=1) for i in bins]).T            
        setattr(self,"downsamp_{}".format(data_type),down_freq)
        
        if remove_original:
            delattr(self,data_type)

            
    def RngMed(self,med_width,sum_type="norm_sft_power"):
        """
        Calculate the running median for a set of data, data the same as the first point half of median window size is appended to the start and same to the end.
        
        Parameters
        -------------
        data: array
            1d array of data to find the running median for
        med_width: int
            the width of the running median window in sample points
        Returns
        ------------------
        med: array
           the running median of the data set with chosen window size
        """
        m = med_width // 2
        self.rng_med = np.zeros(np.shape(getattr(self,sum_type)))
        for ind,data in enumerate(getattr(self,sum_type)):
            data1 = np.insert(data,0,np.ones(m)*data[0])
            data1 = np.insert(data1,-1,np.ones(m)*data[-1])

            med = []
            for idx,pos in enumerate(data):
                med.append(np.nanmedian(data1[idx+m-m:idx+m+m+1]))
            self.rng_med[ind] = med
            del med


    def get_sh(self,pos='mean',inattr = None):
        """
        generate average psds as a function of time for data segment.
        args
        -----------
        data: array
            noise_data from function get_multi_likelihood (i.e data = [norm_data, epochs_data, tsft, noise_data]) 
            or similar so data[3] is noise data
        returns
        ----------
        psds: array
            array of average psd for band, where gaps are nans

        """
        psds = []
        if hasattr(self,"sft"):
            sh_dat = self.sft
        elif hasattr(self,"rng_med"):
            sh_dat = self.rng_med
        else:
            if inattr is None:
                print("No input attribute defined and not sft or running median")
            else:
                sh_dat = getattr(self,inattr)
        for i in range(len(sh_dat)):
            if pos == 'mean':
                shm = np.mean(sh_dat[i])
            elif pos == 'hmean':
                shm = st.hmean(sh_dat[i])
            elif pos == 'median':
                shm = np.median(sh_dat[i])
            else:
                j = pos
                shm = sh_dat[i][j]
            if shm == 2.0 or np.isnan(shm):
                psds.append(np.nan)
            else:
                psds.append(shm*2.0/self.tsft)

        return np.array(psds)
    
    def write_sft_files(self, output_path, narrowband = False, fmin = None, fmax = None):
        """
        Save and array of ffts to a set of LIGO sft files
        Parameters
        ---------------------
        output_path: string
            path where sfts will be saved
        narrowband: bool (optional: default - False)
            if True then will save one sft file containing all sfts, set between fmin and fmax, false saves a separate file for each sft
        fmin: float (optional)
            minimum frequency of sft
        fmax: float (optional)
            maximum frequency of sft
        

        """
        
        # if there are no epochs the define them here
        if self.epochs is None:
            self.epochs = np.linspace(self.start_time,self.start_time + self.nsft*self.tsft,self.nsft)
            
        if self.delta_f is None:
            self.delta_f = 1./self.tsft
        
        #set the length of the fft if fmin and max are set 
        #fft_length_index = int(self.tsft*1./(self.sample_frequency)/2.)
        if fmin is None:
            fmin = self.fmin
        if fmax is None:
            fmax = self.fmax
            
        if fmin < self.fmin:
            raise Exception("Please regenerate SFT down to this frequency")
        if fmax > self.fmax:
            raise Exception("Please regenerate SFT up to this frequency")
        
        fmin_index = int((fmin-self.fmin)*self.tsft)
        fmax_index = int((self.fmax-fmax)*self.tsft + 1)
        
        fft_length_index = fmax_index - fmin_index

        
        # if narrownaded sfts wanted then set up the sft vector for them to be appended to
        if narrowband:
            vct = lalpulsar.CreateSFTVector(int(self.nsft),int(fft_length_index ))
        
        
        #loop over the sfts and either save to files, or save to a narrowbanded sft
        sft_temp = None
        if self.nsft == 1:
            enm = enumerate([self.sft])
        else:
            enm = enumerate(self.sft)
        for t,fft in enm:
            if not narrowband:
                if sft_temp is None:
                    sft_temp = lal.CreateCOMPLEX8FrequencySeries(self.det_name, lal.LIGOTimeGPS(self.epochs[t]), fmin, 1./self.tsft, lal.SecondUnit, fft_length_index )
                sft_temp.epoch = self.epochs[t]
                sft_temp.data.data[:] = np.array(fft[fmin_index:fmax_index])
                sft_temp.deltaF = self.delta_f
                sft_temp.f0 = self.fmin
                
                filename_prefix = "{}/{}_{}SFT_T{}-{}_F{}_{}.sft".format(output_path, self.det_name,self.tsft, int(self.epochs[t]) ,int(self.epochs[-1]-self.epochs[0]), fmin, fmax)
                lalpulsar.WriteSFT2file(sft_temp, filename_prefix, "")
            elif narrowband:
                vct.data[t].data.data[:] = np.array(fft[fmin_index:fmax_index]).astype("complex64")
                vct.data[t].deltaF = self.delta_f
                vct.data[t].f0 = self.fmin
                vct.data[t].epoch = self.epochs[t]
                if self.det_name is not None:
                    vct.data[t].name = self.det_name
                else:
                    vct.data[t].name = "NoDet"
            
        if narrowband: 
            filename_prefix = "{}/{}-{}_{}SFT_T{}-{}_F{}_{}.sft".format(output_path,self.det_name,self.nsft,self.tsft, int(self.epochs[0]) ,int(self.epochs[-1]-self.epochs[0]), fmin, fmax)
            lalpulsar.WriteSFTVector2NamedFile(vct, filename_prefix, "")
            
