import numpy as np
#from pycbc.frame import read_frame as read_frame_pycbc
from .sft import SFT

class TimeSeries:
    
    def __init__(self):
        """ Class to contain timeseries"""

        self.timeseries = None
        self.delta_t = None
        self.sample_frequency = None
        self.start_time = None
        self.end_time = None


    def LAL_sfts_from_timeseries(self):

        self.timeseries = self.timeseries.astype("float64") # make real8type
        self.length = len(self.timeseries)
        vect = lal.CreateREAL8Vector(self.length)
        vect.data = self.timeseries

        real8time = lal.CreateREAL8TimeSeries("", self.start_time, 0, self.delta_t, lal.SecondUnit, self.length)
        real8time.data = vect

        # !!!! NOT WORKING
        self.epochs = None
        lalpulsar.MakeSFTsFromREAL8TimeSeries(r8, self.epochs,"hann",0)
        
    def sfts_from_timeseries(self, tsft, window=None, fmin=None, fmax=None,real=False,tstart=None, overlap = 0):
        """
        Convert a timeseries array into a set of complex ffts or real fft
        Parameters
        ---------------------
        tsft: float
            length of an sft in seconds
        window: string (optional)
            type of window to use (only hanning right now will update) (optional)
        fmin: float
            lower frequency
        fmax: float
            upper frequency
        real: bool optional (True)
            return real part of fft if true
        tstart: int
            start time of fft
        overlap: float (optional)
            length of overlap as a fraction, i.e. 0.9 is a 90% overlap, default 0
        Returns
        ------------------
        sft: SFT
            return sft which can then be written to file etc
        
        """
        
        # initialise sft object
        sft = SFT()
        
        # define the sampling frequency as 1/dt if not defined already
        if self.sample_frequency is None:
            self.sample_frequency = 1./self.delta_t

        if self.delta_t is None:
            self.delta_t = 1./self.sample_frequency
            
        # define the frequency bin size for the sft
        sft_df = 1./tsft
            
        # set the bin index for fmin and fmax
        if fmin is None:
            fmin_index = int(0)
        else:
            fmin_index = int(fmin*tsft)
        if fmax is None:
            fmax_index = int(self.sample_frequency*tsft + 1)
        else:
            fmax_index = int(fmax*tsft + 1)
            
        # define the start time if not defined already
        if tstart is not None:
            if self.start_time is None:
                self.start_time = tstart
        elif tstart is None:
            if self.start_time is not None:
                tstart = self.start_time
            else:
                tstart = 0
        # set the detector name to none if it is not defined
        if not hasattr(self,"det_name"):
            self.det_name = "none"

        # set the length of each segment in time, and the length of the fft = N/2
        segment_length_index = int(tsft*self.sample_frequency)
        fft_length_index = int(segment_length_index/2)

        # split the time series up into segments of length segment_length_index overlapping by overlap
        # segments which re not the correct length are ignored, i.e. at the end of the data the segment will be cut off
        time_segments = []
        segment_number = np.floor(float(len(self.timeseries))/(segment_length_index*(1-overlap))).astype(int)
        for i in range(segment_number):
            temp_seg = np.array(self.timeseries[int(i*segment_length_index*(1-overlap)):int(int(i*segment_length_index*(1-overlap)) + segment_length_index)])
            if len(temp_seg) == segment_length_index:
                time_segments.append(temp_seg)
            else:
                print("Segment not correct length, omitting SFT no {}".format(i))
            del temp_seg

        
        tlen = int(len(time_segments))
        flen = int(len(time_segments[0]))
    
        
        # set up arrays for ffts if ral and complex
        if real:
            ffts = np.zeros((tlen, int(fft_length_index + 1)))
            fft_freqs = np.zeros((tlen, int(fft_length_index + 1)))
        elif not real:
            ffts = np.zeros((tlen, int(fft_length_index + 1)), dtype=np.complex_)
            fft_freqs = np.zeros((tlen, int(fft_length_index + 1 )))

        # create each sft
        epochs = np.zeros(tlen)
        for t,seg in enumerate(time_segments):
            # set window function, default hanning, need to add more options
            if window == "hann" or window == "hanning":
                seg_wind = np.hanning(segment_length_index) * seg
            elif window is None:
                seg_wind = seg
            elif type(window) in [list,np.array,np.ndarray]:
                # use input array as window function
                if len(window) != len(seg):
                    raise Exception("Window must be same length as the segment")
                else:
                    seg_wind = window * seg
            if real:
                epochs[t] = tstart + t*tsft
                fft_calc = np.fft.rfft(seg_wind)
                ffts[t] = fft_calc#[:int(len(fft_calc/2) )]
                fft_freqs[t] = np.fft.rfftfreq(len(ffts[t]), self.delta_t)
            else:
                epochs[t] = tstart + t*tsft
                fft_calc = np.fft.fft(seg_wind)
                # take first half of fft
                ffts[t] = fft_calc[:int(len(fft_calc)/2 + 1)]
                # take absolute of fft freqs as including the nyquist freq is negative 
                fft_freqs[t] = np.abs(np.fft.fftfreq(len(fft_calc), self.delta_t)[:int(len(fft_calc)/2 + 1)])

        # set paramters of the sft
        sft.frequencies = fft_freqs[:,fmin_index:fmax_index]
        sft.fmin = sft.frequencies.min()
        sft.fmax = sft.frequencies.max()
        sft.delta_f = sft_df
        sft.sample_frequency = self.sample_frequency
        sft.tsft = tsft
        sft.sft = ffts[:,fmin_index:fmax_index]
        sft.epochs = epochs
        sft.det_name = self.det_name
        sft.nsft = segment_number
        return sft
    
    def write_sft_from_timeseries(self,output_path,tsft,window=None,fmin=None,fmax=None,narrowband = False):
        """
        write a gwf file to an sft
        args
        ----------
        output_path: string
            path to output file if narrowband, path to output dir if not narrowband
        tsft: float
            length of each sft
        window: string or array {optional:"hann"}
            window name, or array same width as data segment
        fmin: float {optional: 0}
            minimum frequency
        fmax: float {optional: }
            maximum frequency
        narrowband: bool {optional: False}
            either write to a narrowband combined set of sfts (True), or write different sft file for each sft (False)

        """
        
        sft = self.sfts_from_timeseries(tsft, window=window, fmin=fmin, fmax=fmax,real=False)
        sft.write_sft_files(output_path,narrowband=narrowband)
        
class LoadTimeSeries(TimeSeries):
    
    def __init__(self,filename,channel_name,start_time=None,end_time=None):
        """
        Load timeseries as an array from gwf file, uses pycbcs readframe
        Args
        -----------
        filename: string
            gwf filename
        channel_name: string:
            data channel_name
        start_time: int (optional: default - None)
            start time where you want to read the gwf file from, default is to read whole file
        end_time: int (optional: default - None)
            dn time where you want to read the gwf file to, default is to read whole file
        """
        self.read_frame(filename=filename,channel_name=channel_name,start_time=start_time,end_time=end_time)
    
    def read_frame(self,filename,channel_name,start_time=None,end_time=None):
        
        fr = read_frame_pycbc(filename,channels = channel_name,start_time=start_time,end_time=end_time)
        
        self.timeseries = np.array(fr.data)
        self.delta_t = float(fr.delta_t)
        self.delta_f = float(fr.delta_f)
        self.sample_frequency = float(fr.sample_rate)
        self.start_time = float(fr.start_time)
        self.end_time = float(fr.end_time)
        self.duration = float(fr.duration)
        
        # temporary want to get detectro from gwf file
        self.det_name = channel_name.split(":")[0]
        

