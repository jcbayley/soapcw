import numpy as np
from scipy.signal import spectrogram
import lal
import lalpulsar
import pycbc
try:
    from pylal import Fr
except:
    print("No module Pylal installed, some funtions will fail")#from pylal import Fr



class MakeSFTs(object):

    def __init__(self):
        pass

    def get_spectrogram(self,frame_filename, channel_name, fft_length, overlap=None, fmin=None, fmax=None):
        """
        A small script to make a spectrogram of a gwf frame file using scipys spectrogram
        Parameters
        --------
        frame_filename: string
            path to frame file
        channel_name: string
            name of channel
        fft_length: int
            length of the fft in seconds 
        overlap: float (optional)
            what percentage are the SFTs overlapping by
        window: tuple (optional)
            which window type (inputs from scipy.signal.spectrogram)
        fmin: float (optional)
            lower frequency of sfts
        fmax: float (optional)
            upper frequency of sfts        
        """
        # read strain fro frame file
        # need to change to read in multiple frame files
        self.strain, self.gpsstart, self.ts = self.read_frame(frame_filename, channel_name)
        self.fs = 1./self.ts
        #create spectrogram 
        spect = self.make_sft(self.strain, self.fs, fft_length, overlap=overlap, fmin=fmin,fmax=fmax)
        self.spectrogram = spect[2]
        self.frequencies = spect[0]
        self.times = spect[1]

        if meandivide:
            self.mean_divide()

    def make_sft_from_gwf(self,output_path, frame_filename, channel_name, fft_length, detector_name, window = None, fmin=None, fmax = None):
        """
        take input of gwf frame files and output set of sfts of length fft_length
        Parameters
        ------------------
        output_path: string
            path to save sfts to
        frame_filename: string or list
            string of frame filename or list of filenames
        channel_name: string
            channel to load within the frame file
        fft_length: float
            length of the fft in seconds
        detector_name: string
            string of detector name, i.e. 'H1', 'L1'
        window: string (optional)
            which window type to use (default None) (only hanning available right now)
        fmin: float
            lower frequency in Hz
        fmax: float
            upper frequency in Hz
        """
        # read strain fro frame file
        # need to change to read in multiple frame files
        self.strain, self.gpsStart, self.ts = self.read_frame(frame_filename, channel_name)
        self.fs = 1./self.ts

        if fmin is None:
            fmin_index = int(0)
            fmin = 0
        else:
            fmin_index = int(fmin*fft_length)
            fmin = fmin
        if fmax is None:
            fmax_index = int(self.fs*fft_length)
            fmax = fft_length/2.
        else:
            fmax_index = int(fmax*fft_length)
            fmax = fft_length/2.

        #create ffts
        fft_freqs, ffts = self.sfts_from_timeseries(self.strain, self.fs, fft_length, window=window, fmin=fmin_index, fmax=fmax_index)

        # write ffts
        self.sft_file_from_array(output_path, ffts, self.fs, fft_length, detector_name, self.gpsStart, fmin,fmax)



    def read_frame(self,filename, strain_name="H1:GDS-CALIB_STRAIN", readstrain=True):
        """
        Read frame files
        Parameters
        ---------
        filename: string
            path to frame file
        channel_name: string
            name of channel
        Returns
        ---------
        strain: array
            array of string values
        gpsStart: float
            start time in gps time
        ts: float
            time between samples
        """
        if type(filename) is str:
            sd = Fr.frgetvect(filename, strain_name)    
            strain = sd[0]
            gpsStart = sd[1] 
            ts = sd[3][0]
        elif type(filename) == list:
            strain = []
            gpsStart = 0
            ts = 0
            for f in filename:
                # need to add in condition to make sure all same samplig etc and check all in order, i.e can join them up
                # not finished
                sd = Fr.frgetvect(filename, strain_name)
                strain.append(sd[0])
                gpsStart = sf[1]
                ts = sd[3][0]

        return strain, gpsStart, ts


    def mean_divide(self):
        """
        Divide the spectrogram by the mean in time
        very very basic attempt at line removal
        """
        meansp = np.mean(self.spectrogram.T,axis=0)
        for i in range(len(self.spectrogram.T)):
            self.spectrogram.T[i] = self.spectrogram.T[i]/meansp 

    def sfts_from_timeseries(self,timeseries, fs, fft_length, window=None, fmin=None, fmax=None,real=True):
        """
        Convert a timeseries array into a set of complex ffts
        Parameters
        ---------------------
        timeseries: array
            timeseries to be converted
        fs: float
            frequency of sampling in Hz
        fft_length: float
            segment length of times series to create fft from
        window: string (optional)
            type of window to use (only hanning right now will update) (optional)
        fmin: float
            lower frequency
        fmax: float
            upper frequency
        
        Returns
        ------------------
        fft_freqs: array
            2d array of frequencies for each segment
        ffts: array
            2d array of the complex ffts for each segment
        
        """

        if fmin is None:
            fmin_index = int(0)
        else:
            fmin_index = int(fmin*fft_length)
        if fmax is None:
            fmax_index = int(fs*fft_length)
        else:
            fmax_index = int(fmax*fft_length)

        fft_length_index = int(fft_length*fs)

        time_segments = []
        segment_number = np.floor(float(len(timeseries))/fft_length_index).astype(int)

        for i in range(segment_number):
            time_segments.append(np.array(timeseries[i*fft_length_index:fft_length_index*(i+1)]))

        if real:
            ffts = np.zeros((len(time_segments),len(time_segments[0])/2))
            fft_freqs = np.zeros((len(time_segments),len(time_segments[0])))#/2 + 1))
        elif not real:
            ffts = np.zeros((len(time_segments),len(time_segments[0])/2),dtype=np.complex_)
            fft_freqs = np.zeros((len(time_segments),len(time_segments[0])))#/2 + 1))

        for t,seg in enumerate(time_segments):
            if window == "hann":
                seg_wind = np.hanning(fft_length_index) * seg
            elif window is None:
                seg_wind = seg
            if real:
                fft_calc = np.fft.rfft(seg_wind)
                ffts[t] = fft_calc[:int(len(fft_calc)/2 )]
                fft_freqs[t] = np.fft.rfftfreq(fft_length_index, 1./fs)
            else:
                fft_calc = np.fft.fft(seg_wind)
                ffts[t] = fft_calc[:int(len(fft_calc)/2 )]
                fft_freqs[t] = np.fft.fftfreq(fft_length_index, 1./fs)

        fft_freqs_slice = fft_freqs[:,fmin_index:fmax_index]
        ffts_slice = ffts[:,fmin_index:fmax_index]
        return fft_freqs_slice, ffts_slice

    def sft_file_from_array(self, output_path, ffts, fs, fft_length, detector_name, gpsStart, fmin, fmax):
        """
        Save and array of ffts to a set of sft files ()
        Parameters
        ---------------------
        output_path: string
            path where sfts will be saved
        ffts: array
            array of complex ffts for each segment in ascending time order
        fs: float
            frequency of sampling in Hz
        fft_length: float
            segment length of times series to create fft from
        detector_name: string
            string for detector name i.e. L1,H1 etc
        gpsStart: int
            the start gps time
        f0: float
            start frequency

        """

        fft_length_index = int(fft_length*fs)

        for t,fft in enumerate(ffts):
            sft_temp = lal.CreateCOMPLEX8FrequencySeries(detector_name, lal.LIGOTimeGPS(gpsStart), fmin, 1./fft_length, lal.SecondUnit, int(fft_length_index/2.))

            sft_temp.data.data[:] = fft
            filename_prefix = "{}/{}_DT{}_T{}_F{}_{}.sft".format(output_path, detector_name,fft_length, gpsStart + t*fft_length , fmin,fmax)
            lalpulsar.WriteSFT2file(sft_temp, filename_prefix, None)

        return None
            
            
