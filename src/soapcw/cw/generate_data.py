from __future__ import absolute_import
import numpy as np
import timeit
import lal
import lalpulsar
from lalpulsar import simulateCW
from scipy.special import fresnel
import scipy.stats as st
import time
from astropy.time import Time
import os
import copy
import subprocess
from .sft import SFT
from .timeseries import TimeSeries
from .tools import download_ephemeris_file, LAL_EPHEMERIS_URL


class GenerateSignal:
    
    def __init__(self,alpha=None,delta=None,psi=None,phi0=None,cosi=None,h0=None,f=None,tref=None,earth_ephem=None,sun_ephem=None,snr=None):
        """
        Set up parameters to generate a CW signal in noise
        args
        ---------
        alpha: float
            right ascencion for object, [0, 2pi]
        delta: float
            declination of object [-pi/2, pi/2]
        psi: float
            polarisation of object [0,2pi]
        phi0: float
            initial phase of signal
        cosi: float
            cosine of the inclination of star [0,1]
        h0: float
            Amplitude of the gravitational wave
        f: list or array
            2 component list of frequency parameters [f, fdot]
        tref: float
            reference time for phase (default tstart)
        earth_ephem: string
            path to earth ephemeris file
        sun_ephem: string
            path to sun ephemeris file
        snr: float
            SNR of signal, overwrites the h0 parameter
        """
        if earth_ephem is None:
            self.earth_ephem = "earth00-40-DE430.dat.gz"
        else:
            self.earth_ephem = earth_ephem
        if sun_ephem is None:
            self.sun_ephem = "sun00-40-DE430.dat.gz"
        else:
            self.sun_ephem = sun_ephem
        
        self.alpha = alpha
        self.delta = delta
        self.psi   = psi
        self.phi0  = phi0
        self.cosi  = cosi
        self.h0    = h0
        self.f     = f
        self.tref = None
        
        self.gaps = False
        self.snr = snr
        self.edat = None
        self.antenna_pattern = True
        self.baryinputs = {}
        self.siteinfo = {}
        self.earth = lalpulsar.EarthState()
        self.emit = lalpulsar.EmissionTime()
        
    def param_check(self):
        
        for i,v in enumerate(params[:-2]):
            if np.isnan(v):
                raise Exception("parameter {} is not a valid value".format(params_names[i]))
                
        if self.tref is None:
            print("No reference time: Pulsar referece time same as sft start time")
            self.tref = self.tstart

        
    def get_baryinput(self, detector):
        """
        Load in the baryinputs prior to running
        """
        baryinput = lalpulsar.BarycenterInput()
        if detector == 'SSB':
            # set up the for reference frame at center of earth                                                                              
            baryinput.site.location[0] = 0./lal.C_SI
            baryinput.site.location[1] = 0./lal.C_SI
            baryinput.site.location[2] = 0./lal.C_SI
        else:
            # get the information on specified detector                                                                                      
            # set up the barycenter for given detector    
            siteinfo = lalpulsar.GetSiteInfo(detector)   
            self.siteinfo[detector] = siteinfo                                                                             
            baryinput.site = siteinfo
            baryinput.site.location[0] = baryinput.site.location[0]/lal.C_SI
            baryinput.site.location[1] = baryinput.site.location[1]/lal.C_SI
            baryinput.site.location[2] = baryinput.site.location[2]/lal.C_SI

        return baryinput
        
        
    def detector_velocity(self,edat,epoch,detector):
        '''                                                                                                                                  
        get the three dimensional velocity of the earth                                                                                      
        args                                                                                                                                 
        --------                                                                                                                             
        edat  : the position of the earth and sun at a given time                                                                            
        epoch : the time of the start of the observation                                                                                     
        detector : the detector i.e. 'H1'                                                                                                    
                                                                                                                                         
        returns                                                                                                                              
        --------                                                                                                                             
        emit.vDetector : velocity of detector at arrival time (epoch) in cartesian coords                                                    
        '''

        # initialise the barycenter structure   
        if detector not in self.baryinputs.keys():
             self.baryinputs[detector] = self.get_baryinput(detector)                                                                                             

        # distance to the source in [s^-1]                                                                                                   
        self.baryinputs[detector].dInv = 0.0

        # source right ascension and declination in radians                                                                                  
        self.baryinputs[detector].alpha = 0.0
        self.baryinputs[detector].delta = 0.0

        # finds position and orientation of the earth at arrival time (epoch) for earth state and ephemerus data                             
        lalpulsar.BarycenterEarth(self.earth,epoch,edat)

        # transforms arrival time (epoch) into pulse emission time                                                                           
        lalpulsar.Barycenter(self.emit,self.baryinputs[detector],self.earth)

        return self.emit.vDetector

    def get_detector_velocities(self,epochs: np.array,det: str,edat = None):
        """
        get the velocities of the detector
        args
        -------------
        epochs: array or list
            list of the epoch times you would like to get their velocities
        det: detector
            which detector you would like this data for
        edat: LALEphemeris (optional)
            ephemeris data for solar system (can be auto loaded with get_edat)
        """
        if edat is not None:
            edat = edat
        elif self.edat is None:
            self.get_edat()
            edat = self.edat
        else:
            edat = self.edat

        self.det_vels = np.array(np.zeros((len(epochs),3))*np.nan)
        freqs = np.zeros(len(epochs))
        for i,epoch in enumerate(epochs):
            # get the detector velocity at certain epoch and ephemeris data                                                                  
            #epoch = lal.LIGOTimeGPS(epoch,0)                                                                                                
            vel = self.detector_velocity(edat,epoch,det)
            self.det_vels[i] = vel


    def get_pulsar_path(self,epochs,det,edat=None):
        """
        find the pulsar path in the time frequency plane, given the epochs, parameters and ephemeris data
        args                                                                                                                                 
        ------                                                                                                                               
        epochs : start times of each sft                                                                                                     
        edat   : ephermeris data for earth and sun                                                                                           
        det    : detector i.e 'H1'                                                                                                           
        params : parameters of the pulsar                                                                                                    
        """
        if not hasattr(self, "det_vels") or len(epochs) != len(self.det_vels):
            self.get_detector_velocities(epochs,det,edat)
        
        cosd = np.cos(self.delta)
        cosa = np.cos(self.alpha)
        sind = np.sin(self.delta)
        sina = np.sin(self.alpha)

        vDotn_c = cosd * cosa * self.det_vels[:,0] + cosd * sina  * self.det_vels[:,1] + sind * self.det_vels[:,2]

        if self.tref is None:
            self.tref = epochs[0]

        timeDiff = epochs - self.tref
        fhat = self.f[0] + timeDiff*self.f[1]
        
        return fhat * (1 + vDotn_c)

    def gps_to_gmst_rad(self, epochs):
        """ Taken amd vectorised from LAL code,  https://lscsoft.docs.ligo.org/lalsuite/lal/_x_l_a_l_sidereal_time_8c_source.html#l00062
        !!!!!!!!!! will definitely need updating !!!!!!!!!!!!!!!
        """
        #print("using custom GMST code taken from LAL")
        as_time = Time(epochs, format="gps", scale="utc")
        # keep track of larger part
        t_hi = (as_time.jd - 2451545.0)/36525.0
        # keep track of nanosecond part separately
        t_lo = (as_time.gps % 1e-9)/(1e9 * 36525.0 * 86400.0)
        t = t_hi + t_lo
        
        sidereal_time = (-6.2e-6 * t + 0.093104) * t * t + 67310.54841
        sidereal_time += 8640184.812866 * t_lo
        sidereal_time += 3155760000.0 * t_lo
        sidereal_time += 8640184.812866 * t_hi
        sidereal_time += 3155760000.0 * t_hi
        
        return sidereal_time * np.pi / 43200.0

    def av_antenna(self,alpha,delta,det,tstart,nsft,tsft,antenna=True,use_lal=False,average = True):
        """
        Calculates the time averages of the antenna patterns
        args
        ----------
        alpha: float
            right ascension
        delta: float
            declination
        det: string
            detector, 'H1', 'L1' etc
        T: float
            time of observation in seconds
        returns
        -----------
        A: float
        B: float
        C: float
        """
        # detector parameters in radians, lamb = detectro latitiude, L = longitude,
        # gamma = orientation of detector arms, zeta = angle between arms

        # need to change to get from lalpulsar
        det_params = {"H1": {"lamb": 46.45*np.pi/180., "L": 119.41*np.pi/180., "gamma": 171.8*np.pi/180., "zeta": 90.*np.pi/180.},
                      "L1": {"lamb": 30.56*np.pi/180., "L": 90.77*np.pi/180., "gamma": 243.0*np.pi/180., "zeta": 90*np.pi/180.},
                      "V1": {"lamb": 43.63*np.pi/180., "L": -10.5*np.pi/180., "gamma": 116.5*np.pi/180., "zeta": 90*np.pi/180.}}

        # this is not the source polarisation, that will be used later
        psi = 0.0

        if not antenna:
            return np.array([[1,1,1]])
        else:
            if det not in self.siteinfo.keys():
                self.baryinputs[det] = self.get_baryinput(det)  
            #detr = lalpulsar.GetSiteInfo(det)
            epochs = np.arange(tstart,tstart+tsft*nsft,tsft) + 0.5*tsft
            #gmst_rad = [lal.GreenwichMeanSiderealTime(i) for i in epochs]
            gmst_rad = self.gps_to_gmst_rad(epochs)
            am = [lal.ComputeDetAMResponse(self.siteinfo[det].response, alpha, delta, 0.0, i) for i in gmst_rad]
            fraczeta = (1./np.sin(det_params[det]["zeta"]))
            cos2psi = np.cos(2*psi)
            sin2psi = np.sin(2*psi)
            #A,B,C = 0,0,0
            ABC = []
            for i in range(len(epochs)):
                av,bv = 0,0
                fpl,fcr = am[i]
                av = fraczeta*(fpl*cos2psi - fcr*sin2psi)
                bv = fraczeta*(fcr*cos2psi + fpl*sin2psi)
                ABC.append((2*av*av,2*bv*bv,2*av*bv))
                del av,bv

            if average:
                return np.sum(ABC,axis=0) * 1./len(epochs)
            else:
                return np.array(ABC)

    def get_snr2(self,epochs=None,alpha=None, delta=None, psi=None, phi0=None, cosi=None, Sn=None, det=None, tstart=None, nsft=None, tsft=None, h0 = None,snr = None,antenna=True):
        """
        Calculates the snr or h0, if h0 input calculates snr if snr input calculates h0
        if snr<0 or None will find snr, else will find h0
        args
        ----------
        alpha: float
            right ascension
        delta: float
            declination
        psi: float
            polarisation of the wave
        phi0: float
            phase of the wave
        cosi: float
            cos of the inclination angle iota
        Sh: float or array
            noise spectral density at frequency of signal, not sqrt(Sh), or array if Sh differes with time
            if Sh == None, will calculate Sh from snr and h0
        det: string
            detector, 'H1', 'L1' etc
        tstart: float
            start time of observation
        T: float
            time of observation in seconds
        h0: float
            amplitude of the wave
        snr: float
            signal to noise 
        antenna: bool
            true is antenna pattern

        returns
        -----------
        snr_2 or h0: float
            The signal to noise squared or h0

        """
        params = {'alpha':alpha, 'delta':delta, 'psi':psi,'phi0':phi0,'cosi':cosi,'tstart':tstart,'nsft':nsft,'tsft':tsft,'h0':h0}

        for i,j in params.items():
            if j is None:
                if i in ["tstart","nsft"]:
                    pass
                else:
                    params[i] = getattr(self, i)
            else:
                pass

        ap = (1./2)*(1+params["cosi"]*params["cosi"])
        ac = params["cosi"]
        
        cos2phi0 = np.cos(2*params["phi0"])
        sin2phi0 = np.sin(2*params["phi0"])
        sin2psi = np.sin(2*params["psi"])
        cos2psi = np.cos(2*params["psi"])

        A1 = ap*cos2phi0*cos2psi - ac*sin2phi0*sin2psi
        A2 = ap*cos2phi0*sin2psi + ac*sin2phi0*cos2psi
        A3 = -ap*sin2phi0*cos2psi - ac*cos2phi0*sin2psi
        A4 = -ap*sin2phi0*sin2psi + ac*cos2phi0*cos2psi

        alp1 = A1*A1+A3*A3
        alp2 = A2*A2+A4*A4
        alp3 = A1*A2+A3*A4

        if epochs is None:
            if tstart is None or tsft is None or nsft is None:
                raise Exception("Please define epochs or (tstart,nsft,tsft)")
            else:
                epochs = np.linspace(params["tstart"], params["tstart"] + params["nsft"]*params["tsft"], params["nsft"])
        if params["tsft"] is None:
            params["tsft"] = epochs[1] - epochs[0]
        if params["tstart"] is None:
            params["tstart"] = epochs[0]
        if params["nsft"] is None:
            params["nsft"] = len(epochs)

        if Sn is None:
            raise Exception("Please define the noise floor Sn")
            
        if len(np.shape(Sn)) == 0:
            Sn = np.ones(len(epochs))*Sn

        if len(epochs) != len(Sn):
            raise Exception("Sn should be same length as epochs")

        ABC = self.av_antenna(alpha=params["alpha"],delta=params["delta"],det=det,tstart=params["tstart"],nsft=params["nsft"],tsft=params["tsft"],antenna=antenna,average=False)
        # nan to num accounts for areas when Sn is 0 or nan and sets the SNR to 0 when no data
        A,B,C = ABC[:,0],ABC[:,1],ABC[:,2]
        snr_2 = np.nan_to_num(((params["h0"]*params["h0"]*1*params["tsft"])/(2.0*Sn))*(alp1*A + alp2*B + 2*alp3*C))
        return snr_2
        
       

    def get_edat(self):
        """
        Get the ephemeris data from supplied ephemeris files, if not defined will download filenames
        """
        try:
            self.edat_p = [self.sun_ephem,self.earth_ephem]
            self.edat = lalpulsar.InitBarycenter(earthEphemerisFile=self.earth_ephem,sunEphemerisFile=self.sun_ephem)
        except Exception as e:
            print("Could not load ephemeris file: {} {}, {}".format(self.earth_ephem, self.sun_ephem, e))
            try:
                self.earth_ephem = download_ephemeris_file(LAL_EPHEMERIS_URL.format(self.earth_ephem))
                self.sun_ephem = download_ephemeris_file(LAL_EPHEMERIS_URL.format(self.sun_ephem))
                self.edat_p = [self.sun_ephem,self.earth_ephem]
                self.edat = lalpulsar.InitBarycenter(earthEphemerisFile=self.earth_ephem,sunEphemerisFile=self.sun_ephem)
            except Exception as e:
                raise IOError("Could not read in ephemeris files: {}".format(e))
        


        
    def get_spectrogram(self,fmin=None,fmax=None,tsft=None,epochs=None,Sn=None,tstart=None,nsft=None,dets=None,snr=None,doppler=True,pulsar_path=None,tref=None,antenna=True,noise_spect=None):
        """
        generates a spectrogram from the signal given some inputs
        
        args
        -----
        fmin: float
            lower frequency bound for spectrogram
        fmax: float
            upper frequency bound for spectrogram
        tsft: float
            length of each sft in seconds
        epochs: array (optional)
            array of time epochs to calculate sfts for, if not defined then these are found from tstart, T and tsft
        Sn: float, array, dict
            noise floors for each detector, if float then each detector has the same noise floor for whole run, can put in same float for each detector as array or and array for each detector as noise floor vaires. Best to put in dictionary saying which detector has which noise floor
        tstart: int (optional)
            start time of observation in seconds, in None this is found from epochs
        nsft: int (optional)
            number of sfts, if None this is found from T and tsft or epochs
        T: float (optional)
            length of data in seconds, nsft*tsft
        dets: str or list (optional)
            list of detectors to generate spectrogram for, if None, then uses keys defined in Sn
        snr: float (optional)
            total SNR of injections, this is the integrated recovered SNR for all detectors
        doppler: bool (optional)
            turn of doppler effects if False, default True
        pulsar_path: array (optional)
            path in frequency that a signal track should take, should be arrays for each detector, default find path from signal parmaeters
        tref: int (optional)
            reference time for antenna pattern, default time is tstart.
        antenna: bool (optional)
            turn off the antenna pattern modulations if False, default True
        noise_spect: dict
            real data to inject a signal into, format {"H1": H1_data, "L1": L1_data}
        
        returns
        ----------
        spt: SimulateSpectrogram class
            simulate spectrogram example attributes if one detector at H1
            H1: dict
                {"spect":spectrogram,"pulsar_path":pulsars track etc}
        
        """
        
        spt = SimulateSpectrogram(self)
        
        spt.__gen_spect__(epochs=epochs,tstart = tstart, nsft=nsft,tsft=tsft,fmin=fmin,fmax=fmax,dets=dets,snr=snr,Sn=Sn,doppler=doppler,antenna=antenna,tref=tref,pulsar_path=pulsar_path,noise_spect=noise_spect)
            
        return spt
    
    def get_timeseries(self,duration = None,tstart=None,tref=None,Sn=None,detectors=None,sample_frequency=4096):
        """
        gets the timeseries using lalpulsar.simulateCW.CWSimulator
        
        """

        timeseries = SimulateTimeseries(self)

        timeseries.gen_timeseries(duration=duration, tstart=tstart, tref=tref, Sn=Sn, detectors=detectors, sample_frequency=sample_frequency)

        return timeseries

class SimulateTimeseries:

    def __init__(self, parent):
        self._parent = parent
        self._parent.get_edat()

    def __getattr__(self, name):
        if name in self._parent.__dict__:
            try:
                return getattr(self._parent, name)
            except AttributeError:
                pass

        if name not in self.__dict__:
            raise AttributeError(name)
        return self.__dict__[name]

    def waveform(self,h0, cosi, freq, f1dot):
        def wf(dt):
            dphi = lal.TWOPI * (freq * dt + f1dot * 0.5 * dt**2)
            ap = h0 * (1.0 + cosi**2) / 2.0
            ax = h0 * cosi
            return dphi, ap, ax
        return wf

    
    def gen_timeseries(self,duration=1800, tstart=None, tref=None, Sn=None, detectors=["H1"], sample_frequency=4096):
                
        if self.h0 is None and self.snr is None:
            raise Exception("Please define either SNR or h0")

        if detectors is None and Sn is None:
            raise Exception("Please define the detectors of noise floor Sn")

        if tref is None:
            tref = self.tstart
        
        if Sn is not None:
            # if noise floor defined use it
            detectors = list(Sn.keys())
            Sn = Sn
            self.Sn = Sn


        if self.snr is not None and self.snr>=0 and self.Sn is not None:
            h0_bar = 1
            snr_bar = 0
            for dt in dets:
                if self.snr != 0:
                    snr_bar += np.nansum(self._parent.get_snr2(epochs=self.epochs,alpha=self.alpha,delta=self.delta,psi=self.psi,phi0=self.phi0,cosi=self.cosi, Sn=self.Sn[dt],det=dt,tsft=self.tsft, h0=h0_bar, antenna=self.antenna_pattern))
            scale = self.snr/np.sqrt(snr_bar)
            self.h0 = h0_bar*scale
        elif self.h0 is not None:
            self.h0 = self.h0


        
        # generate waveform
        wf = self.waveform(self.h0, self.cosi, self.f[0], self.f[1])
        for det in detectors:
            setattr(self,det,TimeSeries())
            timeseries = getattr(self,det)
            
            S = simulateCW.CWSimulator(tref, tstart, duration, wf, 1./sample_frequency, self.phi0, self.psi, self.alpha, self.delta, det, earth_ephem_file=self.earth_ephem,sun_ephem_file=self.sun_ephem)
            
            if Sn is None:
                ltime,signal = S.get_strain(sample_frequency)
            else:
                ltime,signal = S.get_strain(sample_frequency,noise_sqrt_Sh=np.sqrt(Sn[det]))
            
            
            timeseries.timeseries = signal
            timeseries.epochs = ltime
            timeseries.sample_frequency = sample_frequency
            timeseries.delta_t = 1./timeseries.sample_frequency
        return timeseries

class SimulateSpectrogram:
    
    def __init__(self,parent):
        self._parent = parent
        
    def __getattr__(self, name):
        if name in self._parent.__dict__:
            try:
                return getattr(self._parent, name)
            except AttributeError:
                pass

        if name not in self.__dict__:
            raise AttributeError(name)
        return self.__dict__[name]

    def fresnel_power(self,f,f0,tsft,alp):
        '''
        gives the Fresnel integrals for the fourier transform of a signal with changing frequency.
        ---------
        args
        --------
        f: float
            frequency
        f0: float
            central frequency of signal
        tsft: float
            length of an sft in seconds
        alp: float
           rate of change of frequency (\dot{f})
        -------
        returns
        --------
        Sw2: float
            power spectrum of signal at frequency w
        '''
        alp = np.abs(alp)
        sqrtalp = np.sqrt(alp/2.0)*tsft
        frq = 2*(f-f0)/(alp*tsft)
        x1 = sqrtalp*(1+frq)
        x2 = sqrtalp*(1-frq)

        S,C = fresnel([x1,x2])
        S1,S2 = S
        C1,C2 = C
        Sw2 = (1/(2*alp*tsft*tsft))*((C1+C2)*(C1+C2) + (S1+S2)*(S1+S2)) 
        return Sw2

    
    def __gen_spect__(self,fmin=None,fmax=None,tsft=None,epochs=None,Sn=None,tstart=None,nsft=None,dets="H1",snr=None,doppler=False,tref=None,rand=True,antenna=True,pulsar_path=None,noise_spect=None):
        """
        generates a normalised spectrogram, such that the mean of the noise is 2, (i.e. chi2 distribued data) and injectes a signal defined by GenerateSignal.
        -----------
        args
        -----------
        tsft: float
            length of each sft
        fmin: float
            lower frequency band
        fmax: float
            upper frequency band
        tstart: float (optional)
            start time in epoch time (set by first element of epochs if epochs defined)
        T: int (optional)
            length of time in number of sfts (set by epochs and tsft if they are defined)
        tref: int (optional)
            reference time for pulsar parameters, default tstart
        det: string or list
            which detector , 'H1','L1' etc, if multiple detectors ["H1","L1",....] etc
        snr: float (optional)
            snr of signal, if none uses h0 from params to find snr (the recovered SNR should be used as an actual measure of SNR, usually the same as this)
        antenna: bool (optioal)
            True for antenna pattern, false for no antenna pattern (default is True)
        rand: bool (optional)
            use random data or flat background with just signal (default is True)
        doppler: bool (optional)
            turn on or off doppler modulation of signal (default is True)
        Sn: float or array
            either a float for a fixed noise level, or array containing noise floor for each sft where gaps are NaN, if there are multiple (N) detectors Sn should N dimensional, either Nx1 or Nx(nsft) 

        """
        
        #self.param_check()
        # set the epochs and sft parameters from inputs
        self._parent.get_edat()
        self.T = None
        if fmin is None:
            raise Exception("Please set fmin as minimum frequency")
        else:
            self.fmin = fmin
        if fmax is None:
            raise Exception("Please set fmax as maximum frequency")
        else:
            self.fmax = fmax
        if tsft is None:
            raise Exception("Please set an sft length")
        else:
            self.tsft = tsft
    
        if snr is not None:
            self.snr = snr
        else:
            self.snr = None

        # define the noise floor if specified
        if dets is None and Sn is None and noise_spect is None:
            raise Exception("Please define the detectors (dets), noisefloors (Sn) or real data (noise_spect) to simulate")
        elif dets is None and type(Sn) in [dict]:
            dets = list(Sn.keys())
        elif dets is None and noise_spect is not None:
            dets = list(noise_spect.keys())
        elif type(dets) in [str]:
            dets = [dets]

        # if using multiple of the same detector, set the duplicate of noise floors
        if len(set(dets)) == len(dets):
            self.det_names = dets
        elif len(set(dets)) < len(dets):
            sa = set(dets)
            sd = {}
            for i in sa:
                sd.setdefault(i,0)
            self.det_names = []
            for i in dets:
                for j in sd.keys():
                    if i == j:
                        self.det_names.append(j + "_{}".format(sd[j]))
                        sd[j] +=1
                    else:
                        pass
        
            
        if noise_spect is not None:
            self.nsft = len(noise_spect[self.det_names[0]])
        
        # set the epochs for each sft, if it has not been defined already
        if epochs is None:
            if tstart is None or tsft is None and nsft is None:
                raise Exception("Please define either epochs or [tstart,tsft,nsft]")
            self.nsft = nsft
            self.T = nsft*tsft
            self.tstart = tstart
            self.epochs = np.linspace(tstart,tstart+(self.nsft-1)*tsft,self.nsft)
        else:
            self.epochs = epochs
            self.tstart = self.epochs[0]
            self.nsft = len(self.epochs)
            self.tsft = epochs[1] - epochs[0]

        # if the noise floor is not defined then define it, also check that if this is a list then 
        # also set it as an array of length epochs
        # want to define Sn as a dictionary so can be redefined later
        if noise_spect is not None:
            if Sn is not None:
                print("Your value of Sn will not be used, using noise spect instead")
            self.Sn = {}
            # for each detector get the positions where the noise is nan and set this as gap (igner noise floor value at this point as would normalise out anyhow)
            for dt in noise_spect.keys():
                sn_temp = np.ones(len(noise_spect[dt]))
                for ind,val in enumerate(np.mean(noise_spect[dt],axis=1)):
                    if val == 2 or np.isnan(val):
                        sn_temp[ind] = np.nan
                self.Sn[dt] = sn_temp
                del sn_temp
        else:
            # defined noise floor as constant over time is detectors defined but Sn not
            if Sn is None and dets is not None:
                self.Sn = {}
                for det in dets:
                    self.Sn[det] = np.ones(len(self.epochs))
            # if Sn defined check if constant or array
            elif type(Sn) in [dict]:
                if all(name in Sn for name in dets):
                    self.Sn = Sn
                    # for each detector if values is a float, then make this value same length as data otherwise ignore
                    for key in self.Sn.keys():
                        if len(np.shape(self.Sn[key])) == 0:
                            self.Sn[key] = np.ones(len(self.epochs))*self.Sn[key]
                        else:
                            if len(self.Sn[key]) != len(self.epochs):
                                raise Exception("Epochs and noise floor not the same length")
            else:
                raise Exception("Make sure keys for Sn and dets match")

        
        # set the scaled h0 of the signal based of the snr and the noise floor for each epoch and detector
        if self.snr is not None and self.snr>=0 and self.Sn is not None:
            h0_bar = 1
            snr_bar = 0
            for dt in dets:
                if self.snr !=0:
                    snr_bar += np.nansum(self._parent.get_snr2(epochs=self.epochs,alpha=self.alpha,delta=self.delta,psi=self.psi,phi0=self.phi0,cosi=self.cosi, Sn=self.Sn[dt],det=dt,tsft=self.tsft, h0=h0_bar, antenna=self.antenna_pattern))
                #print("SNRBAR1: {}".format(snr_bar))
            scale = self.snr/np.sqrt(snr_bar)
            self.h0 = h0_bar*scale
        elif self.h0 is not None and self.Sn is not None:
            self.h0 = self.h0
        else:
            raise Exeption("Please define either snr or h0 and Sn")

        

        self.harmonic_sum_Sn = 0
        # simulate the spectrogram for each detector
        for dn,dt in zip(self.det_names,dets):
            # set the noise floor if defined
            if noise_spect is not None:
                dt_noise_sp = noise_spect[dt]
            else:
                dt_noise_sp = None
            # generate the spectrogram
            det_data = self.__sim_data__(epochs=self.epochs,tstart=self.tstart,nsft=self.nsft,pulsar_path=pulsar_path,tref=tref,det=dt,antenna=antenna, rand=rand,doppler=doppler,Sn = self.Sn[dt],noise_spect=dt_noise_sp)
            # calcualte the harmonic sum of median noise floors for calulation of depth
            if self.h0 !=0:
                self.harmonic_sum_Sn += det_data.median_Sn 
            setattr(self, '{}'.format(dn), det_data)
            #self.__setattr__("{}".format(dn),det_data)
            del det_data
        # depth of signal as mean of depths from each detector
        if self.h0 == 0:
            self.depth = np.inf
        else:
            self.depth = np.sqrt(1/self.harmonic_sum_Sn)/self.h0 


                

    def __sim_data__(self,pulsar_path=None,tref=None,epochs=None,tstart=None,T=None,nsft=None,det='H1',antenna=True, rand=True,doppler=False,Sn=None,noise_spect=None):
        """
        Simulate data with pulsar signal
        -----------
        args
        -----------
        pulsar_path: array
            indicies of the track the pulsar makes in frequency
        tref: int
            reference time for pulsar parameters, default tstart
        epochs: array
            array of times for each sft
        tstart: float
            start time in epoch time
        T: int
            length of time in number of sfts
        nsft: float
            number of sfts to make
        det: string
            which detector , 'H1','L1' etc
        antenna: bool
            True for antenna pattern, false for no antenna pattern
        rand: bool
            use random data or flat background with just signal
        doppler: bool
            turn on or off doppler modulation of signal
        Sn: float or array
            either a float for a fixed noise level, or array containing noise floor for each sft where gaps are NaN
        noise_spect: 2d array
            set the noise to inject signal 


        ----
        kwargs
        ----
        snr: float
            snr of signal, if none uses h0 from params
        ----------
        returns
        ----------
        dict: 
        {"spect_1","spect_2","freq_track_1","freq_track_2","fmin","fmax","h0","SNR_1","SNR_2","SSB_track_index","depth","freqs","tot_snr"}
        """

        # initialise the SFT
        sft = SFT(tsft=self.tsft)

        #print("input minmaxf", self.fmin, self.fmax)
        # define the parameters of the sft
        #print("multipl", self.fmin*self.tsft, np.round(self.fmin*self.tsft), np.round(self.fmin*self.tsft).astype(int))
        #print("multipl2", self.fmax*self.tsft, np.round(self.fmax*self.tsft), np.round(self.fmax*self.tsft).astype(int))
        self.fmin = np.round(self.fmin*self.tsft).astype(int)/float(self.tsft) # min frequency
        self.fmax = np.round(self.fmax*self.tsft).astype(int)/float(self.tsft) # max frequency
        #print("round minmaxf", self.fmin, self.fmax)

        self.nbins = np.round((self.fmax-self.fmin)*self.tsft).astype(int) # number of frequency bins
        #print("nbins:", self.fmin, self.fmax, self.fmax-self.fmin, (self.fmax-self.fmin)*self.tsft, np.round((self.fmax-self.fmin)*self.tsft))
        self.delta_f = 1./self.tsft # separation of bins in frequency
        self.frequencies = np.arange(self.fmin,self.fmax,sft.delta_f)#[:-1] # bin centers for each bin

        sft.fmin = self.fmin
        sft.fmax = self.fmax
        sft.frequencies = self.frequencies
        sft.nbins = self.nbins
        sft.delta_f = self.delta_f
        sft.nsft = len(epochs)
        sft.epochs = epochs

        # initialise the sft as chi2 with 2 degrees of freedom (or ones if no noise)
        if rand:
            sft.norm_sft_power = st.chi2.rvs(int(2),loc=0,scale=1,size=(sft.nsft,sft.nbins))
        if not rand:
            sft.norm_sft_power = np.ones((sft.nsft,sft.nbins))

        # id the noise if defined from real data, then use this
        if noise_spect is not None:
            sft.norm_sft_power = noise_spect
            nsft = len(noise_spect)

        # set reference time as start of sft
        if not tref:
            self.tref = tstart
        elif tref:
            self.tref = tref

        # define doppler parameters of signal
        params_doppler = lalpulsar.PulsarDopplerParams()
        params_doppler.fkdot[0] = self.f[0]
        params_doppler.fkdot[1] = self.f[1]
        params_doppler.refTime  = self.tref
        params_doppler.Alpha    = self.alpha
        params_doppler.Delta    = self.delta

        # gen the frequency of the signal in the detector and SSB
        if doppler and pulsar_path is None:
            sft.pulsar_path = self._parent.get_pulsar_path(epochs,det)[:nsft]
            pulsar_path_ssb = self._parent.get_pulsar_path(epochs,'SSB')[:nsft]
        elif not doppler and pulsar_path is None:
            sft.pulsar_path = np.ones(nsft)*self.f[0]
            pulsar_path_ssb = None
        else:
            sft.pulsar_path = pulsar_path
            pulsar_path_ssb = None

        # get the bin centers of the signal 
        sft.pulsar_index = np.floor((sft.pulsar_path-self.fmin)*self.tsft) #+ 0.5/self.tsft
        bin_edges = (sft.pulsar_index/self.tsft)+self.fmin


        powers = []
        snr_calc = []

        if self.snr == 0 or self.h0 == 0:
            snr_calc.extend(np.zeros(nsft))
            # if no signal injected then fill gaps with mean of chi2 = 2
            for i,s in enumerate(Sn):
                if np.isnan(Sn[i]):
                    sft.norm_sft_power[i] = np.ones(len(sft.norm_sft_power[i]))*2
        else:
            # gets snr for each epoch, and convets nans where Sn is a gap to snr of 0
            snrs = np.nan_to_num(self._parent.get_snr2(epochs = self.epochs,alpha=self.alpha, delta=self.delta, psi=self.psi, phi0=self.phi0, cosi=self.cosi, Sn=Sn, det=det, tsft=self.tsft, h0=self.h0,antenna=antenna))
            # set snr to 0 if all elemts of sft power equa to mean in each segment (i.e. no data in this segment)
            mean_val_gap = np.all(sft.norm_sft_power == np.ones(len(sft.norm_sft_power[0]))*2,axis=1)
            #also zero if nan in Sn or data
            Sn_nan = np.isnan(Sn)
            # gen index where there is no data
            normnan = np.all(np.isnan(sft.norm_sft_power),axis=1)
            gapinds = np.any([mean_val_gap,Sn_nan,normnan],axis=0)
            # set that index snr to 0
            snrs[gapinds] = 0

            # loop over all epochs
            update_index = []
            for i,ind_floor in zip(list(range(len(sft.pulsar_index))),sft.pulsar_index):
                
                # set the index of the frequency bin which the signal is in
                pul_freq = sft.pulsar_path[i]
                # convert the index to an actual frequency
                freq_floor = (ind_floor/self.tsft)+self.fmin #+ 0.5/self.tsft
                
                # set limits on which how many bins the signal should be injected into +/- 3 bins of injection bin
                limits = (np.arange(7).astype(float)-3)

                # set the gaps in data to mean of chi2
                if gapinds[i]:
                    sft.norm_sft_power[i] = np.ones(len(sft.norm_sft_power[i]))*2
                    snrsq = 0
                else:
                    snrsq = snrs[i]
                    
                # append calcualtion of snr as a function of time
                snr_calc.append(snrsq)
                if snrsq == 0:
                    continue
                
                # if spectrum already defined have offset based of snr of noise line
                if noise_spect is not None:
                    # get area around signal injection
                    limlist = np.array(ind_floor + limits).astype(int)
                    # get area within band bounds
                    limlist = limlist[limlist > 0]
                    limlist = limlist[limlist < len(sft.norm_sft_power[0])]
                    # take the median of the power in this area, subtract the chi2 mean and divide by power (SNR_{line_signal} = ((P_{line_signal} + P_noise - P_noise)/P_noise)
                    # !!!!!!!!!!!!!!!!!!!currently turned off!!!!!!!!!!!!!!!!!!!!!!!!!
                    noise_offset = 0#np.median(sft.norm_sft_power[i,limlist]) - 2
                    if np.isnan(noise_offset) or noise_offset < 0:
                        noise_offset = 0
                else:
                    noise_offset = 0
                    
                # loop over the area around injection
                for lim in limits:
                    #bin center location
                    bin_cent_freq = freq_floor + lim/self.tsft
                    bin_floor_index = int(ind_floor + lim) #np.floor(((j1 + lim/tsft) - fmin)/tsft).astype(int)

                    if bin_floor_index < 0 or bin_floor_index >= len(sft.norm_sft_power[0]):
                        continue
                    
                    # distribute the power across \pm n bins accoring to the fresnel integral and multiply by the snr for that bin add noise offset if this exists
                    update_index.append([int(i),int(bin_floor_index),bin_cent_freq,pul_freq,snrsq,noise_offset])

            # update bins which have signal in them
            if len(update_index) != 0:
                update_index = np.array(update_index)
                # [time_ind, bin_floor_ind, bin_cent_freq, pulsar_freq, snr_bin, noise_offset]
                ncps = self.fresnel_power(update_index[:,2], update_index[:,3], self.tsft,self.f[1])*update_index[:,4] + update_index[:,5]
                update_index = np.c_[update_index, ncps]
                # [time_ind, bin_floor_ind, bin_cent_freq, pulsar_freq, snr_bin, noise_offset, ncp]
                update_index = update_index[update_index[:,6] !=0 ]
                # get all indicies to be updated and calculate the non central chi2 quared with corresponding non centrality parameter

                if rand:
                    sft.norm_sft_power[tuple(update_index[:,:2].astype(int).T)] = st.ncx2.rvs(int(2), nc=update_index[:,6], loc=0,scale=1)
                elif not rand:
                    sft.norm_sft_power[tuple(update_index[:,:2].astype(int).T)] = update_index[:,6]

        # calcualte recovered snr signal, should be equal to simulated snr
        sft.snr_recovered = np.sqrt(np.sum(np.array(snr_calc)))
        if self.h0 == 0:
            sft.depth = np.inf
            sft.median_Sn = np.nanmedian(Sn)
        else:
            # estimate the depth
            sft.depth = np.sqrt(np.nanmedian(Sn))/self.h0
            sft.median_Sn = np.nanmedian(Sn)
        return sft
    
    
    def sum_sfts(self,sum_type="norm_sft_power",gap_val = 2,nsfts=48):
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
        for dt in self.det_names:
            data = getattr(self,dt)
            data.sum_sft(sum_type=sum_type,gap_val=gap_val,nsfts=nsfts,remove_original=False)

    def downsamp_frequency(self,data_type="summed_norm_sft_power",stride = 2,remove_original=False):
        '''
        downsample the frequency by taking the mean of "stride" bins starting at base of band
        args
        -------
        stride: double
            number of frequency bins to take the mean of
        '''
        for dt in self.det_names:
            data = getattr(self,dt)
            data.downsamp_frequency(data_type=data_type,stride=stride,remove_original=remove_original)

            

    

