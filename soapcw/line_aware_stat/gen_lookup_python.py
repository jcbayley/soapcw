import scipy.stats as st
import numpy as np
from scipy.integrate import quad
import pickle
import os

#--------------------------------------------------------------------------------
# Numerically finding the integral by integrating over SNR^2, where the variance depends on the standard deviation
#------------------------------------------------------------------------------------


class LineAwareStatistic:

    def __init__(self, powers, ndet=2 ,k=2, N=48, signal_prior_width=1, line_prior_width=5, noise_line_model_ratio=1,approx=True):

        self.ndet = ndet
        self.signal_prior_width = signal_prior_width
        self.line_prior_width = line_prior_width
        self.noise_line_model_ratio = noise_line_model_ratio
        self.powers = powers
        self.k = k
        self.N = N
        self.kN = self.k*self.N
        
        if ndet == 1:
            self.signoiseline,self.signoise,self.sigline,self.sig,self.noise,self.line = self.gen_lookup_one_det(powers,approx=approx,signal_prior_width=signal_prior_width,line_prior_width=line_prior_width,noise_line_model_ratio=noise_line_model_ratio)
        elif ndet == 2:
            self.signoiseline,self.signoise,self.sigline,self.sig,self.noise,self.line = self.gen_lookup_two_det(powers,powers,approx=approx,signal_prior_width=signal_prior_width,line_prior_width=line_prior_width,noise_line_model_ratio=noise_line_model_ratio)
        else:
            raise Exception("This currently only works for 1 or 2 detectors")    
        
    def chi2_sig(self,lamb: np.array,gs: float,pv: float) -> np.array:
        """
        returns the likelihood of two powers multiplied by the prior on the snr**2
        args
        -------
        lamb: float
            snr^2
        g1: float
            SFT power in detector 1
        g2: float
            SFT power in detector 2
        k: int
            number of degrees of freedom
        N: int
            Number of summed SFTS
        pv: float
            width of exponetial distribution of prior
        returns
        ---------
        func*prior: float
        likelihood multiplied by prior
        """
        if len(np.shape(gs)) == 0:
            func = st.ncx2.pdf(gs,df=self.kN,nc=lamb,loc=0,scale=1)
        else:
            func = np.prod([st.ncx2.pdf(i,df=self.kN,nc=lamb,loc=0,scale=1) for i in gs], axis=0)
        wid = 1./pv
        return func*wid*np.exp(-wid*lamb)


    def chi2_line(self,lamb,gs,pv):
        """
        returns the likelihood of two powers multiplied by the prior on the snr**2
        works for 1 or 2 detectors only

        args
        -------
        lamb: float
        snr^2
        g1: float
        SFT power in detector 1
        g2: float
        SFT power in detector 2
        k: int
        number of degrees of freedom
        N: int
        Number of summed SFTS
        pv: float
        width of exponetial distribution of prior
        returns
        ---------
        func*prior: float
        likelihood multiplied by prior
        """
        if len(np.shape(gs)) == 0:
            func = st.ncx2.pdf(gs,df=self.kN,nc=lamb,loc=0,scale=1)
        else:
            func = 1./len(gs)*(st.chi2.pdf(gs[0],df=self.kN,loc=0,scale=1)*st.ncx2.pdf(gs[1],df=self.kN,nc=lamb,loc=0,scale=1) + st.ncx2.pdf(gs[0],df=self.kN,nc=lamb,loc=0,scale=1)*st.chi2.pdf(gs[1],df=self.kN,loc=0,scale=1))
        wid = 1./pv
        return func*wid*np.exp(-wid*lamb)
    
    
    def chi2_noise(self,gs):
        if len(np.shape(gs)) == 0:
            func = st.chi2.pdf(gs,df=self.kN,loc=0,scale=1)
        else:
            func = np.prod([st.chi2.pdf(i,df=self.kN,loc=0,scale=1) for i in gs],axis=0)
        return func


    def two_det(self,g1,g2,signal_prior_width=10,line_prior_width=10,noise_line_model_ratio=1., approx=True):
        """
        integrate likelihood and prior to get evidence for powers g1 and g2
        args
        -------
        g1: float
        g2: float
        k: int
        N: int
        signal_prior_width: float
        line_prior_width: float
        noise_line_model_ratio: float
        approx: bool
            choose to approximate the integral with trapz or full integnoise_line_model_ration
        """
        if approx:
            l = np.linspace(0,100,500)
            sig_int = np.trapz(np.nan_to_num(self.chi2_sig(l,[g1,g2],signal_prior_width)),l)
            line_int = np.trapz(np.nan_to_num(self.chi2_line(l,[g1,g2],line_prior_width)),l)
        else:
            sig_int, sig_err = quad(self.chi2_sig,0,np.inf, args = ([g1,g2],signal_prior_width))
            line_int, line_err = quad(self.chi2_line,0,np.inf, args = ([g1,g2],line_prior_width))

        noise = self.chi2_noise([g1,g2])
        siglinenoise = sig_int/(noise_line_model_ratio*line_int + noise )
        signoise = sig_int/noise
        sigline = sig_int/line_int
        
        return siglinenoise,signoise,sigline,sig_int,noise,line_int

    def one_det(self,g1,signal_prior_width=10,line_prior_width=10,noise_line_model_ratio=1., approx=True):
        """
        integrate likelihood and prior to get evidence for powers g1 and g2
        args
        -------
        approx: bool
            choose to approximate the integral with trapz or full integnoise_line_model_ration
        """

        if approx:
            l = np.linspace(0,100,500)
            sig_int = np.trapz(np.nan_to_num(self.chi2_sig(l,g1,signal_prior_width)),l)
            line_int = np.trapz(np.nan_to_num(self.chi2_line(l,g1,line_prior_width)),l)
        else:
            sig_int, sig_err = quad(self.chi2_sig,0,np.inf, args = (g1,signal_prior_width))
            line_int, line_err = quad(self.chi2_line,0,np.inf, args = (g1,line_prior_width))

        noise = self.chi2_noise(g1)
        siglinenoise = sig_int/(noise_line_model_ratio*line_int + noise )
        signoise = sig_int/noise
        sigline = sig_int/line_int
        
        return siglinenoise,signoise,sigline,sig_int,noise,line_int

    
    def gen_lookup_one_det(self,powers,approx=True,signal_prior_width=10,line_prior_width=10,noise_line_model_ratio=1):
        """
        calculate lookup table for values of x in the detector
        args
        -----------
        powers: array, list
            list of spectrogram powers to calcualte statistic at
        returns
        ---------
        signoiseline: log(sig/(noise+line))
        signoise: log(sig/(noise))
        sigline: log(sig/(line))
        sig: log(sig)
        noise: log(noise)
        line: log(line)
        """

        signoiseline = np.zeros((len(powers)))
        signoise = np.zeros((len(powers)))
        sigline = np.zeros((len(powers)))
        sig = np.zeros((len(powers)))
        noise = np.zeros((len(powers)))
        line = np.zeros((len(powers)))
        
        for i in range(len(powers)):
            ig = self.one_det(powers[i],signal_prior_width,line_prior_width,noise_line_model_ratio,approx=approx)
            signoiseline[i] = ig[0]
            signoise[i] = ig[1]
            sigline[i] = ig[2]
            sig[i] = ig[3]
            noise[i] = ig[4]
            line[i] = ig[4]
                    
        return signoiseline,signoise,sigline,sig,noise,line

    def gen_lookup_two_det(self,powers1,powers2,approx=True,signal_prior_width=10,line_prior_width=10,noise_line_model_ratio=1):
        """
        calculate lookup table for values of x and y in each detector
        args
        ----------
        powers1: array, list
            list of spectrogram powers to calcualte statistic at in det1
        powers2: array, list
            list of spectrogram powers to calcualte statistic at in det2


        returns
        ---------
        signoiseline: log(sig/(noise+line))
        signoise: log(sig/(noise))
        sigline: log(sig/(line))
        sig: log(sig)
        noise: log(noise)
        line: log(line)
        """

        signoiseline = np.zeros((len(powers1),len(powers2)))
        signoise = np.zeros((len(powers1),len(powers2)))
        sigline = np.zeros((len(powers1),len(powers2)))
        sig = np.zeros((len(powers1),len(powers2)))
        noise = np.zeros((len(powers1),len(powers2)))
        line = np.zeros((len(powers1),len(powers2)))
        
        for i in range(len(powers1)):
            for j in range(len(powers2)):
                if j > i:
                    continue
                ig = self.two_det(powers1[i],powers2[j],signal_prior_width,line_prior_width,noise_line_model_ratio,approx=approx)
                # symmetric matrix so set opposite elements to same, format it [(x1,x2),(y1,y2)]
                signoiseline[(i,j),(j,i)] = ig[0]
                signoise[(i,j),(j,i)] = ig[1]
                sigline[(i,j),(j,i)] = ig[2]
                sig[(i,j),(j,i)] = ig[3]
                noise[(i,j),(j,i)] = ig[4]
                line[(i,j),(j,i)] = ig[4]
                    
        return signoiseline,signoise,sigline,sig,noise,line


    def save_lookup(self,outdir,log=True, stat_type = "signoiseline"):
        """
        save the lookup table for two detectors with the line aware statistic
        
        Args
        --------------
        outdir: string
        directory to save lookup table file
        pow_range: tuple
        ranges for the spectrogram power (lower, upper, number), default (1,400,500)
        
        """
        minimum,maximum,num = min(self.powers),max(self.powers),len(self.powers)
        log_str = "log_" if log else ""
        fname = os.path.join(outdir,"{}{}_{}det_{}degfree_{}_{}_{}.pkl".format(log_str,stat_type,self.ndet,self.kN,self.signal_prior_width,self.line_prior_width,self.noise_line_model_ratio))
        
        if os.path.isfile(fname):
            pass
        else:
            with open(fname,'wb') as f:
                header = {"power_ranges":(minimum,maximum,num), "fraction_ranges": (1, 1, 1), "signal_prior":self.signal_prior_width, "line_prior_width":self.line_prior_width, "noise_line_model_ratio":self.noise_line_model_ratio}
                if log:
                    pickle.dump([np.log(getattr(self,stat_type)), header],f)
                elif not log:
                    pickle.dump([getattr(self,stat_type), header],f)


    def save_multiple(self,powerrange,signal_prior_widthrange,line_prior_widthrange,noise_line_model_ratiorange):

        pass
                    
#------------------------------------
# Line aware statistic with consistent amplitude
# --------------------------------------



class LineAwareAmpStatistic:

    def __init__(self, powers, fractions, ndet=2 ,k=2, N=48, signal_prior_width=1, line_prior_width=5, noise_line_model_ratio=1,approx=True):

        self.ndet = ndet
        self.signal_prior_width = signal_prior_width
        self.line_prior_width = line_prior_width
        self.noise_line_model_ratio = noise_line_model_ratio
        self.powers = powers
        self.fractions = fractions
        self.k = k
        self.N = N
        self.kN = self.k*self.N
        
        self.signoiseline,self.signoise,self.sigline,self.sig,self.noise,self.line = self.gen_lookup_two_det(powers,powers,fractions,approx=approx,signal_prior_width=signal_prior_width,line_prior_width=line_prior_width,noise_line_model_ratio=noise_line_model_ratio)
            
    def chi2_sig(self,lamb,gs,pv,fraction):
        """
        returns the likelihood of two powers multiplied by the prior on the snr**2
        args
        -------
        lamb: float
            snr^2
        g1: float
            SFT power in detector 1
        g2: float
            SFT power in detector 2
        k: int
            number of degrees of freedom
        N: int
            Number of summed SFTS
        pv: float
            width of exponetial distribution of prior
        returns
        ---------
        func*prior: float
        likelihood multiplied by prior
        """
        if len(np.shape(gs)) == 0:
            func = st.ncx2.pdf(gs,df=self.kN,nc=lamb,loc=0,scale=1)
        else:
            # only for 2 detector
            if fraction > 1:
                lamb_list = [lamb*fraction,lamb]
            else:
                lamb_list = [lamb,lamb*fraction]

            func = np.prod([st.ncx2.pdf(gs[i],df=self.kN,nc=lamb_list[i],loc=0,scale=1) for i in range(len(gs))], axis=0)                
        wid = 1./pv
        return func*wid*np.exp(-wid*lamb)


    def chi2_line(self,lamb,gs,pv,fraction):
        """
        returns the likelihood of two powers multiplied by the prior on the snr**2
        works for 1 or 2 detectors only

        args
        -------
        lamb: float
        snr^2
        g1: float
        SFT power in detector 1
        g2: float
        SFT power in detector 2
        k: int
        number of degrees of freedom
        N: int
        Number of summed SFTS
        pv: float
        width of exponetial distribution of prior
        returns
        ---------
        func*prior: float
        likelihood multiplied by prior
        """
        if len(np.shape(gs)) == 0:
            func = st.ncx2.pdf(gs,df=self.kN,nc=lamb,loc=0,scale=1)
        else:
            if fraction > 1:
                func = 1./len(gs)*(st.chi2.pdf(gs[0],df=self.kN,loc=0,scale=1)*st.ncx2.pdf(gs[1],df=self.kN,nc=lamb*fraction,loc=0,scale=1) + st.ncx2.pdf(gs[0],df=self.kN,nc=lamb,loc=0,scale=1)*st.chi2.pdf(gs[1],df=self.kN,loc=0,scale=1))
            else:
                func = 1./len(gs)*(st.chi2.pdf(gs[0],df=self.kN,loc=0,scale=1)*st.ncx2.pdf(gs[1],df=self.kN,nc=lamb,loc=0,scale=1) + st.ncx2.pdf(gs[0],df=self.kN,nc=lamb*fraction,loc=0,scale=1)*st.chi2.pdf(gs[1],df=self.kN,loc=0,scale=1))

        wid = 1./pv
        return func*wid*np.exp(-wid*lamb)
    
    
    def chi2_noise(self,gs):
        if len(np.shape(gs)) == 0:
            func = st.chi2.pdf(gs,df=self.kN,loc=0,scale=1)
        else:
            func = np.prod([st.chi2.pdf(i,df=self.kN,loc=0,scale=1) for i in gs],axis=0)
        return func


    def two_det(self,g1,g2,fraction,signal_prior_width=10,line_prior_width=10,noise_line_model_ratio=1., approx=True):
        """
        integrate likelihood and prior to get evidence for powers g1 and g2
        args
        -------
        g1: float
        g2: float
        k: int
        N: int
        signal_prior_width: float
        line_prior_width: float
        noise_line_model_ratio: float
        approx: bool
            choose to approximate the integral with trapz or full integnoise_line_model_ration
        """
        if approx:
            l = np.linspace(0,100,500)
            sig_int = np.trapz(np.nan_to_num(self.chi2_sig(l,[g1,g2],fraction,signal_prior_width)),l)
            line_int = np.trapz(np.nan_to_num(self.chi2_line(l,[g1,g2],fraction,line_prior_width)),l)
        else:
            sig_int, sig_err = quad(self.chi2_sig,0,np.inf, args = ([g1,g2],fraction,signal_prior_width))
            line_int, line_err = quad(self.chi2_line,0,np.inf, args = ([g1,g2],fraction,line_prior_width))

        noise = self.chi2_noise([g1,g2])
        siglinenoise = sig_int/(noise_line_model_ratio*line_int + noise )
        signoise = sig_int/noise
        sigline = sig_int/line_int
        
        return siglinenoise,signoise,sigline,sig_int,noise,line_int

    def gen_lookup_two_det(self,powers1,powers2,fractions,approx=True,signal_prior_width=10,line_prior_width=10,noise_line_model_ratio=1):
        """
        calculate lookup table for values of x and y in each detector
        args
        ----------
        powers1: array, list
            list of spectrogram powers to calcualte statistic at in det1
        powers2: array, list
            list of spectrogram powers to calcualte statistic at in det2


        returns
        ---------
        signoiseline: log(sig/(noise+line))
        signoise: log(sig/(noise))
        sigline: log(sig/(line))
        sig: log(sig)
        noise: log(noise)
        line: log(line)
        """

        signoiseline = np.zeros((len(powers1),len(powers2), len(fractions)))
        signoise = np.zeros((len(powers1),len(powers2),len(fractions)))
        sigline = np.zeros((len(powers1),len(powers2),len(fractions)))
        sig = np.zeros((len(powers1),len(powers2),len(fractions)))
        noise = np.zeros((len(powers1),len(powers2),len(fractions)))
        line = np.zeros((len(powers1),len(powers2),len(fractions)))
        print(np.shape(signoiseline))
        for i in range(len(powers1)):
            for j in range(len(powers2)):
                for k in range(len(fractions)):
                    if j > i:
                        continue
                    ig = self.two_det(powers1[i],powers2[j],fractions[k],signal_prior_width,line_prior_width,noise_line_model_ratio,approx=approx)
                    # symmetric matrix so set opposite elements to same, format it [(x1,x2),(y1,y2)]
                    signoiseline[(i,j),(j,i),(k,k)] = ig[0]
                    signoise[(i,j),(j,i),(k,k)] = ig[1]
                    sigline[(i,j),(j,i),(k,k)] = ig[2]
                    sig[(i,j),(j,i),(k,k)] = ig[3]
                    noise[(i,j),(j,i),(k,k)] = ig[4]
                    line[(i,j),(j,i),(k,k)] = ig[4]
                    
        return signoiseline,signoise,sigline,sig,noise,line


    def save_lookup(self,outdir,log=True, stat_type = "signoiseline"):
        """
        save the lookup table for two detectors with the line aware statistic
        
        Args
        --------------
        outdir: string
            directory to save lookup table file
        
        """
        minimum,maximum,num = min(self.powers),max(self.powers),len(self.powers)
        minimumf,maximumf,numf = min(self.fractions),max(self.fractions),len(self.fractions)
        log_str = "log_" if log else ""
        fname = os.path.join(outdir,"{}{}amp_{}det_{}degfree_{}_{}_{}.pkl".format(log_str,stat_type,self.ndet,self.kN,self.signal_prior_width,self.line_prior_width,self.noise_line_model_ratio))
        
        if os.path.isfile(fname):
            pass
        else:
            with open(fname,'wb') as f:
                header = {"power_ranges":(minimum,maximum,num), "fraction_ranges": (minimumf, maximumf, numf), "signal_prior":self.signal_prior_width, "line_prior_width":self.line_prior_width, "noise_line_model_ratio":self.noise_line_model_ratio}
                if log:
                    pickle.dump([np.log(getattr(self,stat_type)), header],f)
                elif not log:
                    pickle.dump([getattr(self,stat_type), header],f)

                    



class LineAwareAmpStatistic_old:

    def __init__(self, powers, ndet=2, fractions = None, k = 2, N = 48, signal_prior_width = 1, line_prior_width = 5, noise_line_model_ratio=1, approx=True):
        """

        """
        self.ndet = ndet
        self.signal_prior_width = signal_prior_width
        self.line_prior_width = line_prior_width
        self.noise_line_model_ratio = noise_line_model_ratio
        self.powers = powers
        self.fractions = fractions
        self.k = k
        self.N = N
        self.kN = self.k*self.N

        self.noisefunc = st.chi2(df=self.N*self.k,loc=0,scale=1)
        
        if ndet == 1:
            raise Exception("This function does not yet exist")
            self.signoiseline,self.signoise,self.sigline,self.sig,self.noise,self.line = self.gen_lookup_one_det(powers,fractions,approx=approx,signal_prior_width=signal_prior_width,line_prior_width=line_prior_width,noise_line_model_ratio=noise_line_model_ratio)
        elif ndet == 2:
            self.signoiseline,self.signoise,self.sigline,self.sig,self.noise,self.line = self.gen_lookup_two_det(powers,fractions, approx=approx,signal_prior_width=self.signal_prior_width,line_prior_width=self.line_prior_width,noise_line_model_ratio=self.noise_line_model_ratio)


    def chi2_sig(self,lamb,g1,g2,pv,fraction):
        """
        function to generate the signal probability as a function of the power in each detector. includes a factor which includes the noise disrtibution and duty cycle for each of the detectors. works for 2 detectors.
        args
        ---------
        lamb: float
            factor proportional to the SNR
        g1: float
            SFT power in detector 1
        g2: float
            SFT power in detector 2
        k: int
            number of degrees of freedom for chi2 distribution
        N: int
            number of summed SFTs 
        pv: float
            width of the exponentional distribution used for the prior on lamb
        fraction: float
            noise_line_model_ratio of the detectors duty cycle and psd, \frac{S_2 f_1}{S_1 f_2}
        returns
        ----------
        prob: float
            probability of these two powers being a signal given the detector powers, duty cycles and noise psd
        """
        if fraction>=1:
            func = st.ncx2.pdf(g1,df=self.kN,nc=lamb,loc=0,scale=1)*st.ncx2.pdf(g2,df=self.kN,nc=lamb*fraction,loc=0,scale=1)
        else:
            func = st.ncx2.pdf(g1,df=self.kN,nc=lamb/fraction,loc=0,scale=1)*st.ncx2.pdf(g2,df=self.kN,nc=lamb,loc=0,scale=1)

        wid = 1./pv
        return func*wid*np.exp(-wid*lamb)

    def chi2_line(self,lamb,g1,g2,pv,fraction):
        """
        function to generate the line probability as a function of the power in each detector. includes a factor which includes the noise disrtibution and duty cycle for each of the detectors. works for 2 detectors.
        args
        ---------
        lamb: float
            factor proportional to the SNR
        g1: float
            SFT power in detector 1
        g2: float
            SFT power in detector 2
        k: int
            number of degrees of freedom for chi2 distribution
        N: int
            number of summed SFTs 
        pv: float
            width of the exponentional distribution used for the prior on lamb
        fraction: float
            noise_line_model_ratio of the detectors duty cycle and psd, \frac{S_2 f_1}{S_1 f_2}
        returns
        ----------
        prob: float
            probability of these two powers being a noise line given the detector powers, duty cycles and noise psd
        """
        ch2_v = self.noisefunc.pdf([g1,g2])
        
        if fraction >= 1:
            func = ch2_v[0]*st.ncx2.pdf(g2,df=self.kN,nc=lamb*fraction,loc=0,scale=1) + st.ncx2.pdf(g1,df=self.kN,nc=lamb,loc=0,scale=1)*ch2_v[1]
        else:
            func = ch2_v[0]*st.ncx2.pdf(g2,df=self.kN,nc=lamb,loc=0,scale=1) + st.ncx2.pdf(g1,df=self.kN,nc=lamb/fraction,loc=0,scale=1)*ch2_v[1]

        wid = 1./pv
        return 0.5*func*wid*np.exp(-wid*lamb)

    def chi2_noise(self,g1,g2):
        """
        function to generate the noise probability as a function of the power in each detector.
        args
        ---------
        g1: float
            SFT power in detector 1
        g2: float
            SFT power in detector 2
        k: int
            number of degrees of freedom for chi2 distribution
        N: int
            number of summed SFTs 
        returns
        ----------
        prob: float
            probability of these two powers being noise given the detector powers
        """
        func = np.prod(self.noisefunc.pdf([g1,g2]))
        return func
    
    def line_aware_amp(self,g1,g2,fraction,signal_prior_width=10,line_prior_width=10,noise_line_model_ratio=1.):
        """
        marginalises over the "SNR" for both the noise and line case and combines into an odds noise_line_model_ratio(using quad)
        args
        ---------
        g1: float
            SFT power in detector 1
        g2: float
            SFT power in detector 2
        k: int
            number of degrees of freedom for chi2 distribution
        N: int
            number of summed SFTs 
        signal_prior_width: float
            width of the exponentional distribution used for the prior on lamb in the signal case
        line_prior_width: float
            width of the exponentional distribution used for the prior on lamb in the line case
        fraction: float
            noise_line_model_ratio of the detectors duty cycle and psd, \frac{S_2 f_1}{S_1 f_2}
        noise_line_model_ratio: float
            noise_line_model_ratio of the prior on the line model to the gaussian noise model, \frac{M_{N}}{M_{G}}
        returns
        ----------
        bsgl1: float
            odds noise_line_model_ratio of all three models \frac{M_{S}}{M_{L} + M_{G}}
        bsgl2: float
            odds noise_line_model_ratio of signal and gaussian noise models \frac{M_{S}}{M_{G}}
        bsgl3: float
            odds noise_line_model_ratio of signal and line models \frac{M_{S}}{M_{L}}
        sig_int: float
            likelihood of signal model
        noise_func: float
            likelihood of gaussian noise model
        line_int: float
            likelihood of line model
        """
        noise = self.chi2_noise(g1,g2)
        sig_int, sig_err = quad(self.chi2_sig,0,np.inf, args = (g1,g2,signal_prior_width,fraction))
        line_int, line_err = quad(self.chi2_line,0,np.inf, args = (g1,g2,line_prior_width,fraction))
        bsgl1 = sig_int/(noise_line_model_ratio*line_int + noise)
        bsgl2 = sig_int/(noise)
        bsgl3 = sig_int/line_int
        return bsgl1,bsgl2,bsgl3,sig_int,noise,line_int

    def line_aware_amp_approx(self,g1,g2,fraction,k,N,signal_prior_width=1,line_prior_width=1,noise_line_model_ratio=1.):
        """
        marginalises over the "SNR" for both the noise and line case and combines into an odds noise_line_model_ratio (approximate using trapz)
        args
        ---------
        g1: float
            SFT power in detector 1
        g2: float
            SFT power in detector 2
        k: int
            number of degrees of freedom for chi2 distribution
        N: int
            number of summed SFTs 
        signal_prior_width: float
            width of the exponentional distribution used for the prior on lamb in the signal case
        line_prior_width: float
            width of the exponentional distribution used for the prior on lamb in the line case
        fraction: float
            noise_line_model_ratio of the detectors duty cycle and psd, \frac{S_2 f_1}{S_1 f_2}
        noise_line_model_ratio: float
            noise_line_model_ratio of the prior on the line model to the gaussian noise model, \frac{M_{N}}{M_{G}}
        returns
        ----------
        bsgl1: float
            odds noise_line_model_ratio of all three models \frac{M_{S}}{M_{L} + M_{G}}
        bsgl2: float
            odds noise_line_model_ratio of signal and gaussian noise models \frac{M_{S}}{M_{G}}
        bsgl3: float
            odds noise_line_model_ratio of signal and line models \frac{M_{S}}{M_{L}}
        sig_int: float
            likelihood of signal model
        noise_func: float
            likelihood of gaussian noise model
        line_int: float
            likelihood of line model
        """
        l = np.linspace(0,100,500)
        noise = self.chi2_noise(g1,g2)
        sig_int = np.trapz(np.nan_to_num(self.chi2_sig(l,g1,g2,signal_prior_width,fraction)),l)
        line_int = np.trapz(np.nan_to_num(self.chi2_line(l,g1,g2,line_prior_width,fraction)),l)
        signoiseline = sig_int/(noise_line_model_ratio*line_int + noise)
        signoise = sig_int/noise
        sigline = sig_int/line_int
        return signoiseline,signoise,sigline,sig_int,noise,line_int

    def line_aware_amp_cpp(self,g1,g2,logfrac,signal_prior_width=1,line_prior_width=1,noise_line_model_ratio=1.):
        """
        marginalises over the "SNR" for both the noise and line case and combines into an odds noise_line_model_ratio (approximate using trapz)
        args
        ---------
        g1: float
            SFT power in detector 1
        g2: float
            SFT power in detector 2
        k: int
            number of degrees of freedom for chi2 distribution
        N: int
            number of summed SFTs 
        signal_prior_width: float
            width of the exponentional distribution used for the prior on lamb in the signal case
        line_prior_width: float
            width of the exponentional distribution used for the prior on lamb in the line case
        logfrac: float
            noise_line_model_ratio of the detectors duty cycle and psd, \frac{S_2 f_1}{S_1 f_2}
        noise_line_model_ratio: float
            noise_line_model_ratio of the prior on the line model to the gaussian noise model, \frac{M_{N}}{M_{G}}
        returns
        ----------
        bsgl1: float
            log odds noise_line_model_ratio of all three models \frac{M_{S}}{M_{L} + M_{G}}
        bsgl2: float
            log odds noise_line_model_ratio of signal and gaussian noise models \frac{M_{S}}{M_{G}}
        bsgl3: float
            log odds noise_line_model_ratio of signal and line models \frac{M_{S}}{M_{L}}
        sig_int: float
            log likelihood of signal model
        noise_func: float
            log likelihood of gaussian noise model
        line_int: float
            log likelihood of line model
        """
        sig_int = integrals.Integral([g1],g2,logfrac,self.k,self.N,signal_prior_width,"signal")
        line_int = integrals.Integral([g1],g2,logfrac,self.k,self.N,line_prior_width,"line")
        bsgl1 = sig_int/(noise_line_model_ratio*line_int + chi2_noise(g1,g2,self.k,self.N))
        bsgl2 = sig_int/(chi2_noise(g1,g2,self.k,self.N))
        bsgl3 = sig_int/line_int
        return np.log(bsgl1),np.log(bsgl2),np.log(bsgl3),np.log(sig_int),np.log(chi2_noise(g1,g2,k,N)),np.log(line_int)

    
    def gen_lookup_two_det(self,powers,fractions,approx=True,signal_prior_width=1,line_prior_width=1,noise_line_model_ratio=1):
        """
        calculates the odds noise_line_model_ratio in ch_integrals_approx_noise for given values of power and noise_line_model_ratio.
        args
        --------
        x: array
            array of the SFT power in detector 1
        y: array
            array of SFT power in detector 2
        n1: array
            array of the noise_line_model_ratios of the duty cycle and noise psds, \frac{f_{1} S_{2}}{f_{2} S_{1}}
        int_type: string
            which statistic to use ("chi2" only option for now)
        approx: bool
            choose whether to use trapz or quad to complete integrals
        signal_prior_width: float
            width on prior for signal model
        line_prior_width: float
            width on prior for line model
        noise_line_model_ratio: float
            noise_line_model_ratio of the priors on the line and signal models
        returns
        ----------
        val: array
            array of odd noise_line_model_ratio with 3 models
        val1: array
            array of odd noise_line_model_ratio with signal/noise model
        val2: array
            array of odd noise_line_model_ratio with signal/line model
        val3: array
            array of signal likelihoods
        val4: array
            array of noise likelihoods
        val: array
            array of line likelihoods
        """
        signoiseline = np.zeros((len(fractions),len(powers),len(powers)))
        signoise = np.zeros((len(fractions),len(powers),len(powers)))
        sigline = np.zeros((len(fractions),len(powers),len(powers)))
        sig = np.zeros((len(fractions),len(powers),len(powers)))
        noise = np.zeros((len(fractions),len(powers),len(powers)))
        line = np.zeros((len(fractions),len(powers),len(powers)))
        for i in range(len(powers)):
            for j in range(len(powers)):
                if j > i:
                    continue
                for k in range(len(fractions)):
                    if approx:
                        las = self.line_aware_amp_approx(powers[i],powers[j],fractions[k],signal_prior_width,line_prior_width,noise_line_model_ratio)
                    else:
                        las = self.line_aware_amp(powers[i],powers[j],fractions[k],signal_prior_width,line_prior_width,noise_line_model_ratio)

                    signoiseline[(k,k),(i,j),(j,i)] = las[0]
                    signoise[(k,k),(i,j),(j,i)] = las[1]
                    sigline[(k,k),(i,j),(j,i)] = las[2]
                    sig[(k,k),(i,j),(j,i)] = las[3]
                    noise[(k,k),(i,j),(j,i)] = las[4]
                    line[(k,k),(i,j),(j,i)] = las[5]
                    
        return signoiseline,signoise,sigline,sig,noise,line

    def gen_lookup_one_det(self,powers,fractions,approx=True,signal_prior_width=1,line_prior_width=1,noise_line_model_ratio=1):
        """
        calculates the odds noise_line_model_ratio in ch_integrals_approx_noise for given values of power and noise_line_model_ratio.
        args
        --------
        x: array
            array of the SFT power in detector 1
        y: array
            array of SFT power in detector 2
        n1: array
            array of the noise_line_model_ratios of the duty cycle and noise psds, \frac{f_{1} S_{2}}{f_{2} S_{1}}
        int_type: string
            which statistic to use ("chi2" only option for now)
        approx: bool
            choose whether to use trapz or quad to complete integrals
        signal_prior_width: float
            width on prior for signal model
        line_prior_width: float
            width on prior for line model
        noise_line_model_ratio: float
            noise_line_model_ratio of the priors on the line and signal models
        returns
        ----------
        val: array
            array of odd noise_line_model_ratio with 3 models
        val1: array
            array of odd noise_line_model_ratio with signal/noise model
        val2: array
            array of odd noise_line_model_ratio with signal/line model
        val3: array
            array of signal likelihoods
        val4: array
            array of noise likelihoods
        val: array
            array of line likelihoods
        """
        signoiseline = np.zeros((len(fractions),len(powers),len(powers)))
        signoise = np.zeros((len(fractions),len(powers),len(powers)))
        sigline = np.zeros((len(fractions),len(powers),len(powers)))
        sig = np.zeros((len(fractions),len(powers),len(powers)))
        noise = np.zeros((len(fractions),len(powers),len(powers)))
        line = np.zeros((len(fractions),len(powers),len(powers)))
        for i in range(len(powers)):
            for j in range(len(powers)):
                if j > i:
                    continue
                for k in range(len(fractions)):
                    if approx:
                        las = self.line_aware_amp_approx(powers[i],powers[j],fractions[k],signal_prior_width,line_prior_width,noise_line_model_ratio)
                    else:
                        las = self.line_aware_amp(powers[i],powers[j],fractions[k],signal_prior_width,line_prior_width,noise_line_model_ratio)

                    signoiseline[(k,k),(i,j),(j,i)] = las[0]
                    signoise[(k,k),(i,j),(j,i)] = las[1]
                    sigline[(k,k),(i,j),(j,i)] = las[2]
                    sig[(k,k),(i,j),(j,i)] = las[3]
                    noise[(k,k),(i,j),(j,i)] = las[4]
                    line[(k,k),(i,j),(j,i)] = las[5]
                    
        return signoiseline,signoise,sigline,sig,noise,line
            
    def gen_lookup_cpp(self,g1,g2,logfrac,signal_prior_width,line_prior_width,noise_line_model_ratio):
        """
        calculates the odds noise_line_model_ratio in ch_integrals_approx_noise for given values of power and noise_line_model_ratio.
        args
        --------
        g1: array
            array of the SFT power in detector 1
        g1: array
            array of SFT power in detector 2
        logfrac: array
            array of the noise_line_model_ratios of the duty cycle and noise psds, \frac{f_{1} S_{2}}{f_{2} S_{1}}
        signal_prior_width: float
            width on prior for signal model
        line_prior_width: float
            width on prior for line model
        noise_line_model_ratio: float
            noise_line_model_ratio of the priors on the line and signal models
        returns
        ----------
        val: array
            array of odd noise_line_model_ratio with 3 models
        val1: array
            array of signal likelihoods
        val2: array
            array of line likelihoods
        val3: array
            array of noise likelihoods
        """
        return np.log(integrals.Integral(list(g1),list(g2),list(logfrac),self.k,self.N,signal_prior_width,line_prior_width,noise_line_model_ratio))


    def save_lookup(self,outdir,stat_type="signoiseline",log=True):
        """
        save the lookup table for two detectors with the line aware statistic
        
        Args
        --------------
        outdir: string
        directory to save lookup table file
        pow_range: tuple
        ranges for the spectrogram power (lower, upper, number), default (1,400,500)
        
        """
        minimum,maximum,num = min(self.powers),max(self.powers),len(self.powers)
        minimum_frac,maximum_frac,num_frac = min(self.fractions),max(self.fractions),len(self.fractions)
        log_str = "log_" if log else ""
        fname = os.path.join(outdir,"{}{}amp_{}det_{}degfree_{}_{}_{}.pkl".format(log_str,stat_type,self.ndet,self.kN,self.signal_prior_width,self.line_prior_width,self.noise_line_model_ratio))
        if os.path.isfile(fname):
            pass
        else:
            with open(fname,'wb') as f:
                header = {"power_ranges":(minimum,maximum,num), "fraction_ranges": (minimum_frac, maximum_frac, num_frac), "signal_prior":self.signal_prior_width, "line_prior_width":self.line_prior_width, "noise_line_model_ratio":self.noise_line_model_ratio}
                if log:
                    pickle.dump([np.log(getattr(self,stat_type)), header],f)
                elif not log:
                    pickle.dump([getattr(self,stat_type), header],f)



                
