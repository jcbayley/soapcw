# cython: language_level=3
from __future__ import division
import numpy as np
import timeit
import itertools
import time
import pickle
import copy
from sys import stdout
from libc.math cimport log,exp,sqrt 
#from scipy.interpolate import interp2d,RectBivariateSpline
from scipy.special import logsumexp

class single_detector(object):
 
    def __init__(self, tr, obs, prog = False, lookup_table=None, make_vitmap = True):
        '''
        initialising viterbi class
        '''
        self.prog = prog
        
        # make sure input data of right type
        tr = np.array(tr).astype('double')
        obs = np.array(obs).astype('double')

        # make sure input data of right shape
        if len(np.shape(tr)) != 1:
            raise Exception("transition matrix wrong shape, should be Kx1 array")
        if len(np.shape(obs)) != 2:
            raise Exception("observation wrong shape, should be NxM array")
        
        self.run(tr, obs, lookup_table)
        if make_vitmap:
            self.get_vitmap()

        
    #------------------------------------------------------------------------------------------------
    # Get Viterbi map
    #------------------------------------------------------------------------------------------------

    def get_vitmap(self,log=False):
        """
        normalise the viterbi output such that the sum of eah column is 1
        """
        path_m = []
        for i in self.V:
            sump = logsumexp(i)
            val = np.exp(i-sump)
            path_m.append(val)
        path_m = np.array(path_m)
        if log:
            path_m = np.log(path_m)

        self.vitmap = path_m

    #------------------------
    # Read lookup table
    #--------------------------
    
    def read_lookup(self,lookup_table):
        """
        Read the lookup table for chosen statistic
        """
        if isinstance(lookup_table, str):
            with open(lookup_table,'rb') as f1:
                likelihood, params = pickle.load(f1)
            ranges = np.linspace(params["power_ranges"][0],params["power_ranges"][1],int(params["power_ranges"][2]))
            fact = abs(1./(ranges[0] - ranges[1]))
        else:
            # if it is a line aware stat object then creat ranges from the object
            # get the log odds ratio as the statitsic to use 
            likelihood = np.log(lookup_table.signoiseline)
            ranges = lookup_table.powers
            fact = abs(1./(ranges[0] - ranges[1]))
        
        return likelihood, ranges, fact
    
    
    #-------------------------------------------------------------------------------------------------
    # Basic Viterbi Algorithm
    #-------------------------------------------------------------------------------------------------

    
    def run(self,double[:] tr, double[:, :] obs, lookup_table = None):
        '''
        Run the viterbi algorithm for given single set of data and 3x1 transition matrix.
        This returns the track through the data which gives the largest sum of power.
        Args
        ------
        tr: array
            transition matrix of sixe Kx1
        obs: array
            observation data of size NxM
        Returns
        -------
        vit_track: array
            index of the viterbi path found in data
        max_end_prob: float
            maximum probability at end of path
        V: array
           viterbi matrix
        prev: array
            previous positions of paths for each bin
        '''
        
        # find shape of the observation and create empty array for citerbi matrix and previous track positions
        shape = np.shape(obs)
        cdef double[:, :] V = np.zeros(shape)
        cdef int[:, :] prev = np.zeros(shape,dtype=np.int32)
        
        # defining variables
        cdef int t,i,j         # indexes
        cdef double pbar = 0   # progress bar
        cdef double pt
        
        # finding length and width of observation
        cdef int length = shape[0]#len(obs)
        cdef int width = shape[1]#len(obs[0])
        
        # find size of the transition matrix
        cdef int tr_length = len(tr)
        # find half width of transntion, i.e. number of bins up and down it can move
        cdef int tr_width = int((tr_length-1)/2 )

        # vectors to store lookup tables
        cdef double[:] logarr
        cdef double fact
        cdef int[:,:] obs_ind
        if lookup_table is not None:
            # get log statistic, the ranges of lookuptable and spacing factor
            logarr, ranges, fact = self.read_lookup(lookup_table)
            # shift observation to index for array
            obs_ind_arr = (obs - ranges[0])*fact
            obs_ind_arr[obs_ind_arr >= len(ranges)] = len(ranges)-1
            obs_ind_arr[obs_ind_arr < 0] = 0
            # save array to vector
            obs_ind = np.array(obs_ind_arr).astype(np.int32)
            del obs_ind_arr
        else:
            # if using the sum of power, set element of index to 2d array
            obs_ind = np.arange(np.prod(shape)).reshape(shape).astype(np.int32)
            # flatten input spectrogram for access in 1d
            logarr = np.ravel(obs)
        
        # run for first time index, i.e. fill with observation
        for i in range(width):
            V[0][i] = logarr[obs_ind[0][i]]
            prev[0][i] = 1
        
        # run iterative part of algorithm
        for t in range(1,length):
            pt = t/(length)*100
            for i in range(width):
                temp = -1e6
                for j in range(tr_length):
                    if i+j-tr_width>=0 and i+j-tr_width<=width-1:
                        o = obs_ind[t][i]
                        value = tr[j] + logarr[o] + V[t-1][i+j-tr_width]

                        if value>temp:
                            temp = value
                            V[t][i] = temp
                            prev[t][i] = i+j-tr_width
                        elif value == temp and j == int(tr_length/2.):
                            temp = value
                            V[t][i] = temp
                            prev[t][i] = i+j-tr_width
                            

            if self.prog == True:
                if pt>pbar:
                    stdout.write('\r{} %'.format(round(pt)))
                    stdout.flush()
                    pbar+=100./len(obs)
    
        cdef double max_end_prob = max(V[length-1][i] for i in range(width))
        cdef int previous
        cdef int[:] vit_track = np.zeros(length,dtype=np.int32)
        
                        
        for i in range(width):
            if V[length-1][i] == max_end_prob:
                vit_track[length-1] = i # appends maximum path value from final step
                previous = prev[length-1][i]
                break

        for t in range(len(V)-2,-1,-1):
            vit_track[t] = previous # insert previous step
            previous = prev[t][previous]

        self.vit_track = np.array(vit_track)
        self.max_end_prob = max_end_prob
        self.V = np.array(V)
        self.prev = np.array(prev)
        

class two_detector(object):
 
    def __init__(self, tr, obs1, obs2, lookup_table_2det = None, lookup_table_1det = None, prog = False,fractions=None, make_vitmap=True):
        '''
        viterbi algorithm for two detectors with an optional line aware statistic
        uses S = \frac{p(ss \mid x)}{p(nn \mid x)+p(ns \mid x)+p(sn \mid x)}
        [currently configured for averaging over 48 SFT need to change to correctly use any]
        Args
        ---------
        tr1: array
            transition matrix
        obs11: array
            observation from detector 1 (float64 array)
        obs21: array
            observation from detector 2 (float64 array)
        lookup_table_1det: string (optional)
            filepath of lookup table for the one detector case or LineAwareStatistic claess
        lookup_table_2det: string ( optional)
            filepath of lookup table for the two detector caseor LineAwareStatistic claess 
        prog: bool (optional)
            show progress bar if True 
        fractions: array (optional)
            array of the ratio of the noise floor and amount of data between detectors
        returns
        -----------
        vit_track1: array
            path in detector 1
        vit_track2: array
            path in detector2
        vit_track_ref: array
            path in reference detector
        V: array
            viterbi values in reference detector
        prev: array
            previous positions before jump
        max_prob: double
            maximum end path probability

        '''

        if len(np.shape(obs1)) < 2 or len(np.shape(obs1)) < 2:
            raise Exception("Please input 2D array for observation data")
        if np.shape(obs1) != np.shape(obs2):
            raise Exception("Please make sure teh two input data have the same dimensions")

        tr = np.array(tr).astype('double')
        obs1 = np.array(obs1).astype('double')
        obs2 = np.array(obs2).astype('double')
        self.prog = prog

        if fractions is not None and lookup_table_2det is not None:
            self.run_lookup_amp(tr, obs1, obs2,lookup_table=lookup_table_2det,fractions=fractions)
        else:
            self.run(tr, obs1, obs2,lookup_table_1det=lookup_table_1det, lookup_table_2det=lookup_table_2det)
            
        self.get_track()
        if make_vitmap:
            self.get_vitmap()

    #------------------------------------------------------------------------------------------------
    # Get Viterbi map
    #------------------------------------------------------------------------------------------------

    def get_vitmap(self,log=False):
        """
        normalise the viterbi output such that the sum of eah column is 1
        """
        path_m = []
        for i in self.V:
            sump = logsumexp(i)
            val = np.exp(i-sump)
            path_m.append(val)
        path_m = np.array(path_m)
        if log:
            path_m = np.log(path_m)

        self.vitmap = path_m


    #--------------------------------------------------------
    # Read lookup tables
    #--------------------------------------------------------

    def read_lookup(self,lookup_table):
        """
        Read a lookup table from a file of a LineAwareStatistic object
        args 
        ------
        lookup_table: string or LineAwareStatistic
            filename to saved lookupt table or lineaware statistic object
        returns
        ---------
        likelihood: array
            returns odds ratio from file to use as statistic
        ranges: array
            ranges of the x,y axis of the odds ratio
        fact: float
            spacing of the ranges
        """

        if isinstance(lookup_table, str):
            # if it is a filename the load the file and create ranges from that
            with open(lookup_table,'rb') as f1:
                likelihood, params = pickle.load(f1)

            ranges = np.linspace(params["power_ranges"][0],params["power_ranges"][1],int(params["power_ranges"][2]))
            fact = abs(1./(ranges[0] - ranges[1]))
        else:
            # if it is a line aware stat object then creat ranges from the object
            # get the log odds ratio as the statitsic to use 
            likelihood = np.log(lookup_table.signoiseline)
            ranges = lookup_table.powers
            fact = abs(1./(ranges[0] - ranges[1]))
        
        return likelihood, ranges, fact

    def read_lookup_amp(self,lookup_table):
        """
        Read a lookup table from a file of a LineAwareStatisticAmplitude object
        args 
        ------
        lookup_table: string or LineAwareStatisticAmplitude
            filename to saved lookupt table or lineaware statistic object
        returns
        ---------
        likelihood: array
            returns odds ratio from file to use as statistic
        pow_ranges: array
            ranges of the x,y axis of the odds ratio (spectrogram power)
        frac_ranges: array
            ranges of the x,y axis of the odds ratio ()
        pow_fact: float
            spacing of the ranges of power
        frac_fact: float
            spacing of the ranges of data fraction
        """
        if isinstance(lookup_table, str):  
            with open(lookup_table,'rb') as f1:
                likelihood, params = pickle.load(f1)

            pow_ranges = np.linspace(params["power_ranges"][0],params["power_ranges"][1],int(params["power_ranges"][2]))
            frac_ranges = np.linspace(params["fraction_ranges"][0],params["fraction_ranges"][1],int(params["fraction_ranges"][2]))
            pow_fact = abs(1./(pow_ranges[0] - pow_ranges[1]))
            frac_fact = abs(1./(frac_ranges[0] - frac_ranges[1]))
        else:
            # if it is a line aware stat object then creat ranges from the object
            # get the log odds ratio as the statitsic to use 
            likelihood = np.log(lookup_table.signoiseline)
            pow_ranges = lookup_table.powers
            frac_ranges = lookup_table.fractions
            pow_fact = abs(1./(pow_ranges[0] - pow_ranges[1]))
            frac_fact = abs(1./(frac_ranges[0] - frac_ranges[1]))
        
        return likelihood.T, pow_ranges, frac_ranges, pow_fact, frac_fact

    
    #---------------------------------------------------------
    # Two detector viterbi
    #--------------------------------------------------------

    def run(self,transition, obs1, obs2, lookup_table_1det = None,lookup_table_2det = None):
        """
        viterbi algorithm for two detectors with an lookup table for the line veto statistic

        Where there is only one detectors data use the single detector search with the same statistic
        Args
        ---------
        tr1: array
            transition matrix
        obs11: array
            observation from detector 1
        obs21: array
            observation from detector 2 
        lookup_table: float
            links to a file which contains a 2d lookup table
        returns
        -----------
        V: array
            viterbi values in reference detector
        prev: array
            previous positions before jump
        """
        
        cdef double[:, :, :] tr = transition
        #cdef double[:, :] obs1 = obs11
        #cdef double[:, :] obs2 = obs21
        
        shape = np.shape(obs1)
        
        cdef int length = min(len(obs1),len(obs2))
        cdef int width = len(obs1[0])
        
        cdef double[:, :] val = np.ones(shape)*-1e6
        cdef int[:, :] prev = np.zeros(shape,dtype = np.int32)
        cdef int[:, :] det1 = np.zeros(shape,dtype = np.int32)
        cdef int[:, :] det2 = np.zeros(shape,dtype = np.int32)
        
        cdef int tr_len = len(tr)
        cdef int sep_len = len(tr[0])
        cdef int tr_dist = int(len(tr)/2. - 0.5)
        cdef int sep_dist = int(len(tr)/2. - 0.5)
        
        cdef int i
        cdef int j
        cdef int k
        cdef int m
        cdef int t
        cdef int x
        cdef int o1
        cdef int o2

        
        cdef long max_pos
        cdef long[:] indicies
        cdef double max_val
        cdef double value
        cdef double temp
        cdef int edge_range = 0

        # initialise and load lookuptable files for both line aware statistics
        cdef double[:,:] logarr_2d
        cdef double[:] logarr_1d
        cdef double fact_1d
        cdef double fact_2d
        cdef int[:,:] obs1_ind_1d
        cdef int[:,:] obs1_ind_2d

        obs1data = np.array([len(np.unique(obs1seg)) for obs1seg in obs1]) != 1
        obs2data = np.array([len(np.unique(obs2seg)) for obs2seg in obs2]) != 1
        
        if lookup_table_1det is not None:
            logarr_1d, ranges_1d, fact_1d = self.read_lookup(lookup_table_1det)
            
            obs1_ind_arr_1d = (obs1 - ranges_1d[0])*fact_1d
            obs1_ind_arr_1d[obs1_ind_arr_1d >= len(ranges_1d)] = len(ranges_1d)-1
            obs1_ind_arr_1d[obs1_ind_arr_1d < 0] = 0
            obs1_ind_1d = np.array(obs1_ind_arr_1d).astype(np.int32)
            del obs1_ind_arr_1d

            obs2_ind_arr_1d = (obs2-ranges_1d[0])*fact_1d
            obs2_ind_arr_1d[obs2_ind_arr_1d >= len(ranges_1d)] = len(ranges_1d)-1
            obs2_ind_arr_1d[obs2_ind_arr_1d < 0] = 0
            obs2_ind_1d = np.array(obs2_ind_arr_1d).astype(np.int32)
            del obs2_ind_arr_1d

        if lookup_table_2det is not None:
            logarr, ranges_2d, fact_2d = self.read_lookup(lookup_table_2det)
            diag = copy.copy(np.diag(logarr))
            logarr_2d = logarr

            # normalise the input powers to the ranges of lookup table so they are now indices
            obs1_ind_arr_2d = (obs1 - ranges_2d[0])*fact_2d
            # set powers outside range to end index
            obs1_ind_arr_2d[obs1_ind_arr_2d >= len(ranges_2d)] = len(ranges_2d)-1
            obs1_ind_arr_2d[obs1_ind_arr_2d < 0] = 0
            obs1_ind_2d = np.array(obs1_ind_arr_2d).astype(np.int32)

            # same for second detector
            obs2_ind_arr_2d = (obs2-ranges_2d[0])*fact_2d
            obs2_ind_arr_2d[obs2_ind_arr_2d >= len(ranges_2d)] = len(ranges_2d)-1
            obs2_ind_arr_2d[obs2_ind_arr_2d < 0] = 0
            obs2_ind_2d = np.array(obs2_ind_arr_2d).astype(np.int32)

            if lookup_table_1det is None:
                obs1_ind_1d = np.array(obs1_ind_arr_2d).astype(np.int32)
                obs2_ind_1d = np.array(obs2_ind_arr_2d).astype(np.int32)
                logarr_1d = diag
            
            del diag    
            del obs2_ind_arr_2d
            del obs1_ind_arr_2d

        sumv = False
	# if no lookup tables provided then use the sum of the powers
        if lookup_table_1det is None and lookup_table_2det is None:
            # if using the sum of power, set element of index to 2d array
            obs1_ind_1d = np.arange(np.prod(shape)).reshape(shape).astype(np.int32)
            obs1_ind_2d = np.arange(np.prod(shape)).reshape(shape).astype(np.int32)
            obs2_ind_1d = np.arange(np.prod(shape)).reshape(shape).astype(np.int32) + np.prod(shape)
            obs2_ind_2d = np.arange(np.prod(shape)).reshape(shape).astype(np.int32) + np.prod(shape)
            # flatten input spectrogram for access in 1d
            logarr_1d = np.ravel([obs1,obs2])
            sumv = True
            #logarr_2d = np.ravel([obs1,obs2])

        # initial viterbi loop for first time segment
        for i in range(width):
            temp = -1e6
            if obs1data[0] and obs2data[0]:
                # if data in both detectors
                for k in range(sep_len): 
                    if i+k-sep_dist>=0+edge_range and i+k-sep_dist<width-edge_range:
                        for m in range(sep_len):
                            if i+m-sep_dist>=0+edge_range and i+m-sep_dist<width-edge_range:
                                if sumv:
                                    value = tr[1][k][m] + obs1[0][i+k-sep_dist] + obs2[0][i+k-sep_dist]
                                else:
                                    o1 = obs1_ind_2d[0][i+k-sep_dist]
                                    o2 = obs2_ind_2d[0][i+m-sep_dist]
                                    value = tr[1][k][m] + logarr_2d[o1,o2]
                                if value > temp:
                                    temp = value
                                    val[0][i] = temp
                                    prev[0][i] =  i
                                    det1[0][i] =  i+k-1 
                                    det2[0][i] =  i+m-1
                                else:
                                    continue

            elif obs1data[0] and not obs2data[0]:
                o1 = obs1_ind_1d[0][i]
                k,m = 1,1
                value = tr[1][k][m] + logarr_1d[o1]
            elif obs2data[0] and not obs1data[0]:
                o2 = obs2_ind_1d[0][i]
                k,m = 1,1
                value = tr[1][k][m] + logarr_1d[o2]
            elif not obs2data[0] and not obs1data[0]:
                o1 = obs1_ind_2d[0][i]
                k,m = 1,1
                value = tr[1][k][m] + logarr_1d[o1]

                
            if value > temp:
                temp = value
                val[0][i] = temp
                prev[0][i] =  i
                det1[0][i] =  i+k-1 
                det2[0][i] =  i+m-1
            else:
                continue

        # i is current frequency position
        # j is transition in reference detector
        # k is separation of first detector
        # m is separation of second detector

        # run main loop
        pbar = 0 # progressbar
        for t in range(1,length):
            pt = t/(float(length))*100. # progress bar status
            for i in range(width):
                temp = -np.inf
                for j in range(tr_len):
                    if i+j-tr_dist>=0 and i+j-tr_dist<width:
                        if obs1data[t] and obs2data[t]:
                            for k in range(sep_len):
                                if i+k-sep_dist>=0+edge_range and i+k-sep_dist<width-edge_range:
                                    for m in range(sep_len):
                                        if i+m-sep_dist>=0+edge_range and i+m-sep_dist<width-edge_range:
                                            if sumv:
                                                value = tr[j][k][m] + obs1[t][i+k-sep_dist] + obs2[t][i+k-sep_dist] + val[t-1][i+j-tr_dist]
                                            else:
                                                o1 = obs1_ind_2d[t][i+k-sep_dist]
                                                o2 = obs2_ind_2d[t][i+m-sep_dist]
                                                value = tr[j][k][m] + logarr_2d[o1,o2] + val[t-1][i+j-tr_dist]

                                            if value > temp:
                                                temp = value
                                                val[t][i] = temp
                                                prev[t][i] =  i+j-tr_dist
                                                det1[t][i] =  i+k-sep_dist
                                                det2[t][i] =  i+m-sep_dist
                                            else:
                                                continue
                        else:
                            if obs1data[t] and not obs2data[t]:
                                o1 = obs1_ind_1d[t][i]
                                k,m = 1,1
                                value = tr[j][k][m] + logarr_1d[o1] + val[t-1][i+j-tr_dist]
                            elif obs2data[t] and not obs1data[t]:
                                o2 = obs2_ind_1d[t][i]
                                k,m = 1,1
                                value = tr[j][k][m] + logarr_1d[o2] + val[t-1][i+j-tr_dist]
                            elif not obs2data[t] and not obs1data[t]:
                                o1 = obs1_ind_2d[t][i]
                                k,m = 1,1
                                value = tr[j][k][m] + logarr_1d[o1] + val[t-1][i+j-tr_dist]

                            
                            if value > temp:
                                temp = value
                                val[t][i] = temp
                                prev[t][i] =  i+j-tr_dist
                                det1[t][i] =  i+k-sep_dist
                                det2[t][i] =  i+m-sep_dist
                            else:
                                continue

            if self.prog == True:
                if pt>pbar:
                    stdout.write('\r{} %'.format(round(pt)))
                    stdout.flush()
                    pbar+=100./length

        self.V = np.array(val)
        self.prev = np.array(prev)
        self.det1 = np.array(det1)
        self.det2 = np.array(det2)
        self.width = width
        self.length = length

    def run_lookup_amp(self,tr1, obs11, obs21,lookup_table = None, fractions = None):
        """
        viterbi algorithm for two detectors with an lookup table for the line veto statistic
        Args
        ---------
        tr1: array
            transition matrix
        obs11: array
            observation from detector 1
        obs21: array
            observation from detector 2 
        lookup_table: float
            links to a file which contains a 2d lookup table or LineAwareStatistic class
        returns
        -----------
        V: array
            viterbi values in reference detector
        prev: array
            previous positions before jump
        """
        cdef double[:, :, :] tr = tr1
        cdef double[:, :] obs1 = obs11
        cdef double[:, :] obs2 = obs21
        
        shape = np.shape(obs1)
        
        cdef int length = min(len(obs1),len(obs2))
        cdef int width = len(obs1[0])
        
        cdef double[:, :] val = np.ones(shape)*-1e6
        cdef int[:, :] prev = np.zeros(shape,dtype = np.int32)
        cdef int[:, :] det1 = np.zeros(shape,dtype = np.int32)
        cdef int[:, :] det2 = np.zeros(shape,dtype = np.int32)
        
        cdef int tr_len = len(tr)
        cdef int sep_len = len(tr[0])
        cdef int tr_dist = int(len(tr)/2. - 0.5)
        cdef int sep_dist = int(len(tr)/2. - 0.5)
        
        cdef int i
        cdef int j
        cdef int k
        cdef int m
        cdef int t
        cdef int x

        cdef int o1
        cdef int o2
        
        cdef long max_pos
        cdef long[:] indicies
        cdef double max_val
        cdef double value
        cdef double temp
        cdef int edge_range = 0


        cdef double fact
        cdef double fact_frac

        likelihood, pow_ranges, frac_ranges, pow_fact, frac_fact = self.read_lookup_amp(lookup_table)

        # define the likelihood as two array one which is the transpose of the other, such that the fractions of time one detector can always be compared with a value less than 1
        shp = np.shape(likelihood)
        logarr1 = np.zeros((2,shp[0],shp[1],shp[2]))
        logarr1[0] = likelihood
        logarr1[1] = ([np.array(lkl).T for lkl in likelihood])
        cdef double[:,:,:,:] logarr = logarr1

        # if the fractions are above a value of 1, inverse the lookup table so that only values between 0 and 1 have to be generated.
        if len(np.shape(fractions)) == 1:
            fractions = np.array([np.ones(width)*i for i in fractions])
            
        det_order1 = np.ones(np.shape(fractions)).astype(np.int32)
        det_order1[fractions > 1] = int(1)
        det_order1[fractions <= 1] = int(0)
        cdef int[:,:] det_order = det_order1
        del det_order1
        
        fractions[fractions > 1] = 1./fractions[fractions > 1]
        
        # make sure the fractions are the same dimensions as the input data and convert into an index
        frac_ind1 = np.array((fractions-frac_ranges[0])*frac_fact).astype(np.int32)
        frac_ind1[frac_ind1 < 0] = 0
        frac_ind1[frac_ind1 >= len(frac_ranges)] = -1
        cdef int[:,:] frac_ind = frac_ind1
        del frac_ind1
        
        obs1_ind_arr = (obs11 - pow_ranges[0])*pow_fact
        obs1_ind_arr[obs1_ind_arr >= len(pow_ranges)] = -1
        obs1_ind_arr[obs1_ind_arr < 0] = 0
        cdef int[:,:] obs1_ind = np.array(obs1_ind_arr).astype(np.int32)
        del obs1_ind_arr
        
        obs2_ind_arr = (obs21-pow_ranges[0])*pow_fact
        obs2_ind_arr[obs2_ind_arr >= len(pow_ranges)] = -1
        obs2_ind_arr[obs2_ind_arr < 0] = 0
        cdef int[:,:] obs2_ind = np.array(obs2_ind_arr).astype(np.int32)
        del obs2_ind_arr
        
        for i in range(width):
            temp = -1e6
            for k in range(sep_len): 
                if i+k-sep_dist>=0+edge_range and i+k-sep_dist<width-edge_range:
                    for m in range(sep_len):
                        if i+m-sep_dist>=0+edge_range and i+m-sep_dist<width-edge_range:
                            o1 = int(obs1_ind[0][i+k-sep_dist])
                            o2 = int(obs2_ind[0][i+m-sep_dist])
                            value = tr[1][k][m] + logarr[det_order[0,i],frac_ind[0,i],o1,o2]
                            if value > temp:
                                temp = value
                                val[0][i] = temp
                                prev[0][i] =  i
                                det1[0][i] =  i+k-1 
                                det2[0][i] =  i+m-1
                            else:
                                continue

        # i is current frequency position
        # j is transition in reference detector
        # k is separation of first detector
        # m is separation of second detector


        
        pbar = 0
        for t in range(1,length):
            pt = t/(float(length))*100.
            for i in range(width):
                temp = -np.inf
                for j in range(tr_len):
                    if i+j-tr_dist>=0 and i+j-tr_dist<width:
                       for k in range(sep_len):
                            if i+k-sep_dist>=0+edge_range and i+k-sep_dist<width-edge_range:
                                for m in range(sep_len):
                                    if i+m-sep_dist>=0+edge_range and i+m-sep_dist<width-edge_range:
                                        o1 = int(obs1_ind[t][i+k-sep_dist])
                                        o2 = int(obs2_ind[t][i+m-sep_dist])
                                        value = tr[j][k][m] + val[t-1][i+j-tr_dist] + logarr[det_order[t,i],frac_ind[t,i],o1,o2]

                                        if value > temp:
                                            temp = value
                                            val[t][i] = temp
                                            prev[t][i] =  i+j-tr_dist
                                            det1[t][i] =  i+k-sep_dist
                                            det2[t][i] =  i+m-sep_dist
                                        else:
                                            continue
            if self.prog == True:
                if pt>pbar:
                    stdout.write('\r{} %'.format(round(pt)))
                    stdout.flush()
                    pbar+=100./length

        self.V = np.array(val)
        self.prev = np.array(prev)
        self.det1 = np.array(det1)
        self.det2 = np.array(det2)
        self.width = width
        self.length = length

    def get_track(self):
        """
        Takes in the viterbi data from a run and identifies the track thorugh whcih gives the maximum probability
        Returns
        -------------
        vit_track1 : array
           array of indicies for track in obs1
        vit_track2 : array
           array of indicies for track in obs2
        vit_track: array
           array of indicies for track in the "reference" detector
        max_end_prob: float
           value of maxmimum end statistic
        """
        cdef long previous = 1
        cdef long[:] vit_trackr = np.zeros(self.length).astype(int)
        cdef long[:] vit_track2 = np.zeros(self.length).astype(int)
        cdef long[:] vit_track1 = np.zeros(self.length).astype(int)
        
        cdef float max_prob = np.array([self.V[self.length-1][i] for i in range(self.width)]).max()
        cdef long max_prob_index = np.array([self.V[self.length-1][i] for i in range(self.width)]).argmax()
        
        vit_trackr[self.length-1] = max_prob_index
        vit_track1[self.length-1] = self.det1[self.length-1][max_prob_index]
        vit_track2[self.length-1] = self.det2[self.length-1][max_prob_index]
        previous = self.prev[self.length-1][max_prob_index]
        

        for t in range(self.length-2,-1,-1):
            vit_trackr[t] = previous # insert previous step
            vit_track1[t] = self.det1[t][previous]
            vit_track2[t] = self.det2[t][previous]
            previous = self.prev[t][previous]
    
        self.vit_track1 = np.array(vit_track1)
        self.vit_track2 = np.array(vit_track2)
        self.vit_track = np.array(vit_trackr)
        self.max_end_prob = max_prob
        
        #return np.array(vit_track1), np.array(vit_track2), np.array(vit_trackr), np.array(val),np.array(prev),max_prob

        
class three_detector(object):
 
    def __init__(self, tr, obs1, obs2, obs3, prog = False):
        '''
        initialising viterbi class
        '''
        self.prog = prog
        self.tracks = self.run(tr, obs1, obs2, obs3)

    

    #---------------------------------------------------------
    # Three detector viterbi
    #--------------------------------------------------------

    def run(self,tr1, obs11, obs21, obs31):
        """
        viterbi algorithm for multiple detectors 2 at the moment
        Args
        ---------
        tr1: array
            transition matrix
        obs11: array
            observation from detector 1
        obs21: array
            observation from detector 2
        returns
        -----------
        vit_track1: array
            path in det 1
        vit_track2: array
            path in 2
        vit_trackr: array
            path in reference detector
        val: array
            values in reference detector
        prev: array
            previous positions
        det1: array
            det1 data
        det2: array
            det 2 data
        """
        #tr_len = np.arange(len(tr))-len(tr)/2
        #range_tr = np.arange(3)
        
    
        cdef double[:, :, :, :] tr = tr1
        cdef double[:, :] obs1 = obs11
        cdef double[:, :] obs2 = obs21
        cdef double[:, :] obs3 = obs31

        shape = np.shape(obs1)
        
        cdef int length = min(len(obs1),len(obs2),len(obs3))
        cdef int width = len(obs1[0])
        
        cdef double[:, :] val = np.ones(shape)*-1e6
        cdef int[:, :] prev = np.zeros(shape,dtype = np.int32)
        cdef int[:, :] det1 = np.zeros(shape,dtype = np.int32)
        cdef int[:, :] det2 = np.zeros(shape,dtype = np.int32)
        cdef int[:, :] det3 = np.zeros(shape,dtype = np.int32)
        
        cdef int tr_len = len(tr)
        
        cdef int i
        cdef int j
        cdef int k
        cdef int m
        cdef int t
        cdef int x
        
        cdef long max_pos
        cdef long[:] indicies
        cdef double max_val
        cdef double value
        cdef double temp
        #cdef double[:] max_vals
    
        for i in range(width):
            
            temp = -1e6
            for k in range(tr_len): 
                if i+k-1>=0 and i+k-1<width:
                    for m in range(tr_len):
                        if i+m-1>=0 and i+m-1<width:
                            for l in range(tr_len):
                                if i+l-1>=0 and i+l-1<width:
                                    value = obs1[0][i+k-1] + obs2[0][i+m-1] + obs3[0][i+l-1] + tr[1][k][m][l]
                                if value > temp:
                                    temp = value
                                    val[0][i] = temp
                                    prev[0][i] =  i
                                    det1[0][i] =  i+k-1 
                                    det2[0][i] =  i+m-1
                                    det3[0][i] =  i+l-1
                                else:
                                    continue

    
        # i is current frequency position
        # j is transition in reference detector
        # k is separation from other detector
        
        pbar = 0
        for t in range(1,length):
            pt = t/(float(length))*100.
            for i in range(width):
                
                temp = -1e6
                for j in range(tr_len):
                    if i+j-1>=0 and i+j-1<width:
                       for k in range(tr_len):
                            if i+k-1>=0 and i+k-1<width:
                                for m in range(tr_len):
                                     if i+m-1>=0 and i+m-1<width:
                                         for l in range(tr_len):
                                             if i+l-1>=0 and i+l-1<width:
                                                 value = obs1[t][i+k-1] + obs2[t][i+m-1] + obs3[t][i+l-1] + tr[j][k][m][l] + val[t-1][i+j-1]
                                                 if value > temp:
                                                     temp = value
                                                     val[t][i] = temp
                                                     prev[t][i] =  i+j-1
                                                     det1[t][i] =  i+k-1 
                                                     det2[t][i] =  i+m-1
                                                     det3[0][i] =  i+l-1
                                                 else:
                                                     continue
            
            if self.prog == True:
                if pt>pbar:
                    stdout.write('\r{} %'.format(round(pt)))
                    stdout.flush()
                    pbar+=100./length

        
    
        cdef long previous = 1
        cdef long[:] vit_trackr = np.zeros(length).astype(int)
        cdef long[:] vit_track1 = np.zeros(length).astype(int)
        cdef long[:] vit_track2 = np.zeros(length).astype(int)
        cdef long[:] vit_track3 = np.zeros(length).astype(int)
        
        cdef float max_prob = np.array([val[length-1][i] for i in range(width)]).max()
        cdef long max_prob_index = np.array([val[length-1][i] for i in range(width)]).argmax()
        
        vit_trackr[length-1] = max_prob_index
        vit_track1[length-1] = det1[length-1][max_prob_index]
        vit_track2[length-1] = det2[length-1][max_prob_index]
        vit_track3[length-1] = det3[length-1][max_prob_index]
        previous = prev[length-1][max_prob_index]
        

        for t in range(length-2,-1,-1):
            vit_trackr[t] = previous # insert previous step
            vit_track1[t] = det1[t][previous]
            vit_track2[t] = det2[t][previous]
            vit_track3[t] = det3[t][previous]
            previous = prev[t][previous]
    
        self.vit_track1    = np.array(vit_track1)
        self.vit_track2    = np.array(vit_track2)
        self.vit_track3    = np.array(vit_track3)
        self.vit_track = np.array(vit_trackr)
        self.V            = np.array(val)
        self.prev         = np.array(prev)
        self.max_end_prob = max_prob
    
        #return np.array(vit_track1), np.array(vit_track2), np.array(vit_track3), np.array(vit_trackr), np.array(val),np.array(prev), max_prob

class single_detector_mem_n(object):
 
    def __init__(self, tr, obs, prog = False):
        '''
        
        Find single detector algorithm with a memory, uses the summed pixel power as statistic

        initialising viterbi class
        '''
        self.prog = prog
        self.tracks = self.run(tr, obs)

    
    #--------------------------------------------------------------------------
    # n dimensional memory viterbi
    #------------------------------------------------------------------------

    def run(self, tr, obs):
        ''' Run viterbi algroithm with memory of n
        the amount of memory is determined by the transition matrix
        args
        -------
        tr : array
            transition matrix, any dimension greater that 1 - type: array
        obs: array
            observation - type: array
        
        returns
        -------
        vit_track: array
            optimum path through data
        val: array
            values of viterbi
        prev: array
            previous positions of viterbi
        '''
        
        if len(np.shape(tr))<=1:
            raise Exception("ERROR: Please use a transition matrix with dimensions larger than one")
        
        #np.insert(obs,0,-1e6,axis=1)
        #np.insert(obs,len(obs[0]),-1e6,axis=1)

        #cdef double[:] tr = tr1
        #cdef double[:, :] obs = obs1
        
        range_tr = np.arange(len(tr))
        cdef int n = len(tr)
        shape = np.shape(tr)
        cdef int length = len(shape)
        
        dimensions = (len(obs),len(obs[0]))
        dimensions = list(dimensions)
        
        cdef int length_t = dimensions[0]
        cdef int width = dimensions[1]
        
        cdef int i
        cdef int t
        cdef int k
        cdef int c
        
        cdef double value
        cdef double temp
        
        for i in range(len(shape)-1):
            dimensions.append(n)
        
        dimensions = tuple(dimensions)

        val    = np.ones(dimensions)*-1e6
        prev   = np.ones(dimensions)
    
        range_j = list(itertools.product(range(n),repeat=length-1))
    
        for i in range(width):
            for j in range_j:
                val[0][i][j]    = obs[0][i]
                prev[0][i][j]   = np.nan
    
        cdef float pbar = 0
        for t in range(1,length_t):
            obst = obs[t]
            valt = val[t-1]
            pt = t/float(length_t)*100.
            for i in range(width): 
                for j in range_j:
                    cond = True
                    for c in range(1,len(j)+1):
                        b = i+np.sum(j[:c])-len(j[:c])
                        if b >= 0 and b<width: 
                            continue
                        else:
                            cond = False
                            break
                    if cond:
                        temp = -1e6
                        for k in range(n):
                            m = list(j[1:])
                            m.append(k)
                            if t<length:
                                for n1 in range(len(m)):
                                    if n1>=t-1:
                                        m[n1] = 1
                            m = tuple(m)
                            value = obst[i] + tr[j[0]][m] + valt[i+j[0]-1][m]
    
                        
                            if value>temp:
                                temp = value
                                val[t][i][j] = temp
                                prev[t][i][j] = k

            if self.prog == True:
                if pt>pbar:
                    stdout.write('\r{} %'.format(round(pt)))
                    stdout.flush()
                    pbar+=100./length_t

        cdef int previous   = 1
        previous_1 = None
        vit_track   = [0]

        #cdef double max_end_prob = max(val[length-1][i] for i in range(width))

        temp = -1e6
        for i in range(width):
            for j in range_j:
                value = val[-1][i][j]
                if value>temp:
                    temp = value
                    vit_track[0] = i
                    previous = i
                    previous_1 = j
                    previous_2 = int(prev[-1][i][j])
    
                
        for t in range(length_t-2,-1,-1):
            vit_track.append(previous+previous_1[0]-1)
            previous += previous_1[0]-1
            previous_1 = list(previous_1[1:])
            previous_1.append(int(previous_2))
            previous_1 = tuple(previous_1)
            previous_2 = prev[t][previous][previous_1]
        
        vit_track = vit_track[::-1]

        self.vit_track = np.array(vit_track)
        self.V = np.array(val)
        self.prev = np.array(prev)
        self.max_end_prob = None # needs to be fixed
        
        #return (vit_track,val,prev)

class two_detector_mem_n(object):
 
    def __init__(self, tr, obs1, obs2, prog = False):
        '''
        initialising viterbi class
        '''
        self.prog = prog
        self.tracks = self.run(tr, obs1, obs2)

    
    #------------------------------------------------------------------------
    # Multi viterbi vit n dimensional memory
    #-----------------------------------------------------------------------
    

    def run(self, tr, obs11, obs22):
        ''' Run viterbi algroithm with memory of n
        the amount of memory is determined by the transition matrix
        args
        -------
        tr : transition matrix, any dimension greater that 1 - type: array
        obs: observation - type: array
        
        returns
        -------
        vit_track: optimum path through data
        '''
        
        if len(np.shape(tr))<=1:
            raise Exception("ERROR: Please use a transition matrix with dimensions larger than one")
        
        range_tr = np.arange(len(tr))
        
        cdef double[:, :] obs1 = obs11
        cdef double[:, :] obs2 = obs22
        
        cdef int n      = len(tr)
        shape           = np.shape(tr)
        cdef int length = len(shape)
        
        dimensions = (min(len(obs1),len(obs2)),min(len(obs1[0]),len(obs2[0])))
        dimensions = list(dimensions)
        
        cdef int length_t = dimensions[0]
        cdef int width = dimensions[1]
        
        cdef int i
        cdef int k
        cdef int m
        cdef int j1
        cdef int n1
        cdef int b
        cdef int t
        cdef double value
        cdef double temp

        
        for i in range(len(shape)-3):
            dimensions.append(n)
        
        dimensions = tuple(dimensions)

        val    = np.ones(dimensions)*-1e6
        prev   = np.ones(dimensions)
        pos1   = np.ones(dimensions)
        pos2   = np.ones(dimensions)
        
    
        range_j = list(itertools.product(range(n),repeat=length-3))
    
        for i in range(width):
            for j in range_j:
                temp = -1e6
                for k in range(n):
                    if i+k-1>=0 and i+k-1<width:
                        for m in range(n):
                            if i+m-1>=0 and i+m-1<width:
                                value = obs1[0][i+k-1] + obs2[0][i+m-1] + tr[tuple(np.ones(len(shape)-2).astype('int'))][k][m]
                                if value>temp:
                                    temp = value
                                    val[0][i][j]    = value
                                    prev[0][i][j]   = np.nan
                                    pos1[0][i][j]   = k
                                    pos2[0][i][j]   = m
    
        pbar = 0
        for t in range(1,length_t):
            obs1t = obs1[t]
            obs2t = obs2[t]
            valt = val[t-1]
            pt = t/float(length_t)*100.
            for i in range(width): 
                for j in range_j:
                    cond = True
                    for c in range(1,len(j)+1):
                        b = i+np.sum(j[:c])-len(j[:c])
                        if b >= 0 and b<width: 
                            continue
                        else:
                            cond = False
                            break
                    if cond:
                        temp = -1e6
                        for j1 in range(n):
                            a = list(j[1:])
                            a.append(j1)
                            if t<length:
                                for n1 in range(len(a)):
                                    if n1>=t-1:
                                        a[n1] = 1
                            a = tuple(a)
                            for k in range(n):
                                if i+k-1>=0 and i+k-1<width:
                                    for m in range(n):
                                        if i+m-1>=0 and i+m-1<width:
                                            value = obs1t[i+k-1] + obs2t[i+m-1] + tr[j][j1][k][m] + valt[i+j[0]-1][a]
                                            if value > temp:
                                                temp = value
                                                val[t][i][j]  = temp
                                                prev[t][i][j] = j1
                                                pos1[t][i][j] = k
                                                pos2[t][i][j] = m
                        """
                        for k in range(length):
                            m = list(j[1:])
                            m.append(k)x
                            if t<length:
                                for n1 in range(len(m)):
                                    if n1>=t-1:
                                        m[n1] = 1
                            m = tuple(m)
                            value = obst[i] + tr[j[0]][m] + valt[i+j[0]-1][m]
                            if value>temp:
                                temp = value
                                val[t][i][j] = temp
                                prev[t][i][j] = k
                        """
                    
            if self.prog == True:
                if pt>pbar:
                    stdout.write('\r{} %'.format(round(pt)))
                    stdout.flush()
                    pbar+=100./length_t

        previous   = 1
        previous_1 = None
        vit_track1  = [0]
        vit_track2  = [0]
        vit_trackr  = [0]

        temp = -1e6
        for i in range(width):
            for j in range_j:
                value = val[-1][i][j]
                if value>temp:
                    temp = value
                    vit_trackr[0] = i
                    vit_track1[0] = i+pos1[-1][i][j]-1
                    vit_track2[0] = i+pos2[-1][i][j]-1
                    previous = i
                    previous_1 = j
                    previous_2 = int(prev[-1][i][j])
    
        
        for t in range(length_t-2,-1,-1):
            vit_trackr.append(previous+previous_1[0]-1)
            previous += previous_1[0]-1
            previous_1 = list(previous_1[1:])
            previous_1.append(int(previous_2))
            previous_1 = tuple(previous_1)
            vit_track1.append(previous+pos1[t][previous][previous_1]-1)
            vit_track2.append(previous+pos2[t][previous][previous_1]-1)
            previous_2 = prev[t][previous][previous_1]
        
        vit_trackr = vit_trackr[::-1]   
        vit_track1 = vit_track1[::-1] 
        vit_track2 = vit_track2[::-1]

        self.vit_track1 = np.array(vit_track1)
        self.vit_track2 = np.array(vit_track2)
        self.vit_track = np.array(vit_trackr)
        self.V = np.array(val)
        self.prev = np.array(prev)
        
        #return (vit_track1,vit_track2,vit_trackr,val,prev)


### -----------------------------------------------------------------

## Single detector algorithm where data can include large gaps 

### ----------------------------------------------------------------

class single_detector_gaps(object):

    def __init__(self, tr, obs,times,tsft):
        """
        single detector viterbi algorithm where the data can include large gaps in data
        NOTE: This is for large gaps in data, the single_detector algorithm works faster if gaps are filled with the mean of data

        Parameters
        -----------------
        tr: list
            transition matrix elements
        obs: array
            an array of the sfts
        times: list
            times corresponding to each of the sfts
        tsft: float
            the length of each sfts in s
        Returns
        -------------------
        self.vit_track: list
            optimum path through dataset
        self.vit_track_times: list
            list of the times corresponding to each of the path elements
        """
        self.vits, self.sts, self.gps = self.vit_gaps(obs, times, tr=tr, tsft=tsft)
        self.vit_track, self.vit_track_times = self.get_track(self.vits, self.sts, self.gps,tsft=tsft)
        self.max_end_prob = max(self.vits[-1].V[-1])
        
    def max_track(self,tr,length,diff,bnd,ind,band_width):
        tr_val = np.arange(len(tr)).astype(int) - np.floor(len(tr)/2).astype(int)
        result = [seq for seq in itertools.combinations_with_replacement(tr_val, length) if sum(seq) == diff]
        max_seq = None
        thresh = -np.inf
        thresh2 = -np.inf
        #print ind,bnd, diff, [np.cumsum(s)+ind for s in result]
        for seq in result:
            #print seq
            abspath = np.array(np.cumsum(seq) + ind)
            if np.any(abspath < 0) == True or np.any(abspath >= band_width) == True:
                continue
            else:
                s = sum([tr[elem+1] for elem in seq]) 
                l = len(np.where(np.array(seq) == 0)[0])
                if s > thresh:
                    max_seq = np.insert(abspath,0,ind)[:-1] 
                    thresh = s
                elif s == thresh and l > thresh2:
                    max_seq = np.insert(abspath,0,ind)[:-1] 
                    thresh = s
                    thresh2 = l

        return max_seq,s

    def gap_run(self,viterbi, band_width, diff, tsft, tr):
        length = diff - 1
        max_vals = np.zeros(band_width)
        max_paths = np.zeros((band_width,length))
        #print band_width, length
        for bnd in range(band_width):
            low = bnd - length
            high = bnd + length + 1
            if low < 0:
                low = 0
            if high >= band_width:
                high = band_width
            thres = 0
            vend = viterbi.V[-1,low:high]
        
            val = 0
            ind = 0
            for j in range(len(vend)):
                if vend[j] > thres:
                    val = vend[j]
                    ind = low + j
                    thres = vend[j] 
            diff_ind = bnd-ind
            gaptrack = self.max_track(tr,length,diff_ind,bnd,ind,band_width)
            #print bnd,ind,low,high,diff_ind,length,gaptrack
        
            max_paths[bnd] = np.array(gaptrack[0])
            max_vals[bnd] = gaptrack[1] + val + length*2
    
        return max_vals, max_paths

    def find_path(self,prev,end_ind):
        length = len(prev)
        previous = prev[-1][end_ind]
        vit_track =  np.zeros(length,dtype=np.int32)
        vit_track[-1] = end_ind
        for t in range(length-2,-1,-1):
            vit_track[t] = previous # insert previous step
            previous = prev[t][previous]
        
        return vit_track
        

    def vit_gaps(self,data1,epochs1,tr,tsft):

        band_width = len(data1[0])
        vit = []
        starts = []
        gappaths = []
        gapend = 0
        vitstart = epochs1[0]
        max_vals = None
    

        data = []
        epochs = []
        for i,idx in enumerate(epochs1):
            if i == 0 or i == len(epochs1)-1:
                data.append(data1[i])
                epochs.append(idx)
            elif i > 0 and i != len(epochs1)-1:
                diff = epochs1[i]-epochs1[i-1]
                if diff == 2:
                    data.append(np.ones(band_width)*2)
                    epochs.append(idx-tsft)
                    data.append(data1[i])
                    epochs.append(idx)
                else:
                    data.append(data1[i])
                    epochs.append(idx)
        data = np.array(data)
        epochs = np.array(epochs)
    
        for i,idx in enumerate(epochs):
            if i == 0 :
                pass
            elif i > 0:# and i != len(epochs)-1:
                diff = epochs[i]-epochs[i-1]
                diff_ind = int(diff/tsft)
                if diff_ind >= 2 and diff_ind < band_width + 1:
                    if max_vals is not None:
                        viterbi = single_detector(tr, np.vstack([max_vals,data[gapend:i]]))
                        starts.append(np.arange(len(viterbi.vit_track))*tsft + vitstart - tsft)
                        vitstart = idx 
                    else:
                        viterbi = single_detector(tr, data[gapend:i])
                        starts.append(np.arange(len(viterbi.vit_track))*tsft + vitstart)
                        vitstart = idx 
        
                    vit.append(viterbi)
                    gapend = i
                    max_vals, max_paths = self.gap_run(viterbi, band_width, diff_ind, tsft, tr)
                    #print max_vals, max_paths
                    gappaths.append(max_paths)
                    if i == len(epochs)-1:
                        viterbi = single_detector(tr, np.vstack([max_vals,data[gapend:]]))
                        starts.append(np.arange(len(viterbi.vit_track))*tsft + idx - tsft)
                        vit.append(viterbi)
            
                elif diff_ind >= band_width + 1:
                
                    diff1 = band_width + 1
                    if max_vals is not None:
                        viterbi = single_detector(tr, np.vstack([max_vals,data[gapend:i]]))
                        starts.append(np.arange(len(viterbi.vit_track))*tsft + vitstart - tsft)
                        vitstart = idx 
                    else:
                        viterbi = single_detector(tr, data[gapend:i])
                        starts.append(np.arange(len(viterbi.vit_track))*tsft + vitstart)
                        vitstart = idx
        

                    vit.append(viterbi)
                    gapend = i
                    max_vals, max_paths = self.gap_run(viterbi, band_width , diff1, tsft, tr)
                    #print max_vals, max_paths
                    gappaths.append(max_paths) 
                    if i == len(epochs)-1:
                        viterbi = single_detector(tr, np.vstack([max_vals,data[gapend:]]))
                        starts.append(np.arange(len(viterbi.vit_track))*tsft + idx - tsft)
                        vit.append(viterbi)
                  
                else:
                    continue
                
        return vit,starts, gappaths

    def get_track(self,vit,starts,gappaths,tsft=1):
        vit_track = []
        vit_track_time = []
        gpb = False
        gp = None
        if len(starts) > 1:
            for v in range(len(starts))[::-1]:
                if gpb:
                    p1 = self.find_path(vit[v].prev,int(vit_track[-1]))[::-1]
                    vit_track.extend(p1[1:])
                    vit_track_time.extend(starts[v][::-1][1:])
                else:
                    p1 = vit[v].vit_track[::-1]
                    vit_track.extend(p1)
                    vit_track_time.extend(starts[v][::-1])
                st = vit_track[-1]
                if v-1 >= 0:
                    gp = gappaths[v-1][st][::-1]
                    vit_track.extend(gp)
                    vit_track_time.extend(np.array(np.arange(len(gp))*tsft + starts[v-1][-1])[::-1])#+np.array(range(len(gp))[::-1]))
                    #vit_track_time.extend(range(len(gp))[::-1])
                    gpb = True
                else:
                    gpb = False
        else:
            vit_track = vit[0].vit_track
        return vit_track[::-1], vit_track_time[::-1]




'''

class N_detector(object):
 
    def __init__(self, tr, obs):
        """
        UNTESTED: DO NOT USE
        initialising viterbi class
        """
        self.prog = prog
        self.tracks = self.run(tr, obs)

    
    #------------------------------------------------------------------------
    # Viterbi with N detectors
    #-----------------------------------------------------------------------
    

    def run(self, tr, mobs):
        """

        Run viterbi algroithm with memory of n
        the amount of memory is determined by the transition matrix
        args
        -------
        tr : transition matrix, any dimension greater that 1 - type: array
        obs: observation - type: array
        
        returns
        -------
        vit_track: optimum path through data
        """
        
        if len(np.shape(tr))<=1:
            raise Exception("ERROR: Please use a transition matrix with dimensions larger than one")
        
        range_tr = np.arange(len(tr))
        
        n_detectors = len(mobs)

        cdef double[:, :, :] obs_list = mobs
        
        cdef int n      = len(tr)
        shape           = np.shape(tr)
        cdef int length = len(shape)
        
        min_len1 = np.inf
        min_len2 = np.inf
        for obs in obs_list:
            if len(obs) < min_len1:
                min_len1 = len(obs)
            if len(obs[0]) < min_len2:
                min_len2 = len(obs[0])
            
        dimensions = (min_len1, min_len2)
        dimensions = list(dimensions)
        
        cdef int length_t = dimensions[0]
        cdef int width = dimensions[1]
        
        cdef int i
        cdef int k
        cdef int m
        cdef int j1
        cdef int n1
        cdef int b
        cdef int t
        cdef double value
        cdef double temp

        
        for i in range(len(shape)-3):
            dimensions.append(n)
        
        dimensions = tuple(dimensions)

        val    = np.ones(dimensions)*-1e6
        prev   = np.ones(dimensions)
        pos1   = np.ones(dimensions)
        pos2   = np.ones(dimensions)
        
    
        range_j = list(itertools.product(range(n),repeat=n_detectors))
    
        for i in range(width):
            temp = -1e6
            for k in range(n):
                if i+k-1>=0 and i+k-1<width:
                    for m in range(n):
                        if i+m-1>=0 and i+m-1<width:
                            value = obs1[0][i+k-1] + obs2[0][i+m-1] + tr[tuple(np.ones(len(shape)-2).astype('int'))][k][m]
                            if value>temp:
                                temp = value
                                val[0][i]    = value
                                prev[0][i]   = np.nan
                                pos1[0][i]   = k
                                pos2[0][i]   = m
    
        pbar = 0
        for t in range(1,length_t):
            obs1t = obs1[t]
            obs2t = obs2[t]
            valt = val[t-1]
            pt = t/float(length_t)*100.
            for i in range(width): 
                for j in range_j:
                    cond = True
                    for c in range(1,len(j)+1):
                        b = i+np.sum(j[:c])-len(j[:c])
                        if b >= 0 and b<width: 
                            continue
                        else:
                            cond = False
                            break
                    if cond:
                        temp = -1e6
                        for j1 in range(n):
                            a = list(j[1:])
                            a.append(j1)
                            if t<length:
                                for n1 in range(len(a)):
                                    if n1>=t-1:
                                        a[n1] = 1
                            a = tuple(a)
                            for k in range(n):
                                if i+k-1>=0 and i+k-1<width:
                                    for m in range(n):
                                        if i+m-1>=0 and i+m-1<width:
                                            value = obs1t[i+k-1] + obs2t[i+m-1] + tr[j][j1][k][m] + valt[i+j[0]-1][a]
                                            if value > temp:
                                                temp = value
                                                val[t][i][j]  = temp
                                                prev[t][i][j] = j1
                                                pos1[t][i][j] = k
                                                pos2[t][i][j] = m
                        """
                        for k in range(length):
                            m = list(j[1:])
                            m.append(k)x
                            if t<length:
                                for n1 in range(len(m)):
                                    if n1>=t-1:
                                        m[n1] = 1
                            m = tuple(m)
                            value = obst[i] + tr[j[0]][m] + valt[i+j[0]-1][m]
                            if value>temp:
                                temp = value
                                val[t][i][j] = temp
                                prev[t][i][j] = k
                        """
                    
            if self.prog == True:
                if pt>pbar:
                    stdout.write('\r{} %'.format(round(pt)))
                    stdout.flush()
                    pbar+=100./length_t

        previous   = 1
        previous_1 = None
        vit_track1  = [0]
        vit_track2  = [0]
        vit_trackr  = [0]

        temp = -1e6
        for i in range(width):
            for j in range_j:
                value = val[-1][i][j]
                if value>temp:
                    temp = value
                    vit_trackr[0] = i
                    vit_track1[0] = i+pos1[-1][i][j]-1
                    vit_track2[0] = i+pos2[-1][i][j]-1
                    previous = i
                    previous_1 = j
                    previous_2 = int(prev[-1][i][j])
    
        
        for t in range(length_t-2,-1,-1):
            vit_trackr.append(previous+previous_1[0]-1)
            previous += previous_1[0]-1
            previous_1 = list(previous_1[1:])
            previous_1.append(int(previous_2))
            previous_1 = tuple(previous_1)
            vit_track1.append(previous+pos1[t][previous][previous_1]-1)
            vit_track2.append(previous+pos2[t][previous][previous_1]-1)
            previous_2 = prev[t][previous][previous_1]
        
        vit_trackr = vit_trackr[::-1]   
        vit_track1 = vit_track1[::-1] 
        vit_track2 = vit_track2[::-1]

        self.vit_track1 = np.array(vit_track1)
        self.vit_track2 = np.array(vit_track2)
        self.vit_track = np.array(vit_trackr)
        self.V = np.array(val)
        self.prev = np.array(prev)
        
        #return (vit_track1,vit_track2,vit_trackr,val,prev)

'''

    
