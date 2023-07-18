from __future__ import print_function
import numpy as np
import matplotlib
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except:
    print("cannot import matplotlib")
import timeit
import scipy.stats as st
import os
import copy
import pickle
from scipy.signal import butter, filtfilt
from scipy.special import logsumexp


def day_track_sum(track):
    '''
    finds the track of a pulsar in the day average case
    args
    ------
    track: array
        path in 1800s sfts
    returns: array
        averages path over one day
    '''
    data_av1 = []
    for i in np.linspace(0,len(track)-49,len(track)-48)[::48]:
        av = np.mean(track[int(i):int(i+48)])
        data_av1.append(av)
    return np.array(data_av1)

def transition_matrix(par,log=True):
    '''
    generate a symmetric transition matrix for 0 memory and 1 detector
    args
    -------
    par: float
        ratio of two outer elements to the centre element of the transition matrix
    kwargs
    ------
    log: bool
        return the log of the transition matrix if True, just probabilities if false
    returns 
    -------
    tr: array
        3x1 array of transition probabilities 
    '''
    v1 = 1.
    v2 = v1/par
    tr = [v2,v1,v2]
    if log:
        return np.log(tr/np.sum(tr))
    if not log:
        return tr/np.sum(tr)

def tr_p_long(par,log=True,n=2):
    '''
    generate a symmetric transition matrix for 0 memory and 1 detector
    args
    -------
    par: float
        ratio of two outer elements to the centre element of the transition matrix
    kwargs
    ------
    log: bool
        return the log of the transition matrix if True, just probabilities if false
    returns 
    -------
    tr: array
        3x1 array of transition probabilities 
    '''
    v1 = 1.
    v2 = v1/par
    if n == 2:
        tr = [v2,v2,v1,v2,v2]
    if n == 3:
        tr = [v2,v2,v2,v1,v2,v2,v2]
    if log:
        return np.log(tr/np.sum(tr))
    if not log:
        return tr/np.sum(tr)

def transition_matrix_2d(par1,par2,par3,log=True):
    '''
    generate a symmetric transition matrix for 0 memory and 1 detector
    args
    -------
    par1: float
        ratio of left and rightjump elements to the centre jump element of the transition matrix
    par2: float
        ratio detector 1 being in left or right bin compared to reference
    par3: float
        ratio detector 2 being in left or right bin compared to reference
    kwargs
    ------
    log: bool
        return the log of the transition matrix if True, just probabilities if false
    returns 
    -------
    tr: array
        3x3 array of transition probabilities 
    '''
    j1 = 1.
    jlr = j1/par1
    plr1 = j1/par2
    plr2 = j1/par3
    tr_j= np.array([jlr,j1,jlr])
    tr_p1 = np.array([plr1,j1,plr1])
    tr_p2 = np.array([plr2,j1,plr2])
    tr_p = np.array([tr_p1[0]*tr_p2,tr_p1[1]*tr_p2,tr_p1[2]*tr_p2])
    tr = [tr_p*tr_j[0],tr_p*tr_j[1],tr_p*tr_j[2]]
    if log:
        return np.log(tr/np.sum(tr))
    if not log:
        return tr/np.sum(tr)


def transition_matrix_2d_5(leftright1,leftright2,sep2,sep3,log=True):
    '''
    generate a symmetric transition matrix for 0 memory and 1 detector, allowing 2 jumps either side
    args
    -------
    par1: float
        ratio of left and rightjump elements to the centre jump element of the transition matrix
    par2: float
        ratio detector 1 being in left or right bin compared to reference
    par3: float
        ratio detector 2 being in left or right bin compared to reference
    kwargs
    ------
    log: bool
        return the log of the transition matrix if True, just probabilities if false
    returns 
    -------
    tr: array
        3x3 array of transition probabilities 
    '''
    j1 = 1.
    jlr1 = j1/leftright1
    jlr2 = j1/leftright2
    plr1 = j1/sep2
    plr2 = j1/sep3
    tr_j= np.array([jlr2,jlr1,j1,jlr1,jlr2])
    tr_p1 = np.array([plr1,plr1,j1,plr1,plr1])
    tr_p2 = np.array([plr2,plr2,j1,plr2,plr2])
    tr_p = np.array([tr_p1[0]*tr_p2,tr_p1[1]*tr_p2,tr_p1[2]*tr_p2,tr_p1[3]*tr_p2,tr_p1[4]*tr_p2])
    tr = [tr_p*tr_j[0],tr_p*tr_j[1],tr_p*tr_j[2],tr_p*tr_j[3],tr_p*tr_j[4]]
    if log:
        return np.log(tr/np.sum(tr))
    if not log:
        return tr/np.sum(tr)


def track_power(track,data):
    '''
    returns an array the power along any given track 
    args
    -------
    track: array
        the track found by viterbi
    data: 2d array
        data to search through
    returns
    -------
    val: array
        power along given track
    '''
    val = []
    for i,j in enumerate(track):
        val.append(data[i,j])
    return val

def stat_on_path(track,data):
    '''
    returns the viterbi statistic along a given path (currently assumes flat transition matrix) 
    args
    -------
    track: array
        the track found by viterbi
    data: 2d array
        data to search through
    returns
    -------
    val: array
        stat along track
    '''
    # get summed stat along trck
    val = np.array([data[i,j] for i,j in enumerate(track)])
    
    # calculate difference between this and element before
    norm_val = np.insert(np.diff(val), 0, val[0])
    del val
    return norm_val


def ft_power(track):
    f0 = 2./86164.09
    fs = 1./1800
    w = np.exp(-2*np.pi*1j*f0*np.arange(len(track))/fs)
    power = np.abs(np.sum(track*w))**2

def fft_stat(data,paths):
    """ Statistic taking the fft of the track at 1 siderial day"""
    path = [i for i in paths for j in range(48)]
    power = []
    for i,j in enumerate(path[:len(data)]):
        power.append(data[i][j])
    f0 = 1./86164.09
    f1 = 1./85000
    f2 = 1./89000
    signal = np.array(power)
    fs = 1./1800
    w = np.exp(-2*np.pi*1j*f0*np.arange(len(signal))/fs)
    w_back1 = np.exp(-2*np.pi*1j*f1*np.arange(len(signal))/fs)
    w_back2 = np.exp(-2*np.pi*1j*f2*np.arange(len(signal))/fs)
    power_stat = abs(sum(signal*w))**2
    power_back1 = abs(sum(signal*w_back1))**2
    power_back2 = abs(sum(signal*w_back2))**2
    power_back = (power_back1+power_back2)/2.0
    return power_stat/power_back

def highpass_filter(data, cut, fs, order=5):
    """
    simple highpass filter the data with given cutoff and samplin frequency
    """
    nyq = 0.5*fs
    cut1 = cut / nyq
    b, a = butter(order, cut1, btype='highpass')
    y = filtfilt(b, a, data)
    return y

def res(track,path):
    """
    find the summed residual between the viterbi track and the pulsars path
    """
    separation = 0
    for i in range(len(track)):
        separation += np.abs(track[i]-path[i])
    return separation

def frac_off(track,path,noise = None,gap_val = 2):
    """
    find the fraction of track elements which are not in the same frequency bin as the injected signal
    """
    frac = 0
    track_len = 0
    if noise is None:
        pass
    else:
        for i in range(len(noise)):
            if np.isnan(noise[i]) == False or noise[i] == gap_val:
                track_len += 1
            else:
                pass
        
    for i in range(len(track)):
        if noise is not None:
            if track[i] == path[i]:
                if np.isnan(noise[i]) == False or noise[i] == gap_val:
                    frac += 1
                else:
                    pass
            else:
                pass
        else:
            if track[i] == path[i]:
                frac += 1
            else:
                pass
    if noise is None:
        return frac/float(len(track))
    else:
        return frac/float(track_len)
    

def find_rmeds(pul_track,vit_track):
    """
    find the rms (root median squared)of the difference of two tracks
    Parameters
    ------------
    pul_tracks: array
         array of signal tracks, can be single array or list of arrays
    vit_tracks: array
         array of tracks found by search, can be single array of list of arrays
    Returns
    ---------------
    diff: float
        value of rms for list of input tracks
    """
    diff = []
    for elem in range(len(pul_track)):
        pathsqs = ((np.array(pul_track[elem])-np.array(vit_track[elem]))**2)
        diff.append(np.sum(np.median(np.array(pathsqs))))
        #diff.append(1./len(pathsqs)*np.sum(np.array(pathsqs)))

    return np.sqrt(1./(len(diff))*np.sum(diff))

def find_rms(pul_track,vit_track):
    """
    find the rms (root median squared)of the difference of two tracks
    Parameters
    ------------
    pul_tracks: array
         array of signal tracks, can be single array or list of arrays
    vit_tracks: array
         array of tracks found by search, can be single array of list of arrays
    Returns
    ---------------
    diff: float
        value of rms for list of input tracks
    """
    diff = []
    for elem in range(len(pul_track)):
        pathsqs = ((np.array(pul_track[elem])-np.array(vit_track[elem]))**2)
        #diff.append(np.sum(np.median(np.array(pathsqs))))
        diff.append(1./len(pathsqs)*np.sum(np.array(pathsqs)))

    return np.sqrt(1./(len(diff))*np.sum(diff))


