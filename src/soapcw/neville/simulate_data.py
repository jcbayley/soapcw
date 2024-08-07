import lalpulsar
import numpy as np
import lal
from astropy.coordinates import SkyCoord
from astropy import units as u
import pickle as pkl
import os
import sys
import time
import random
import natsort
import copy 
from soapcw import cw

class CW:
    
    def __init__(self, earth_ephem=None, sun_ephem=None):
        """Simulate a CW signal
        """
        self.yeartime = 3.15e7
        self.c = 3e8
        self.r0 = 1.5e11
        self.omega0 = 2*np.pi/self.yeartime
        self.orb_fact = self.r0*self.omega0/self.c

        if earth_ephem is None:
            self.earth_ephem = "earth00-40-DE430.dat.gz"
        else:
            self.earth_ephem = earth_ephem
        if sun_ephem is None:
            self.sun_ephem = "sun00-40-DE430.dat.gz"
        else:
            self.sun_ephem = sun_ephem

        self.get_edat()


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


    def transform_sph_cart2d(self,lon,lat):
        """ Transform lon,lat to x,y coords"""
        # absolute to take 
        x = np.cos(lat)*np.cos(lon)
        y = np.cos(lat)*np.sin(lon)

        return x,y

    def transform_cart2d_sph(self,x,y):
        """ transform x,y to lonlat coords"""
        rs = np.sqrt(x**2 + y**2)

        # if r is outside unit circle set sample to nan
        if type(rs) in [float, int, np.double] and rs > 1:
            x = np.nan#x/rs
            y = np.nan#y/rs
            rs = np.nan#1
        if type(rs) in [np.array, list, np.ndarray]:
            x[rs > 1] = np.nan#x[rs > 1]/rs[rs > 1]
            y[rs > 1] = np.nan#y[rs > 1]/rs[rs > 1]
            rs[rs > 1] = np.nan#1

        lat = np.arccos(rs)
        lon = np.arctan2(y,x)
        lon[lon < 0] += 2*np.pi
        return np.array([lon, lat])
    
    def transform_fsph_offcart2d(self, lon, lat, f, fmins):
        """ Transform lon,lat to x,y coords"""
        # absolute to take 
        x = np.cos(lat)*np.cos(lon)
        y = np.cos(lat)*np.sin(lon)
        
        off = (f - fmins)/10

        return x,y,off

    def transform_offcart2d_fsph(self, x, y, off, fmins):
        """ transform x,y to lonlat coords"""
        rs = np.sqrt(x**2 + y**2)

        # if r is outside unit circle set sample to nan
        if type(rs) in [float, int, np.double] and rs > 1:
            x = np.nan#x/rs
            y = np.nan#y/rs
            rs = np.nan#1
        if type(rs) in [np.array, list, np.ndarray]:
            x[rs > 1] = np.nan#x[rs > 1]/rs[rs > 1]
            y[rs > 1] = np.nan#y[rs > 1]/rs[rs > 1]
            rs[rs > 1] = np.nan#1

        lat = np.arccos(rs)
        lon = np.arctan2(y,x)
        lon[lon < 0] += 2*np.pi
        
        f = 10*off + fmins
        
        return np.array([lon, lat, f])


    def transform_geo_ecliptic(self, alpha, delta):
        """ transform geocentric cords to ecliptic coords using astropy"""
        skyp_true = SkyCoord(ra = alpha,dec=delta,unit="rad", frame="icrs")
        skyp_true_barycenter = skyp_true.barycentrictrueecliptic
        lon_bry, lat_bry = skyp_true_barycenter.data.lon.rad,skyp_true_barycenter.data.lat.rad
        return lon_bry, lat_bry

    def transform_ecliptic_geo(self, lon, lat):
        """ transform ecliptic coords to geocentric coords with astropy"""
        skyp = SkyCoord(lon = lon,lat=lat,unit="rad",frame="barycentrictrueecliptic")
        skyp_icrs = skyp.icrs
        alpha, delta = skyp_icrs.data.lon.rad,skyp_icrs.data.lat.rad
        return alpha, delta

    
    def transform_sph_cart(self, r, lon_bry, lat_bry):
        """ transform radius lon lat to 3d cartesian coords """
        # absolute to take northern ecliptic hemisphere
        x = r*np.cos(lat_bry)*np.cos(lon_bry)
        y = r*np.cos(lat_bry)*np.sin(lon_bry)
        z = r*np.sin(lat_bry)

        return x,y,z

    def transform_ecliptic_amp(self, fmin, width, f0, fdot, lon, lat):
        """
        transform the frequency lngitude and latitude into amplitude offset and longitude
        """
        A = f0*self.r0*self.omega0*np.sin(lat)/self.c * 1./self.max_A
        s = (f0 - fmin)/width
        norm_fdot = fdot/(1e-9)
        return s, A, lon - np.pi, norm_fdot

    def transform_amp_ecliptic(self, fmin, width, s, A, lon, norm_fdot):
        """
        transform amplitude longitude and offset into longitude latitude and frequnecy 
        """
        f0 = fmin + s*width
        
        coslat = self.max_A*self.c*A/(f0*self.r0*self.omega0)
        lat = np.abs(np.arcsin(coslat))
        """
        lat = A*500/f0
        """
        fdot = norm_fdot*1e-9
        
        lon = lon + np.pi
        if type(lon) in [float, int, np.double]:
            lon = np.remainder(lon, 2*np.pi)

        elif type(lon) in [np.array, list, np.ndarray]:
            lon = np.remainder(lon, 2*np.pi)
                
        if type(lat) in [float, int, np.double]:
            if lat > np.pi/2:
                lat = np.nan
            if lat < 0:
                lat = np.nan
        elif type(lat) in [np.array, list, np.ndarray]:
            lat[lat > np.pi/2] = np.nan
            lat[lat < 0] = np.nan
                
        
        return lon, lat, f0, fdot

    def transform_lonlat_normlonlat(self, fmin, width, lon, lat, f, fdot):
        
        lon = lon/(2*np.pi)
        lat = lat/(0.5*np.pi)
        f = (f - fmin)/width
        fdot = (fdot*1e9 + 1)/2

        return lon, lat, f, fdot

    def transform_normlonlat_lonlat(self, fmin, width, lon, lat, f, fdot):
        
        lon = lon*2*np.pi
        lon = np.remainder(lon, 2*np.pi)        
        lat = np.abs(lat)*(0.5*np.pi)
        """
        if type(lat) in [float, int, np.double]:
            if lat > 0.5*np.pi:
                lat = np.nan
            if lat < 0:
                lat = np.nan
        elif type(lat) in [np.array, list, np.ndarray]:
            lat[lat > 0.5*np.pi] = np.nan
            lat[lat < 0] = np.nan
        """
        f = f*width + fmin
        fdot = (fdot*2 - 1)*1e-9

        return lon, lat, f, fdot


            
            
class LoadCW(CW):
    
    def __init__(self, load_dirs, frange_low=40,frange_high = 500,snrrange_low = 20,snrrange_high = 200, load_powers = False, fmin = None, fmax = None, snr_low = None, snr_high = None, param_type = "ph_A_off_fdot", shuffle = False, stat_threshold = None, chunk_size = 100, batch_size=500, chunk_load = False, test_data = False, lat_low = None, lat_high=None, data_type = "cvae"):
        super().__init__()
        if type(load_dirs) == "str":
            self.load_dirs = [load_dirs]
        else:
            self.load_dirs = load_dirs

        self.shuffle = shuffle
        # define the ranges for the directories signals are stored in
        self.frange_low = frange_low if type(frange_low) == list else list(frange_low)
        self.frange_high = frange_high if type(frange_high) == list else list(frange_high)
        self.snrrange_low = snrrange_low
        self.snrrange_high = snrrange_high
        self.lat_low = lat_low
        self.lat_high = lat_high

        self.load_powers = load_powers
        self.max_A = max(frange_high)*self.r0*self.omega0/self.c
        self.file_split = 1000
        ## hardcoded need to change
        #self.epochs = np.arange(362)*(1238296083.0 - 1238209683.0) + 1238209683.0
        self.epochs = None
        self.param_type = param_type
        # define the ranges for data to loaded in
        self.snr_low = self.snrrange_low if snr_low is None else snr_low
        self.snr_high = self.snrrange_high if snr_high is None else snr_high
        self.fmin = self.frange_low if fmin is None else fmin
        self.fmax = self.frange_high if fmax is None else fmax
        self.stat_threshold = stat_threshold
        self.chunk_size = chunk_size
        self.chunk_iter = 0
        self.batch_size = batch_size
        self.chunk_load = chunk_load
        self.test_data = test_data
        self.data_type = data_type

    def __len__(self):
        if self.chunk_load:
            return(np.floor(len(self.data)/self.batch_size).astype(int))
        else:
            return len(self.data)
    
    def __getitem__(self, index):
        if self.data_type == "cvae":
            if self.chunk_load:
                return self.data[index*self.batch_size:(index + 1)*self.batch_size], self.labels[index*self.batch_size:(index + 1)*self.batch_size]
            else:
                return self.data[index],self.labels[index]
        elif self.data_type == "dec":
            if self.chunk_load:
                return self.data[index*self.batch_size:(index + 1)*self.batch_size]
            else:
                return self.data[index],self.locs[index]
        elif self.data_type == "cvae_freq":
            if self.chunk_load:
                return self.data[index*self.batch_size:(index + 1)*self.batch_size], self.labels[index*self.batch_size:(index + 1)*self.batch_size], self.freqs[index*self.batch_size:(index + 1)*self.batch_size]
            else:
                return self.data[index],self.labels[index], self.freqs[index]

    def load_new_chunk(self):
        if self.chunk_size*self.batch_size > len(self.all_data):
            print("Using all data")
            st_ind = 0
            en_ind = len(self.all_data)
            self.chunk_iter = 0
        else:
            st_ind, en_ind = self.chunk_iter*self.chunk_size*self.batch_size,(self.chunk_iter + 1)*self.chunk_size*self.batch_size 
            if en_ind > len(self.all_data):
                st_ind = len(self.all_data) - self.chunk_size*self.batch_size - 1
                en_ind = len(self.all_data)
                print("resetting chunk index ....")
                self.chunk_iter = 0

        print("chunk_number: ", self.chunk_iter)
        self.data = self.all_data[st_ind:en_ind]
        self.labels = self.all_labels[st_ind:en_ind]
        self.widths = self.all_widths[st_ind:en_ind]
        self.parameters = self.all_parameters[st_ind:en_ind]
        self.freqs = self.all_freqs[st_ind:en_ind]
        self.chunk_iter += 1
            
    def gen_train_data(self, numdata):
        
        if not hasattr(self, "filenames"):
            self.get_filenames()
            
        if numdata == "all" or self.test_data:
            num_testdata = numdata
            numdata = len(self.filenames)*1e2
        else:
            num_testdata = numdata

        # list all files
        self.file_split = int(self.filenames[0].split("/")[-1].split("_")[-1].strip(".pkl"))
        # get list of fname indicies for other directories
        self.all_file_split = [int(fname.split("/")[-1].split("_")[-1].strip(".pkl")) for fname in self.filenames]
        
        # find total number of files by sum of filenames
        file_split_cumsum = np.cumsum(self.all_file_split)
        # if amount if data is less than available, set to fname which contains that value
        if numdata <= file_split_cumsum[-1]:
            num_filenames = int(np.where(file_split_cumsum >= numdata)[0].min()) + 1
        else:
            num_filenames =  len(self.filenames)
            numdata = file_split_cumsum[-1]

        indices = np.arange(numdata).astype(int)
        # get indicies for each file
        temp_indices = self.all_file_split[:num_filenames]
        # set the number of peices of data to get from each file
        if numdata < file_split_cumsum[-1]:
            temp_indices[-1] -= file_split_cumsum[num_filenames-1] - numdata
        temp_indices_split = [np.arange(inds) for inds in temp_indices]

        
        #num_filenames = np.ceil(numdata/self.file_split).astype(int)
        #indices = np.arange(numdata).astype(int)
        #temp_indices = indices % self.file_split
        #temp_indices_split = np.split(temp_indices, np.where(np.diff(temp_indices) < 0)[0] + 1)
        
        if not self.chunk_load:
            self.data, self.labels, self.widths, self.parameters = self.load_files(self.filenames[:num_filenames], temp_indices_split)
            self.freqs = np.array([fr["fmin"] for fr in self.parameters])
            if self.test_data:
                #rng=np.random.RandomState(100)
                test_inds = random.sample(list(np.arange(0,len(self.data))), num_testdata)
                self.data = self.data[test_inds]
                self.labels = self.labels[test_inds]
                self.widths = self.widths[test_inds]
                self.parameters = self.parameters[test_inds]
                self.freqs = self.freqs[test_inds]
            if len(self.data) != len(self.parameters):
                print("parameters and data not the same length")

        else:
            self.all_data, self.all_labels, self.all_widths, self.all_parameters = self.load_files(self.filenames[:num_filenames], temp_indices_split)
            self.all_freqs = np.array([fr["fmin"] for fr in self.all_parameters])
            if self.test_data:
                #rng=np.random.RandomState(100)
                #print(num_testdata, len(self.all_data))
                test_inds = random.sample(list(np.arange(0, len(self.all_data))), num_testdata)
                self.data = self.all_data[test_inds]
                self.labels = self.all_labels[test_inds]
                self.widths = self.all_widths[test_inds]
                self.parameters = self.all_parameters[test_inds]
                self.freqs = self.all_freqs[test_inds]

            if len(self.all_data) != len(self.all_parameters):
                print("parameters and data not the same length")

    def get_filenames(self):
        
        filenames = []
        for load_dir in self.load_dirs:
            for fr_ind in range(len(self.frange_low)):
                load_sub_dir = os.path.join(load_dir, "paths", "band_{}_{}".format(self.frange_low[fr_ind], self.frange_high[fr_ind]), "snr_{}_{}".format(self.snrrange_low, self.snrrange_high))
                filenames.extend([os.path.join(load_sub_dir,f) for f in os.listdir(load_sub_dir)])
            
        self.filenames = natsort.natsorted(filenames)
        if not self.test_data:
            if self.shuffle:
                np.random.shuffle(self.filenames)

    def load_files(self, filenames, temp_indices):

        siginj = cw.GenerateSignal()
        siginj.earth_ephem = self.earth_ephem
        siginj.sun_ephem   = self.sun_ephem
        siginj.get_edat()
        
        tot_data = []
        tot_params = []
        tot_all_params = []
        tot_widths = []
        tot_loc_true = []

        for i,filename in enumerate(filenames):
            #print(self.file_split, filename)
            with open(filename.replace("/paths/","/pars/"),"rb") as f:
                params = np.array(pkl.load(f))[temp_indices[i]]

            with open(filename,"rb") as f:
                temp_data = np.array(pkl.load(f))
                data = temp_data[temp_indices[i]]

            if self.load_powers:
                with open(filename.replace("/paths/", "/powers/"), "rb") as f:
                    powers = np.array(pkl.load(f))
                    # subtract mean (N*num deg free) 10*width = N
                    scale_fact = 10*np.array([p["width"] for p in params])
                    degfree = 96 * scale_fact
                    powers1 = (powers[temp_indices[i],1] - degfree[:,None])/degfree[:,None]
                    powers2 = (powers[temp_indices[i],2] - degfree[:,None])/degfree[:,None]

            if self.stat_threshold is not None: 
                with open(filename.replace("/paths/","/stats/"),"rb") as f:
                    stats = np.array(pkl.load(f))[temp_indices[i]]
               
        
            save_pars = []
            widths = []
            if self.param_type == "ph_A_off_fdot_snr":
                num_pars = 5
            elif self.param_type == "ph_A_off_fdot":
                num_pars = 4
            for index,pars in enumerate(params):
                #print(pars)
                # change to ecliptic lon and lat
                pars["fmin"] = pars["fmin"]
                pars["fmax"] = pars["fmax"]
                pars["delta"] = np.arcsin(pars["sindelta"])
                pars["lon"], pars["lat"] = self.transform_geo_ecliptic(copy.copy(pars["alpha"]),copy.copy(pars["delta"]))
                # take abs as only northern hemisphere
                pars["lat"] = np.abs(pars["lat"])
                # reconvert the sky position back to geocentric
                pars["alpha"], pars["delta"] = self.transform_ecliptic_geo(copy.copy(pars["lon"]),copy.copy(pars["lat"]))
                pars["sindelta"] = np.sin(pars["delta"])
                pars["fdot"] = pars["fd"]
                if self.stat_threshold is not None:
                    pars["stat"] = stats[index]

                # convert to amplitude parameters
                off, A, ph, norm_fdot = self.transform_ecliptic_amp(copy.copy(pars["fmin"]), copy.copy(pars["width"]), copy.copy(pars["f"]), copy.copy(pars["fd"]), copy.copy(pars["lon"]), copy.copy(pars["lat"]))
                
                tr_lon, tr_lat, off, tr_fdot = self.transform_lonlat_normlonlat(copy.copy(pars["fmin"]), copy.copy(pars["width"]),copy.copy(pars["lon"]), copy.copy(pars["lat"]), copy.copy(pars["f"]), copy.copy(pars["fdot"]))
                if self.param_type == "ph_A_off_fdot_snr":
                    save_pars.append([ph, A, off, norm_fdot, pars["snr"]/200])
                    num_pars = 5
                elif self.param_type == "ph_A_off_fdot":
                    save_pars.append([ph, A, off, norm_fdot])
                    num_pars = 4
                elif self.param_type == "lon_lat_off_fdot":
                    save_pars.append([tr_lon, tr_lat, off, tr_fdot])
                    num_pars = 4
                widths.append(pars["width"])

            get_index = []
            for ind, par in enumerate(params):
                append = True
                if self.snr_low <= par["snr"] <= self.snr_high:
                    pass
                else:
                    append = False
                if self.stat_threshold is not None:
                    if stats[ind] >= self.stat_threshold:
                        pass
                    else:
                        append = False
                
                if self.lat_low is not None and self.lat_high is not None:
                    if self.lat_low <= par["lat"] <= self.lat_high:
                        pass
                    else:
                        append = False

                if self.fmin is not None and self.fmax is not None:
                    if par["fmin"] <= self.fmin or par["fmin"] >= self.fmax:
                        append = False
                    """
                    if par["f"] < par["fmin"] + 0.25*0.2 or par["f"] > par["fmax"] - 0.25*0.2:
                        append = False
                    """
                if append:
                    get_index.append(ind)

            if self.epochs is None:
                epochs = data[0, 0]
                self.epochs = epochs
                #self.epochs = np.arange(len(data[0,1]))*24*3600 + data[0][0,0]

            temp_widths = np.array(widths)[get_index]
            # rescale to be between 0 and 1
            # when using doouble width use this
            if 40 <= par["fmin"] < 500:
                scale = 1.0 
            elif 500 <= par["fmin"] < 1000:
                scale = 1./2.0
            elif 1000 <= par["fmin"] < 1500:
                scale = 1./3.0
            elif 1500 <= par["fmin"] < 2000:
                scale = 1./4.0

            temp_tracks = data[get_index,1]/(temp_widths[:,None]*scale*1800)
            #temp_tracks = data[get_index,1]/(2*180.)
            #else this
            #temp_tracks = data[get_index,1]/(180.)
            temp_params = params[get_index]
            
            temp_save_pars = np.reshape(np.array(save_pars)[get_index], (temp_tracks.shape[0], num_pars))

            temp_loc_true = []
            for index in range(len(temp_tracks)):
                siginj.tref = self.epochs[0]
                siginj.alpha = temp_params[index]["alpha"]
                siginj.delta = temp_params[index]["delta"]
                siginj.f = [temp_params[index]["f"], temp_params[index]["fd"]]
                trackinj = siginj.get_pulsar_path(epochs=self.epochs,edat=siginj.edat,det="SSB")
                temp_track = temp_tracks[index]*temp_params[index]["width"] + temp_params[index]["fmin"]
                abstrackdiff = np.array(np.abs(trackinj - temp_track)*1800 < 2).astype(int)
                temp_loc_true.append(abstrackdiff)

            temp_loc_true = np.reshape(np.array(temp_loc_true), (temp_tracks.shape[0], temp_tracks.shape[1]))

            temp_save_pars = np.concatenate([temp_save_pars, temp_loc_true], axis = 1)
            del temp_loc_true

            if self.load_powers:
                temp_data = np.reshape(np.array([np.array([temp_tracks[ti], powers1[ti], powers2[ti]]) for ti in range(len(temp_tracks))]), (temp_tracks.shape[0], 3, temp_tracks.shape[1]))
            else:
                temp_data = np.reshape(temp_tracks, (temp_tracks.shape[0], 1, temp_tracks.shape[1]))

            tot_data.append(temp_data)
            tot_all_params.append(temp_params)
            tot_params.append(temp_save_pars)
            tot_widths.append(temp_widths)


        if len(tot_data) == 1:
            ret_dat =  tot_data[0], tot_params[0], tot_widths[0], tot_all_params[0]
        else:
            tot_data = np.concatenate(tot_data)
            tot_params = np.concatenate(tot_params)
            tot_widths = np.concatenate(tot_widths)
            tot_all_params = np.concatenate(tot_all_params)
            ret_dat =  tot_data, tot_params, tot_widths, tot_all_params

        if self.shuffle:
            newinds = np.random.randint(0, len(ret_dat[0]), size = len(ret_dat[0]))

            ret_dat = ret_dat[0][newinds], ret_dat[1][newinds], ret_dat[2][newinds], ret_dat[3][newinds]

        return ret_dat