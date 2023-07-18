from astropy.coordinates import SkyCoord
import numpy as np

def transform_ecliptic_geo(lon, lat):
    """ transform ecliptic coords to geocentric coords with astropy"""
    skyp = SkyCoord(lon = lon,lat=lat,unit="rad",frame="barycentrictrueecliptic")
    skyp_icrs = skyp.icrs
    alpha, delta = skyp_icrs.data.lon.rad,skyp_icrs.data.lat.rad
    return alpha, delta

def transform_normlonlat_lonlat(fmin, width, lon, lat, f, fdot):
    """_summary_                                                                                                        
    Args:                                                                                                                                                            
        fmin (_type_): _description_             
        width (_type_): _description_                                                                                                                                
        lon (_type_): _description_                     
        lat (_type_): _description_            
        f (_type_): _description_                  
        fdot (_type_): _description_      
    Returns:                      
        _type_: _description_                    
    """
    lon = lon*2*np.pi
    lon = np.remainder(lon, 2*np.pi)
    lat = np.abs(lat)*(0.5*np.pi)

    f = f*width + fmin
    fdot = (fdot*2 - 1)*1e-9

    return lon, lat, f, fdot

def transform_normlonlat_alphadelta(fmin, width, lon, lat, f, fdot):
    """_summary_

    Args:
        fmin (_type_): _description_
        width (_type_): _description_
        lon (_type_): _description_
        lat (_type_): _description_
        f (_type_): _description_
        fdot (_type_): _description_

    Returns:
        _type_: _description_
    """

    
    lon, lat, f, fdot = transform_normlonlat_lonlat(fmin, width, lon, lat, f, fdot)

    alpha, delta = transform_ecliptic_geo(lon, lat)

    return alpha, delta, f, fdot
