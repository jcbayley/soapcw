import sys
import os
import appdirs
import requests

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "data", "data.txt")

LAL_EPHEMERIS_URL = "https://git.ligo.org/lscsoft/lalsuite/raw/master/lalpulsar/lib/{}"
EPHEMERIS_CACHE_DIR = appdirs.user_cache_dir(appname="soapcw", appauthor=False)

def download_ephemeris_file(url):
    """
    !!! This is taken from Matt pitkins cwinpy package https://git.ligo.org/cwinpy/cwinpy/-/blob/master/cwinpy/utils.py !!!
    Download and cache an ephemeris files from a given URL. If the file has
    already been downloaded and cached it will just be retrieved from the cache
    location.

    Parameters
    ----------
    url: str
        The URL of the file to download.
    """

    fname = os.path.basename(url)  # extract the file name
    fpath = os.path.join(EPHEMERIS_CACHE_DIR, fname)

    if os.path.isfile(fpath):
        # return previously cached file
        return fpath

    # try downloading the file
    try:
        ephdata = requests.get(url)
    except Exception as e:
        raise RuntimeError(f"Error downloading from {url}\n{e}")

    if ephdata.status_code != 200:
        raise RuntimeError(f"Error downloading from {url}")

    if not os.path.exists(EPHEMERIS_CACHE_DIR):
        try:
            os.makedirs(EPHEMERIS_CACHE_DIR)
        except OSError:
            if not os.path.exists(EPHEMERIS_CACHE_DIR):
                raise
    elif not os.path.isdir(EPHEMERIS_CACHE_DIR):
        raise OSError(f"Cache directory {EPHEMERIS_CACHE_DIR} is not a directory")

    # write out file to cache
    with open(fpath, "wb") as fp:
        fp.write(ephdata.content)

    return fpath


