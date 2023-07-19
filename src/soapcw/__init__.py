# -*- coding: utf-8 -*-
from __future__ import absolute_import
from importlib.metadata import PackageNotFoundError, version
"""Top-level package for soapcw."""

__author__ = """Joe Bayley"""
__email__ = 'joseph.bayley@glasgow.ac.uk'
try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"
from .soap import single_detector, two_detector, three_detector, single_detector_gaps
from .tools import tools, plots
from .cnn import __init__
from .neville import __init__
from .line_aware_stat import __init__
from .cw import __init__
from . import soap_config_parser

#try:
#    from .lookup_table import gen_lookup_python, save_lookup, gen_lookup
#except:
#    from .lookup_table import gen_lookup_python, save_lookup

