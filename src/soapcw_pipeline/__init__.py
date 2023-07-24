# -*- coding: utf-8 -*-
from __future__ import absolute_import
from importlib.metadata import PackageNotFoundError, version

__author__ = """Joe Bayley"""
__email__ = "joseph.bayley@glasgow.ac.uk"
try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"
from . import (
    make_dag_files_astro,
    make_dag_files_lines,
    make_html_page,
    run_full_soap_astro,
    run_full_soap_lines,
    soap_config_parser,
)
