#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `soap` package."""


import unittest

import soapcw as soap
import numpy as np

class TestSoap(unittest.TestCase):
    """Tests for `soap` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_transition_matrix_1d(self):
        """ Testing the outputs of 1d transition matrix"""
        tr = soap.tools.transition_matrix(2)

        transition = np.array([-1.3862943611198906, -0.6931471805599453, -1.3862943611198906])

        np.testing.assert_almost_equal(tr , transition)

    def test_transition_matrix_2d(self):
        """Testing the setup for 2d transition matrix"""

        tr_2d = np.array([[[-461.90331295992905,   -231.64480366052445,   -461.90331295992905,  ],
                            [-231.64480366052445,     -1.3862943611198906, -231.64480366052445,  ],
                            [-461.90331295992905,   -231.64480366052445,   -461.90331295992905,  ]],

                            [[-461.2101657793691,    -230.95165647996453,   -461.2101657793691,   ],
                            [-230.95165647996453,     -0.6931471805599453, -230.95165647996453,  ],
                            [-461.2101657793691,    -230.95165647996453,   -461.2101657793691,   ]],

                            [[-461.90331295992905,   -231.64480366052445,   -461.90331295992905,  ],
                            [-231.64480366052445,     -1.3862943611198906, -231.64480366052445,  ],
                            [-461.90331295992905,   -231.64480366052445,   -461.90331295992905,  ]]])

        np.testing.assert_almost_equal(tr_2d , soap.tools.transition_matrix_2d(2, 1e100, 1e100))

    def test_basic_single(self):
        """Testing basic outputs for the single detector search"""

        arr = np.array([[1,1,1,1,1], 
                        [1,2,3,4,5], 
                        [5,4,3,2,1], 
                        [1,5,4,3,6], 
                        [7,3,6,3,5]])

        sp = soap.single_detector([0,1,0],arr)

        output_map = np.array([[ 1.,  1.,  1.,  1.,  1.],
                            [ 3.,  4.,  5.,  6.,  7.],
                            [ 9.,  9.,  9.,  9.,  9.], 
                            [11., 15., 14., 13., 16.],
                            [22., 19., 21., 19., 22.]])

        vit_track = np.array([1, 1, 1, 1, 0])

        self.assertEqual(sp.max_end_prob, 22.0)
        np.testing.assert_almost_equal(sp.vit_track , vit_track)
        np.testing.assert_almost_equal(sp.V , output_map)

    def test_two_det_simple(self):
        """Testing the outputs fo the two detector search with summed statistic"""

        arr1 = np.array([[1,1,1,1,1], 
                        [1,2,3,4,5], 
                        [5,4,3,2,1], 
                        [1,5,4,3,6], 
                        [7,3,6,3,5]])

        arr2 = np.array([[1,1,2,1,1], 
                        [1,2,3,4,5], 
                        [5,8,3,2,1], 
                        [1,1,1,1,6], 
                        [7,3,6,6,5]])

        tr = soap.tools.transition_matrix_2d(2, 1e100, 1e100)

        sp = soap.two_detector(tr,arr1, arr2)

        output_map = np.array([[ 0.3068528194400547,  0.3068528194400547,  1.3068528194400546,
                                0.3068528194400547,  0.3068528194400547],
                                [ 1.6137056388801092,  3.920558458320164,   6.613705638880109,
                                7.920558458320164,   9.61370563888011  ],
                                [12.534264097200273,  17.22741127776022,   12.534264097200273,
                                12.227411277760218,  10.920558458320164 ],
                                [17.841116916640328,  22.53426409720027,   20.841116916640328,
                                15.534264097200273,  22.841116916640328 ],
                                [35.147969736080384,  27.841116916640324,  33.147969736080384,
                                30.454822555520437,  32.147969736080384 ]])

        vit_track = np.array([2, 2, 1, 1, 0])

        self.assertEqual(sp.max_end_prob, 35.14796829223633)
        np.testing.assert_almost_equal(sp.vit_track , vit_track)
        np.testing.assert_almost_equal(sp.V , output_map)


