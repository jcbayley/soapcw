
import unittest
import numpy as np

try:    
    from soapcw.line_aware_stat import gen_lookup
    cpp_implementation = True
except:
    cpp_implementation = False
    print("No cpp implementation available: install boost and gsl")

from soapcw.line_aware_stat import gen_lookup_python


class TestCWLoad(unittest.TestCase):
    """Tests for cw loading part of the SOAP package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.arr = np.linspace(0.1,100,5)
        self.las_py_approx = gen_lookup_python.LineAwareStatistic(self.arr, ndet=1 ,k=2, N=48, signal_prior_width=2, line_prior_width=3, noise_line_model_ratio=0.5, approx=True)
        self.las_py_exact = gen_lookup_python.LineAwareStatistic(self.arr, ndet=1 ,k=2, N=48, signal_prior_width=2, line_prior_width=3, noise_line_model_ratio=0.5, approx=False)
        if cpp_implementation:
            self.las_cpp_exact = gen_lookup.LineAwareStatistic(self.arr, ndet=1 ,k=2, N=48, signal_prior_width=2, line_prior_width=3, noise_line_model_ratio=0.5)

        self.arr2 = np.linspace(1,100,2)
        self.las_py_approx_2det = gen_lookup_python.LineAwareStatistic(self.arr2, ndet=2 ,k=2, N=48, signal_prior_width=2, line_prior_width=3, noise_line_model_ratio=0.5, approx=True)
        self.las_py_exact_2det = gen_lookup_python.LineAwareStatistic(self.arr2, ndet=2 ,k=2, N=48, signal_prior_width=2, line_prior_width=3, noise_line_model_ratio=0.5, approx=False)

        if cpp_implementation:
            self.las_cpp_exact = gen_lookup.LineAwareStatistic(self.arr2, ndet=1 ,k=2, N=48, signal_prior_width=2, line_prior_width=3, noise_line_model_ratio=0.5)


    def tearDown(self):
        """Tear down test fixtures, if any."""


    def test_gen_lookup_1det_py_approx(self):
        """Test the 1 detector lookup table generation approximate python version"""
        signoiseline = np.array([0.4180712556743122,  0.46573706068942955, 0.5238572889042366, 0.5944770691440852,  0.6772228852444641 ])

        np.testing.assert_almost_equal(signoiseline , self.las_py_approx.signoiseline)
    
    def test_gen_lookup_1det_py_exact(self):
        """Test the 1 detector lookup table generation approximate python version"""
        signoiseline = np.array([0.4168356127981346,  0.46470907883461643, 0.5230282098726995, 0.5938407592402759,  0.6767699501326778 ])

        np.testing.assert_almost_equal(signoiseline , self.las_py_exact.signoiseline)

    def test_gen_lookup_1det_cpp_exact(self):
        """Test the 1 detector lookup table generation approximate python version"""
        signoiseline = np.array([[0.4168403438629973,  0.46470910643499247, 0.5230282061301725, 0.5938407592400682,  0.6767699495907581 ] ])

        if cpp_implementation:
            np.testing.assert_almost_equal(signoiseline , self.las_cpp_exact.signoiseline)
    
    def test_gen_lookup_2det_py_approx(self):
        """Test the 1 detector lookup table generation approximate python version"""
        signoiseline = np.array([[0.28139104078142746, 0.3778010481689715 ],
                                    [0.3778010481689715,  0.6919634194611941 ]])

        np.testing.assert_almost_equal(signoiseline , self.las_py_approx_2det.signoiseline)
    
    def test_gen_lookup_2det_py_exact(self):
        """Test the 1 detector lookup table generation approximate python version"""
        signoiseline = np.array([[0.27942796216954896, 0.37667858291387774],
                                [0.37667858291387774, 0.6915352664549508 ]])

        np.testing.assert_almost_equal(signoiseline , self.las_py_exact_2det.signoiseline)

