
import unittest

from soapcw import cw
import numpy as np



class TestCWLoad(unittest.TestCase):
    """Tests for cw loading part of the SOAP package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.sig = cw.GenerateSignal(alpha=0.5, delta=0.5, psi=0,phi0=0.0,cosi=0.1,h0=1e-15,f=[200, 0.0])

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_edat_load(self):
        self.sig.get_edat()

    def test_detector_velocity(self):
        self.sig.get_edat()
        detv = self.sig.detector_velocity(self.sig.edat, 1234567987, "H1")
        np.testing.assert_almost_equal(detv , np.array([-0.00005208568783271258, -0.00007807592362349714, -0.00003427742373117495]))

    def test_get_detector_velocities(self):
        self.sig.get_detector_velocities([1234567987], "H1")
        np.testing.assert_almost_equal(self.sig.det_vels[0] , np.array([-0.00005208568783271258, -0.00007807592362349714, -0.00003427742373117495]))

    def test_pulsar_path(self):
        self.assertAlmostEqual(199.98212067259186, self.sig.get_pulsar_path(np.array([1234567987]),"H1")[0])