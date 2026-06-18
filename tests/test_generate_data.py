"""Comprehensive unit tests for generate_data module."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from soapcw.cw import GenerateSignal


class TestGenerateSignalInit(unittest.TestCase):
    """Test GenerateSignal initialization."""

    def test_init_with_all_params(self):
        """Test initialization with all parameters."""
        sig = GenerateSignal(
            alpha=0.5,
            delta=0.5,
            psi=0.1,
            phi0=0.2,
            cosi=0.3,
            h0=1e-24,
            f=[100.0, 1e-8],
            tref=1234567890,
            snr=10.0,
        )
        self.assertEqual(sig.alpha, 0.5)
        self.assertEqual(sig.delta, 0.5)
        self.assertEqual(sig.psi, 0.1)
        self.assertEqual(sig.phi0, 0.2)
        self.assertEqual(sig.cosi, 0.3)
        self.assertEqual(sig.h0, 1e-24)
        self.assertEqual(sig.f, [100.0, 1e-8])
        self.assertEqual(sig.tref, 1234567890)
        self.assertEqual(sig.snr, 10.0)

    def test_init_with_minimal_params(self):
        """Test initialization with minimal parameters."""
        sig = GenerateSignal(f=[100.0, 0.0])
        self.assertEqual(sig.f, [100.0, 0.0])
        self.assertIsNone(sig.alpha)
        self.assertIsNone(sig.h0)

    def test_init_ephemeris_defaults(self):
        """Test that ephemeris files default correctly."""
        sig = GenerateSignal()
        self.assertEqual(sig.earth_ephem, "earth00-40-DE430.dat.gz")
        self.assertEqual(sig.sun_ephem, "sun00-40-DE430.dat.gz")

    def test_init_custom_ephemeris(self):
        """Test custom ephemeris file paths."""
        sig = GenerateSignal(earth_ephem="custom_earth.dat", sun_ephem="custom_sun.dat")
        self.assertEqual(sig.earth_ephem, "custom_earth.dat")
        self.assertEqual(sig.sun_ephem, "custom_sun.dat")


class TestGenerateSignalProperties(unittest.TestCase):
    """Test GenerateSignal properties and setters."""

    def setUp(self):
        """Set up test fixtures."""
        self.sig = GenerateSignal()

    def test_earth_ephem_setter_none(self):
        """Test earth_ephem setter with None."""
        self.sig.earth_ephem = None
        self.assertEqual(self.sig.earth_ephem, "earth00-40-DE430.dat.gz")

    def test_sun_ephem_setter_none(self):
        """Test sun_ephem setter with None."""
        self.sig.sun_ephem = None
        self.assertEqual(self.sig.sun_ephem, "sun00-40-DE430.dat.gz")

    def test_earth_ephem_setter_custom(self):
        """Test earth_ephem setter with custom value."""
        self.sig.earth_ephem = "my_earth.dat"
        self.assertEqual(self.sig.earth_ephem, "my_earth.dat")


class TestDetectorVelocity(unittest.TestCase):
    """Test detector velocity calculations."""

    def setUp(self):
        """Set up test fixtures."""
        self.sig = GenerateSignal(
            alpha=0.5, delta=0.5, psi=0, phi0=0.0, cosi=0.1, h0=1e-15, f=[200, 0.0]
        )

    def test_detector_velocity_H1(self):
        """Test detector velocity for H1."""
        self.sig.get_edat()
        detv = self.sig.detector_velocity(self.sig.edat, 1234567987, "H1")
        self.assertEqual(len(detv), 3)
        self.assertIsInstance(detv, np.ndarray)
        # Test specific known values
        np.testing.assert_almost_equal(
            detv,
            np.array([-0.00005208568783271258, -0.00007807592362349714, -0.00003427742373117495]),
            decimal=10,
        )

    def test_detector_velocity_L1(self):
        """Test detector velocity for L1."""
        self.sig.get_edat()
        detv = self.sig.detector_velocity(self.sig.edat, 1234567987, "L1")
        self.assertEqual(len(detv), 3)
        self.assertIsInstance(detv, np.ndarray)

    def test_get_detector_velocities_single_epoch(self):
        """Test get_detector_velocities with single epoch."""
        self.sig.get_detector_velocities([1234567987], "H1")
        self.assertEqual(self.sig.det_vels.shape, (1, 3))
        np.testing.assert_almost_equal(
            self.sig.det_vels[0],
            np.array([-0.00005208568783271258, -0.00007807592362349714, -0.00003427742373117495]),
            decimal=10,
        )

    def test_get_detector_velocities_multiple_epochs(self):
        """Test get_detector_velocities with multiple epochs."""
        epochs = np.array([1234567987, 1234569787, 1234571587])
        self.sig.get_detector_velocities(epochs, "H1")
        self.assertEqual(self.sig.det_vels.shape, (3, 3))
        # Check no NaN values
        self.assertFalse(np.any(np.isnan(self.sig.det_vels)))


class TestPulsarPath(unittest.TestCase):
    """Test pulsar path calculations."""

    def setUp(self):
        """Set up test fixtures."""
        self.sig = GenerateSignal(
            alpha=0.5, delta=0.5, psi=0, phi0=0.0, cosi=0.1, h0=1e-15, f=[200, 0.0]
        )

    def test_pulsar_path_single_epoch(self):
        """Test pulsar path for single epoch."""
        path = self.sig.get_pulsar_path(np.array([1234567987]), "H1")
        self.assertEqual(len(path), 1)
        self.assertAlmostEqual(path[0], 199.98212067259186, places=9)

    def test_pulsar_path_multiple_epochs(self):
        """Test pulsar path for multiple epochs."""
        epochs = np.array([1234567987, 1234569787, 1234571587])
        path = self.sig.get_pulsar_path(epochs, "H1")
        self.assertEqual(len(path), 3)
        # Check all values are close to f[0] (200 Hz)
        self.assertTrue(np.all(np.abs(path - 200) < 0.1))

    def test_pulsar_path_with_spindown(self):
        """Test pulsar path with frequency derivative."""
        self.sig.f = [200.0, 1e-8]
        epochs = np.array([1234567987, 1234567987 + 86400])  # 1 day apart
        path = self.sig.get_pulsar_path(epochs, "H1")
        # Path should show frequency evolution
        self.assertNotAlmostEqual(path[0], path[1])

    def test_pulsar_path_tref_none(self):
        """Test that tref defaults to first epoch."""
        self.sig.tref = None
        epochs = np.array([1234567987, 1234569787])
        path = self.sig.get_pulsar_path(epochs, "H1")
        self.assertEqual(self.sig.tref, epochs[0])


class TestSNRCalculation(unittest.TestCase):
    """Test SNR calculations."""

    def setUp(self):
        """Set up test fixtures."""
        self.sig = GenerateSignal(
            alpha=0.5,
            delta=0.5,
            psi=0.1,
            phi0=0.2,
            cosi=0.3,
            h0=1e-24,
            f=[100.0, 0.0],
            tref=1234567890,
        )

    def test_get_snr2_basic(self):
        """Test basic SNR^2 calculation."""
        epochs = np.array([1234567890, 1234567890 + 1800])
        snr2 = self.sig.get_snr2(
            epochs=epochs,
            alpha=0.5,
            delta=0.5,
            psi=0.1,
            phi0=0.2,
            cosi=0.3,
            Sn=1e-46,
            det="H1",
            tstart=epochs[0],
            nsft=2,
            tsft=1800,
            h0=1e-24,
            antenna=False,
        )
        self.assertEqual(len(snr2), 2)
        self.assertTrue(np.all(snr2 >= 0))

    def test_get_snr2_with_antenna(self):
        """Test SNR^2 calculation with antenna pattern."""
        epochs = np.array([1234567890, 1234567890 + 1800])
        snr2 = self.sig.get_snr2(
            epochs=epochs,
            alpha=0.5,
            delta=0.5,
            psi=0.1,
            phi0=0.2,
            cosi=0.3,
            Sn=1e-46,
            det="H1",
            tstart=epochs[0],
            nsft=2,
            tsft=1800,
            h0=1e-24,
            antenna=True,
        )
        self.assertEqual(len(snr2), 2)

    def test_get_snr2_variable_noise(self):
        """Test SNR^2 with time-varying noise floor."""
        epochs = np.array([1234567890, 1234567890 + 1800, 1234567890 + 3600])
        Sn = np.array([1e-46, 2e-46, 1.5e-46])
        snr2 = self.sig.get_snr2(
            epochs=epochs,
            alpha=0.5,
            delta=0.5,
            psi=0.1,
            phi0=0.2,
            cosi=0.3,
            Sn=Sn,
            det="H1",
            tstart=epochs[0],
            nsft=3,
            tsft=1800,
            h0=1e-24,
            antenna=False,
        )
        self.assertEqual(len(snr2), 3)

    def test_get_snr2_missing_Sn_raises(self):
        """Test that missing Sn raises exception."""
        epochs = np.array([1234567890, 1234567890 + 1800])
        with self.assertRaises(Exception) as context:
            self.sig.get_snr2(
                epochs=epochs,
                alpha=0.5,
                delta=0.5,
                psi=0.1,
                phi0=0.2,
                cosi=0.3,
                Sn=None,
                det="H1",
                tstart=epochs[0],
                nsft=2,
                tsft=1800,
                h0=1e-24,
            )
        self.assertIn("noise floor", str(context.exception).lower())


class TestAntennaPattern(unittest.TestCase):
    """Test antenna pattern calculations."""

    def setUp(self):
        """Set up test fixtures."""
        self.sig = GenerateSignal(
            alpha=0.5,
            delta=0.5,
            psi=0.1,
            phi0=0.2,
            cosi=0.3,
            h0=1e-24,
            f=[100.0, 0.0],
        )

    def test_av_antenna_no_antenna(self):
        """Test antenna pattern calculation with antenna=False."""
        result = self.sig.av_antenna(
            alpha=0.5,
            delta=0.5,
            det="H1",
            tstart=1234567890,
            nsft=2,
            tsft=1800,
            antenna=False,
        )
        np.testing.assert_array_equal(result, np.array([[1, 1, 1]]))

    def test_av_antenna_with_antenna_average(self):
        """Test antenna pattern with averaging."""
        result = self.sig.av_antenna(
            alpha=0.5,
            delta=0.5,
            det="H1",
            tstart=1234567890,
            nsft=2,
            tsft=1800,
            antenna=True,
            average=True,
        )
        self.assertEqual(result.shape, (3,))

    def test_av_antenna_with_antenna_no_average(self):
        """Test antenna pattern without averaging."""
        result = self.sig.av_antenna(
            alpha=0.5,
            delta=0.5,
            det="H1",
            tstart=1234567890,
            nsft=2,
            tsft=1800,
            antenna=True,
            average=False,
        )
        self.assertEqual(result.shape, (2, 3))

    def test_av_antenna_different_detectors(self):
        """Test antenna pattern for different detectors."""
        for det in ["H1", "L1", "V1"]:
            result = self.sig.av_antenna(
                alpha=0.5,
                delta=0.5,
                det=det,
                tstart=1234567890,
                nsft=2,
                tsft=1800,
                antenna=True,
                average=True,
            )
            self.assertEqual(result.shape, (3,))


class TestEphemerisData(unittest.TestCase):
    """Test ephemeris data loading."""

    def setUp(self):
        """Set up test fixtures."""
        self.sig = GenerateSignal(f=[100.0, 0.0])

    def test_get_edat_success(self):
        """Test successful ephemeris data loading."""
        self.sig.get_edat()
        self.assertIsNotNone(self.sig.edat)
        self.assertIsNotNone(self.sig.edat_p)
        self.assertEqual(len(self.sig.edat_p), 2)

    def test_get_edat_called_multiple_times(self):
        """Test that get_edat can be called multiple times."""
        self.sig.get_edat()
        edat_first = self.sig.edat
        self.sig.get_edat()
        # Should succeed without error
        self.assertIsNotNone(self.sig.edat)


class TestFresnelPower(unittest.TestCase):
    """Test Fresnel power calculations."""

    def setUp(self):
        """Set up test fixtures."""
        self.sig = GenerateSignal()
        # Create spectrogram object to access fresnel_power
        from soapcw.cw.generate_data import SimulateGaussianNoiseSpectrogram

        self.spect = SimulateGaussianNoiseSpectrogram(self.sig)

    def test_fresnel_power_on_signal(self):
        """Test Fresnel power at signal frequency."""
        power = self.spect.fresnel_power(f=100.0, f0=100.0, tsft=1800, alp=1e-8)
        self.assertGreater(power, 0)
        self.assertIsInstance(power, (float, np.floating))

    def test_fresnel_power_off_signal(self):
        """Test Fresnel power away from signal frequency."""
        power = self.spect.fresnel_power(f=105.0, f0=100.0, tsft=1800, alp=1e-8)
        self.assertGreater(power, 0)

    def test_fresnel_power_zero_spindown(self):
        """Test Fresnel power with zero frequency derivative."""
        power = self.spect.fresnel_power(f=100.0, f0=100.0, tsft=1800, alp=1e-20)
        self.assertGreater(power, 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_negative_h0(self):
        """Test with negative h0."""
        sig = GenerateSignal(h0=-1e-24, f=[100.0, 0.0])
        self.assertEqual(sig.h0, -1e-24)

    def test_zero_frequency(self):
        """Test with zero frequency."""
        sig = GenerateSignal(f=[0.0, 0.0])
        self.assertEqual(sig.f, [0.0, 0.0])

    def test_large_frequency_derivative(self):
        """Test with large frequency derivative."""
        sig = GenerateSignal(f=[100.0, 1e-5])
        self.assertEqual(sig.f[1], 1e-5)

    def test_extreme_sky_position(self):
        """Test with extreme sky positions."""
        sig = GenerateSignal(alpha=0.0, delta=np.pi / 2)
        self.assertEqual(sig.alpha, 0.0)
        self.assertAlmostEqual(sig.delta, np.pi / 2)

    def test_extreme_cosi(self):
        """Test with extreme cosi values."""
        sig1 = GenerateSignal(cosi=-1.0)
        sig2 = GenerateSignal(cosi=1.0)
        self.assertEqual(sig1.cosi, -1.0)
        self.assertEqual(sig2.cosi, 1.0)


if __name__ == "__main__":
    unittest.main()
