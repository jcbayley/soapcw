"""
Tests for soapcw_pipeline.run_full_soap_astro module
"""
import unittest

import numpy as np

from soapcw_pipeline.run_full_soap_astro import av_noise, get_sfts_in_range, set_limit


class TestSetLimit(unittest.TestCase):
    """Test the set_limit function"""

    def test_set_limit_below_cut(self):
        """Test when data max is below the cut"""
        data = np.array([1, 2, 3, 4, 5])
        vmin, vmax = set_limit(data, cut=10)
        self.assertEqual(vmin, 1)
        self.assertEqual(vmax, 5)

    def test_set_limit_above_cut(self):
        """Test when data max is above the cut"""
        data = np.array([1, 2, 3, 4, 200])
        vmin, vmax = set_limit(data, cut=180)
        self.assertEqual(vmin, 1)
        self.assertEqual(vmax, 180)

    def test_set_limit_exactly_at_cut(self):
        """Test when data max equals the cut"""
        data = np.array([1, 2, 3, 4, 180])
        vmin, vmax = set_limit(data, cut=180)
        self.assertEqual(vmin, 1)
        self.assertEqual(vmax, 180)

    def test_set_limit_negative_values(self):
        """Test with negative values"""
        data = np.array([-10, -5, 0, 5, 10])
        vmin, vmax = set_limit(data, cut=20)
        self.assertEqual(vmin, -10)
        self.assertEqual(vmax, 10)


class TestGetSftsInRange(unittest.TestCase):
    """Test the get_sfts_in_range function"""

    def test_get_sfts_in_range_all_in_range(self):
        """Test when all SFTs are in range"""
        sftlist = [
            "H-H1_GWOSC_O2_4KHZ_R1-1164556817-4096.sft",
            "H-H1_GWOSC_O2_4KHZ_R1-1164560913-4096.sft",
            "H-H1_GWOSC_O2_4KHZ_R1-1164565009-4096.sft",
        ]
        tmin = 1164556817
        tmax = 1164570000
        result = get_sfts_in_range(tmin, tmax, sftlist)
        self.assertEqual(len(result), 3)

    def test_get_sfts_in_range_none_in_range(self):
        """Test when no SFTs are in range"""
        sftlist = [
            "H-H1_GWOSC_O2_4KHZ_R1-1164556817-4096.sft",
            "H-H1_GWOSC_O2_4KHZ_R1-1164560913-4096.sft",
        ]
        tmin = 1200000000
        tmax = 1210000000
        result = get_sfts_in_range(tmin, tmax, sftlist)
        self.assertEqual(len(result), 0)

    def test_get_sfts_in_range_partial(self):
        """Test when some SFTs are in range"""
        sftlist = [
            "H-H1_GWOSC_O2_4KHZ_R1-1164556817-4096.sft",
            "H-H1_GWOSC_O2_4KHZ_R1-1164560913-4096.sft",
            "H-H1_GWOSC_O2_4KHZ_R1-1164565009-4096.sft",
        ]
        tmin = 1164560000
        tmax = 1164562000
        result = get_sfts_in_range(tmin, tmax, sftlist)
        self.assertEqual(len(result), 1)
        self.assertIn("1164560913", result[0])

    def test_get_sfts_in_range_boundary_start(self):
        """Test exact boundary at start time"""
        sftlist = [
            "H-H1_GWOSC_O2_4KHZ_R1-1164556817-4096.sft",
        ]
        tmin = 1164556817
        tmax = 1164570000
        result = get_sfts_in_range(tmin, tmax, sftlist)
        self.assertEqual(len(result), 1)

    def test_get_sfts_in_range_boundary_end(self):
        """Test exact boundary at end time"""
        sftlist = [
            "H-H1_GWOSC_O2_4KHZ_R1-1164556817-4096.sft",
        ]
        tmin = 1164550000
        tmax = 1164556817
        result = get_sfts_in_range(tmin, tmax, sftlist)
        # Should not include SFTs exactly at tmax
        self.assertEqual(len(result), 0)


class TestAvNoise(unittest.TestCase):
    """Test the av_noise function"""

    def test_av_noise_exact_multiple(self):
        """Test when noise length is exact multiple of nsft"""
        noise = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = av_noise(noise, nsft=5)
        expected = np.array([np.median([1, 2, 3, 4, 5]), np.median([6, 7, 8, 9, 10])])
        np.testing.assert_array_equal(result, expected)

    def test_av_noise_not_exact_multiple(self):
        """Test when noise length is not exact multiple of nsft"""
        noise = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        result = av_noise(noise, nsft=3)
        expected = np.array(
            [
                np.median([1, 2, 3]),
                np.median([4, 5, 6]),
                np.median([7]),
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_av_noise_with_nans(self):
        """Test handling of NaN values"""
        noise = np.array([1.0, np.nan, 3.0, 4.0, 5.0, np.nan])
        result = av_noise(noise, nsft=3)
        # nanmedian should ignore NaN values
        self.assertEqual(len(result), 2)
        np.testing.assert_almost_equal(result[0], np.nanmedian([1.0, np.nan, 3.0]))
        np.testing.assert_almost_equal(result[1], np.nanmedian([4.0, 5.0, np.nan]))

    def test_av_noise_single_segment(self):
        """Test with nsft larger than noise length"""
        noise = np.array([1.0, 2.0, 3.0])
        result = av_noise(noise, nsft=10)
        expected = np.array([np.median([1, 2, 3])])
        np.testing.assert_array_equal(result, expected)

    def test_av_noise_default_nsft(self):
        """Test with default nsft value of 48"""
        noise = np.ones(96)
        noise[:48] = 1.0
        noise[48:] = 2.0
        result = av_noise(noise, nsft=48)
        expected = np.array([1.0, 2.0])
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
