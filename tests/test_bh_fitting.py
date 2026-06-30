import os
import numpy as np
from astropy.io import fits
import sys
sys.path.insert(0, '/home/david/Coding/sdgft-fits-bh')
from fit_eht_shadow import (
    calculate_theoretical_shadow_uas,
    get_radial_profile_uas,
    gaussian_ring,
    SDGFT_SHADOW
)

def test_theoretical_shadow():
    # Test M87* theoretical shadow diameter in GR and SDGFT
    # M87* mass = 6.5e9, dist = 16.8 Mpc (16.8e3 kpc)
    diam_gr = calculate_theoretical_shadow_uas(6.5e9, 16.8e3, scale_factor=1.0)
    diam_sdgft = calculate_theoretical_shadow_uas(6.5e9, 16.8e3, scale_factor=SDGFT_SHADOW)
    
    assert 38.0 < diam_gr < 41.0
    assert 41.5 < diam_sdgft < 45.0
    assert np.isclose(diam_sdgft / diam_gr, SDGFT_SHADOW)

def test_gaussian_ring_function():
    # Test our Gaussian ring model output
    r = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    I_vals = gaussian_ring(r, I_peak=10.0, r_peak=20.0, sigma=5.0, bg=0.1)
    
    assert I_vals[2] == 10.1  # Peak + bg
    assert I_vals[0] < 1.0     # Far from peak
    assert np.all(I_vals >= 0.1)

def test_fits_profile_and_scale():
    # Create a tiny mock fits file to verify loading and CDELT conversion
    size = 32
    data = np.zeros((size, size))
    # Make a ring of radius 10 pixels
    y, x = np.indices((size, size))
    r = np.hypot(x - size//2, y - size//2)
    data = np.exp(-0.5 * ((r - 10) / 2.0)**2)
    
    hdu = fits.PrimaryHDU(data.astype(np.float32))
    hdr = hdu.header
    hdr["CDELT2"] = 0.5 / (3600.0 * 1e6)  # 0.5 uas/pixel
    
    # Compute profile
    r_uas, I_val = get_radial_profile_uas(data, hdr)
    
    # The radial grid should have pixel scale of 0.5 uas
    assert np.isclose(r_uas[1] - r_uas[0], 0.5)
    # The peak of intensity should be around 10 pixels * 0.5 uas/pixel = 5.0 uas
    peak_idx = np.argmax(I_val)
    assert np.isclose(r_uas[peak_idx], 5.0)
