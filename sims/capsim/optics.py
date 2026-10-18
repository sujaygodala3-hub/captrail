"""Optical forward model:  I = S[ h * (L * I_ideal) ] + n   (proposal, section 13).

Deliberately simple: Gaussian PSF, linear illumination gradient, area-integrating
sensor, Poisson + read noise, 8-bit quantization. Not a tissue-optics model.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def forward(mask, res_um, *, sigma_um=2.0, contrast=0.35, illum_strength=0.0,
            illum_angle_deg=0.0, pixel_um=1.0, photons=None, read_noise=0.0,
            exposure=0.8, bits=8, rng=None, texture_amp=0.0, texture_corr_um=25.0,
            shift_um=(0.0, 0.0)):
    """Simulate a detector image (float in [0,1]) from a fine-grid lumen mask.

    sigma_um        Gaussian PSF sigma (blur) in object space
    contrast        fractional intensity drop inside the lumen
    illum_strength  peak-to-peak relative illumination variation across the field
    pixel_um        detector pixel size in object space (must be a multiple of res_um)
    photons         expected photons per pixel at intensity 1 (None = no shot noise)
    read_noise      Gaussian noise std in intensity units
    texture_amp     skin texture: band-limited multiplicative noise, fractional std (ASSUMED, sweep it)
    texture_corr_um correlation length of that texture
    shift_um        (dx, dy) sub-pixel position jitter of the scene before imaging (S7 re-seating)
    """
    rng = rng or np.random.default_rng(0)
    ideal = 1.0 - contrast * mask.astype(np.float32)
    ny, nx = mask.shape
    if texture_amp:
        kt = max(1, int(round(min(4.0, texture_corr_um / 4) / res_um)))      # build on a coarse grid, interpolate up
        small = ndi.gaussian_filter(rng.normal(size=(ny // kt + 2, nx // kt + 2)), texture_corr_um / (res_um * kt), mode="wrap")
        tex = ndi.zoom(small, kt, order=1)[:ny, :nx]
        tex = ((tex - tex.mean()) / tex.std()).astype(np.float32)
        ideal = ideal * (1.0 + texture_amp * tex)
    if shift_um != (0.0, 0.0):
        ideal = ndi.shift(ideal, (shift_um[1] / res_um, shift_um[0] / res_um), order=1, mode="nearest")
    if illum_strength:
        yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
        a = np.deg2rad(illum_angle_deg)
        g = (xx - nx / 2) * np.cos(a) + (yy - ny / 2) * np.sin(a)
        g = g / (np.abs(g).max() * 2.0)           # in [-0.5, 0.5]
        ideal = ideal * (1.0 + illum_strength * g)
    if sigma_um > 0:
        ideal = ndi.gaussian_filter(ideal, sigma_um / res_um, mode="nearest")
    f = int(round(pixel_um / res_um))
    if abs(f * res_um - pixel_um) > 1e-6:
        raise ValueError("pixel_um must be an integer multiple of res_um")
    ny2, nx2 = (ny // f) * f, (nx // f) * f
    img = ideal[:ny2, :nx2].reshape(ny2 // f, f, nx2 // f, f).mean(axis=(1, 3)) * exposure
    if photons:
        img = rng.poisson(np.clip(img, 0, None) * photons) / photons
    if read_noise:
        img = img + rng.normal(0.0, read_noise, img.shape)
    q = 2 ** bits - 1
    return np.round(np.clip(img, 0, 1) * q) / q


def depth_model(contrast0, sigma0_um, depth_um, mu_per_um, k):
    """S3 phenomenological depth model (NOT literature values; fit mu and k to real captures).

    contrast = c0 * exp(-mu * depth);  sigma = sqrt(sigma0^2 + (k * depth)^2).
    Immersion oil lowers mu (and k). Returns (contrast, sigma_um).
    """
    return contrast0 * np.exp(-mu_per_um * depth_um), float(np.hypot(sigma0_um, k * depth_um))
