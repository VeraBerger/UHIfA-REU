import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import math
import pdb
import scipy.interpolate as interpolate
import os.path
from scipy.misc import derivative
from astropy.stats import sigma_clip
from numpy.polynomial import Polynomial
import sympy
from sympy import oo

"""
This script creates a one-to-one correlation between blackbody temperature and GALEX FUV-NUV magnitude. 
We first generate blackbody SEDs for the two wavelength ranges covered by the GALEX near- and far-UV filters
Using response functions for each filter of the GALEX space telescope, we perform synthetic photometry on the SEDs 
This provides us with blackbody temperature as function of far-minus-near UV color
"""

# set plot parameters
mpl.rcParams['axes.linewidth'] = 1.75
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rcParams['agg.path.chunksize'] = 2000
plt.style.use(['science', 'no-latex'])
plt.rc('font',family='serif')
plt.figure(figsize=(12, 8))
plt.minorticks_on()
plt.tick_params(direction='out', length=8, width=1, which='major', labelsize=21, top=True, right=True)
plt.tick_params(direction='out', length=4, width=1, which='minor', labelsize=21, top=True, right=True)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

# global variables
h = 6.626e-34
c = 2.998e+8
k = 1.38e-23
# reference wavelengths from SVO filter profile service 
lambda_ref_f = 1535.08 * 1e-10
lambda_ref_n = 2300.78 * 1e-10

# load GALEX filter response functions
fuv_ftc = np.genfromtxt('/Users/veraberger/reu/GALEX_GALEX.FUV.dat') 
nuv_ftc = np.genfromtxt('/Users/veraberger/reu/GALEX_GALEX.NUV.dat')

def planck(wav, T):
    """
    planck function: gives blackbody intensity for a range of wavelengths
    :param wav: (list) a list of wavelengths (meters)
    :param T: (float) a temperature (Kelvin)
    :return: (list) a list of blackbody intensities (W·sr-1·m-3) for each wavelength
    """
    a = 2.0*h*c**2
    b = h*c/(wav*k*T)
    intensity = a/ ((wav**5) * (np.exp(b) - 1.0))
    return intensity

def synth_phot(intensity_n, intensity_f, nuv_lams, fuv_lams, nuv_trans, fuv_trans):
    """
    perform synthetic photometry by integrating blackbodies (BB) over a range of wavelengths from telescope filters
    based on https://mfouesneau.github.io/pyphot/photometry.html#ab-magnitude-system
    :param intensity_n/f (list): list of BB intensities in near-UV and far-UV, respectively
    :param nuv/fuv_lams (list): list of wavelengths from GALEX NUV/FUV filter response functions
    :param nuv/fuv_lams (list): list of effective areas corresp. to wavelengths from GALEX NUV/FUV filter response functions
    :return flux_fuv, flux_nuv: list of monochromatic fluxes
    """ 
    nuv_top = np.trapz(intensity_n * nuv_trans / nuv_lams, nuv_lams) 
    nuv_bottom = np.trapz(nuv_trans / nuv_lams, nuv_lams)
    flux_nuv = nuv_top / nuv_bottom
    fuv_top = np.trapz(intensity_f * fuv_trans /fuv_lams, fuv_lams)
    fuv_bottom = np.trapz(fuv_trans / fuv_lams, fuv_lams)
    flux_fuv = fuv_top / fuv_bottom
    return flux_fuv, flux_nuv

def f_to_m(flux, wavelength): 
    """
    convert flux to AB magnitude
    again using https://mfouesneau.github.io/pyphot/photometry.html#ab-magnitude-system
    :param flux (list): list of fluxes in the given reference wavelength
    :param wavelength (float): reference wavelength for filter
    return (float): AB magnitudes corresponding to input fluxes
    """
    return -2.5 * np.log10(flux) - 2.5*np.log10(wavelength**2 / c) - 48.60

# generate x-axis for blackbody curve in increments from 1nm to 3 micrometer in 1 nm increments
# starting at 1 nm to avoid lambda = 0, which would result in division by zero
wavelengths = np.arange(1e-9, 3e-6, 1e-9) # meters

# get wavelengths and transmissions from filter response functions
nuv_fc_lams = nuv_ftc[:,0]*1e-10
nuv_fc_trans = nuv_ftc[:,1]
fuv_fc_lams = fuv_ftc[:,0]*1e-10
fuv_fc_trans = fuv_ftc[:,1]

# initialize color and temperature arrays
colorArr = []
tempArr = []

# generate colors spanning a range of temperatures
for t in range(1000, 300000, 20):
    bb_intensity = planck(wavelengths, t)
    nuv_intensity = np.interp(nuv_fc_lams, wavelengths, bb_intensity) # returns bb sed at wavelengths given by filter response func
    fuv_intensity = np.interp(fuv_fc_lams, wavelengths, bb_intensity) 

    # perform synthetic photometry on the SEDs
    flux_f, flux_n = synth_phot(nuv_intensity, fuv_intensity, nuv_fc_lams, fuv_fc_lams, nuv_fc_trans, fuv_fc_trans)
    
    # convert fluxes to magnitudes
    mag_f = f_to_m(flux_f, lambda_ref_f)
    mag_n = f_to_m(flux_n, lambda_ref_n)

    # compute FUV - NUV color
    color = mag_f - mag_n
    colorArr.append(color)
    tempArr.append(t)

# turn lists into arrays
tempArr = np.array(tempArr)
colorArr = np.array(colorArr)

# create interpolating function, returns blackbody temperature given an array of FUV-NUV colors
temp_func = interpolate.interp1d(colorArr, tempArr, bounds_error=False, fill_value='extrapolate')

# plot fuv-nuv color
plt.plot(temp_func(colorArr), colorArr, color='#550527', alpha=0.7, linewidth=2)
plt.xticks([10000, 20000, 30000, 40000, 50000], ['10,000', '20,000', '30,000', '40,000', '50,000'])
plt.ylabel('FUV - NUV [AB mag]', size=24)
plt.ylim(-1,5)
plt.xlim(3000, 60000)
plt.xlabel('Temperature [K]', size=24)
plt.scatter(tempArr, colorArr, s=1, c='#FAA613')
plt.show()
