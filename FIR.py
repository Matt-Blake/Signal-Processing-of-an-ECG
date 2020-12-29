"""
    FIR.py (Signal Processing of an ECG program)
    Contains all the FIR filter computation functions for
    the Signal Processing of an ECG program.

    Authors: Matt Blake   (58979250)
             Reweti Davis (23200856)
    Last Modified: 27/12/2020
"""

# Imported libraries
from scipy.signal import freqz, lfilter, firwin, remez, firwin2, convolve
from scipy.fft import fft
import numpy as np
from config import *

# Define constants
FIR_NUMERATOR = 1 # Numerator of FIR filter transfer function
TO_ONE_SIDED_WIDTH = 2 # Used to half bandwidths


# Functions
def createWindowFilters(notches:list, sample_rate:float, notch_width:float, num_taps:int) -> tuple:
    """Compute and return the bandstop  window filter array for the specified notches.
    Adjusting the window type and band width changes attenuation."""

    # Define and compute window filter coefficients
    f1, f2 = notches # Separate the cutoff frequencies specified
    width = notch_width / TO_ONE_SIDED_WIDTH # One sided 3dB bandwidth
    window_used = (FIR_WINDOW_TYPE, width) # Define the window

    # Create an array of frequency steps
    cutoff_1 = [(f1 - width), (f1 + width)] # Window 1
    cutoff_2 = [(f2 - width), (f2 + width)] # Window 2
    cutoff = [cutoff_1[0], cutoff_1[1], cutoff_2[0], cutoff_2[1]] # Joined windows

    # Create filters
    filter_1 = firwin(numtaps=NUM_FIR_TAPS, cutoff=cutoff_1, window=window_used, fs=sample_rate) # Filter 1
    filter_2 = firwin(numtaps=NUM_FIR_TAPS, cutoff=cutoff_2, window=window_used, fs=sample_rate) # Filter 2
    filter_overall =  firwin(numtaps=NUM_FIR_TAPS, cutoff=cutoff, window=window_used, fs=sample_rate) # Overall filter
    
    return filter_1, filter_2, filter_overall



def createOptimalFilters(notches:list, sample_rate:float, notch_width:float, num_taps:int) -> tuple:
    """Compute and return the bandstop  optimal filter arrays for the specified notches.
    Adjusting the window type and band width changes attenuation. A filter is made for each
    stopband, as well as a combined multi-stopband filter"""

    # Define and compute optimal filter coefficients
    f1, f2 = notches # Separate the cutoff frequencies for computation
    width = notch_width / TO_ONE_SIDED_WIDTH # One sided 3dB bandwidth 

    # Create an array of frequency steps
    band_1= [0,  f1 - width, f1 - OPTIMAL_NOTCH_SPACING, f1 + OPTIMAL_NOTCH_SPACING, f1 + width, sample_rate / TO_ONE_SIDED_WIDTH]
    band_2= [0, f2 - width, f2 - OPTIMAL_NOTCH_SPACING, f2 + OPTIMAL_NOTCH_SPACING, f2 + width, sample_rate / TO_ONE_SIDED_WIDTH]
    bands_overall = [0,  f1 - width, f1 - OPTIMAL_NOTCH_SPACING, f1 + OPTIMAL_NOTCH_SPACING,
                    f1 + width, f2 - width, f2 - OPTIMAL_NOTCH_SPACING, f2 + OPTIMAL_NOTCH_SPACING, f2 + width, sample_rate / 2]
    
    # Create filters
    filter_1 = remez(numtaps=NUM_FIR_TAPS, bands=band_1, desired=OPTIMAL_GAINS_SINGLE, fs=sample_rate, weight=OPTIMAL_WEIGHTS_SINGLE) # Filter 1
    filter_2 = remez(numtaps=NUM_FIR_TAPS, bands=band_2, desired=OPTIMAL_GAINS_SINGLE, fs=sample_rate, weight=OPTIMAL_WEIGHTS_SINGLE) # Filter 2
    filter_overall = remez(numtaps=NUM_FIR_TAPS, bands=bands_overall, desired=OPTIMAL_GAINS_OVERALL, fs=sample_rate, weight=OPTIMAL_WEIGHTS_OVERALL) # Overall filter
    
    return filter_1, filter_2, filter_overall



def createFreqSamplingFilters(notches:list, sample_rate:float, notch_width:float, num_taps:int) -> tuple:
    """Compute and return the bandstop frequency sampling filter arrays for the specified notches.
    Adjusting the window type and band width changes attenuation."""

    # Define and compute frequency sampling filter coefficients
    f1, f2 = notches
    width = notch_width / TO_ONE_SIDED_WIDTH # One sided 3dB bandwidth, in Hz
    omega = width - FREQ_SAMP_WIDTH 
    alpha = width - FREQ_SAMP_TRANSITION_WIDTH
    window_type = (FIR_WINDOW_TYPE, width) # Define the window

    # Calculate array of frequency steps
    freq_1 = [0, f1 - width, f1 - alpha, f1 - omega, f1, f1 + omega, f1 + alpha, f1 + width, sample_rate / TO_ONE_SIDED_WIDTH] # Frequency steps for filter 1
    freq_2 = [0, f2 - width, f2 - alpha, f2 - omega, f2, f2 + omega, f2 + alpha, f2 + width, sample_rate / TO_ONE_SIDED_WIDTH] # Frequency steps for filter 2
    freq = [0, f1 - width, f1 - alpha, f1 - omega, f1, f1 + omega, f1 + alpha, f1 + width, f2 - width, f2 - alpha, f2 - omega, f2, f2 + omega, f2 + alpha, f2 + width, sample_rate / 2] # Frequency steps the overall filter

    # Create filters
    filter_1 = firwin2(numtaps=NUM_FIR_TAPS, freq=freq_1, gain=FREQ_SAMP_GAINS_SINGLE, fs=sample_rate, window=window_type) # Create filter 1
    filter_2 = firwin2(numtaps=NUM_FIR_TAPS, freq=freq_2, gain=FREQ_SAMP_GAINS_SINGLE, fs=sample_rate, window=window_type) # Create filter 2
    filter_overall = firwin2(numtaps=NUM_FIR_TAPS, freq=freq, gain=FREQ_SAMP_GAINS_OVERALL, fs=sample_rate, window=window_type) # Create overall filter
    
    return filter_1, filter_2, filter_overall



def applyFIRFilters(filter_1:list, filter_2:list, filter_overall:list, samples:list) -> tuple:
    """Pass data through two cascaded FIR filters, and a single overall filter
    and return the result after each filter"""

    half_filtered = lfilter(filter_1, FIR_NUMERATOR, samples) # Pass signal through the first filter
    full_filtered = lfilter(filter_2, FIR_NUMERATOR, half_filtered) # Then pass signal through the second filter
    overall_filtered = lfilter(filter_overall, FIR_NUMERATOR, samples) # Pass original signal through the combined filter

    return half_filtered, full_filtered, overall_filtered


