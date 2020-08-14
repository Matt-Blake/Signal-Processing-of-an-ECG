"""
    FIR.py
    Contains all the FIR filter computation functions for ENEL420-20S2 Assignment 1.

    Authors: Matt Blake   (58979250)
             Reweti Davis (23200856)
             Group Number: 18
    Last Modified: 14/08/2020
"""

# Imported libraries
from scipy.signal import freqz, lfilter, firwin, remez, firwin2, convolve
from scipy.fft import fft
import numpy as np



#
# FIR Filter functions
#
def createWindowFilters(notches, sample_rate, notch_width, num_taps):
    """Compute and return the bandstop  window filter array for the specified notches. Adjusting the window type and band width changes attenuation."""

    window = ('kaiser', 2.5) #Define the window type
    f1, f2 = notches #Seperate the cutoff frequencies specified
    width = notch_width / 2.0 #5 / 2 = 2.5 Hz one sided 3dB bandwidth

    cutoff_1 = [(f1 - width), (f1 + width)] #Window 1
    cutoff_2 = [(f2 - width), (f2 + width)] #Window 2
    cutoff = [cutoff_1[0], cutoff_1[1], cutoff_2[0], cutoff_2[1]] #Joined windows


    filter_1 = firwin(numtaps=num_taps, cutoff=cutoff_1, window=window, fs=sample_rate) #Filter 1
    filter_2 = firwin(numtaps=num_taps, cutoff=cutoff_2, window=window, fs=sample_rate) #Filter 2
    filter_overall =  firwin(numtaps=num_taps, cutoff=cutoff, window=window, fs=sample_rate) #Overall filter
    
    return filter_1, filter_2, filter_overall



def createOptimalFilters(notches, sample_rate, notch_width, num_taps):
    """Compute and return the bandstop  optimal filter arrays for the specified notches. Adjusting the window type and band width changes attenuation."""

    f1, f2 = notches #Seperate the cutoff frequencies for computation
    width = notch_width / 2.0 #5 / 2 = 2.5Hz one sided 3dB bandwidth
    stop = 100000 #Stop band weighting 
    pass_ = 1 #Passband weighting 
    weight = [pass_, stop, pass_] #Isolated filter weighting
    weight_overall = [pass_, stop, pass_, stop, pass_] #Overall filter weighting
    gains = [1, 0, 1]
    gains_overall = [1, 0, 1, 0, 1] #Indicates stop and passband locations in the specified bands
    alpha = 0.001 #Minimal Spacing of stop band notch to allow convergence

    band_1= [0,  f1 - width, f1 - alpha, f1 + alpha, f1 + width, sample_rate / 2] #Pad the stop band as the method doesnt converge well otherwise
    band_2= [0, f2 - width, f2 - alpha, f2 + alpha, f2 + width, sample_rate / 2]
    bands = [0,  f1 - width, f1 - alpha, f1 + alpha, f1 + width, f2 - width, f2 - alpha, f2 + alpha, f2 + width, sample_rate / 2] #Overall filter bands

    filter_1 = remez(numtaps=num_taps, bands=band_1, desired=gains, fs=sample_rate, weight=weight) #Filter 1
    filter_2 = remez(numtaps=num_taps, bands=band_2, desired=gains, fs=sample_rate, weight=weight) #Filter 2
    filter_overall = remez(numtaps=num_taps, bands=bands, desired=gains_overall, fs=sample_rate, weight=weight_overall) #Overall filter
    
    return filter_1, filter_2, filter_overall



def createFreqSamplingFilters(notches, sample_rate, notch_width, num_taps):
    """Compute and return the bandstop frequency sampling filter arrays for the specified notches. Adjusting the window type and band width changes attenuation."""

    # Define and computer frequency sampling filter coefficients
    f1, f2 = notches
    window_type = ('kaiser', 2.5)
    width = notch_width / 2.0 # One sided 3dB bandwidth, in Hz
    alpha = width - 0.01 # Added transition points to narrow the band further
    omega = width - 0.1 

    gains = [1, 1, 0, 0, 0, 0, 0, 1, 1] # The passband/stopband gains for each filter
    gains_overall = [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1] # The passband/stopband gains for the overall filter

    # Calculate array of frequency steps
    freq_1 = [0, f1 - width, f1 - alpha, f1 - omega, f1, f1 + omega, f1 + alpha, f1 + width, sample_rate / 2] # Frequency steps for filter 1
    freq_2 = [0, f2 - width, f2 - alpha, f2 - omega, f2, f2 + omega, f2 + alpha, f2 + width, sample_rate / 2] # Frequency steps for filter 2
    freq = [0, f1 - width, f1 - alpha, f1 - omega, f1, f1 + omega, f1 + alpha, f1 + width, f2 - width, f2 - alpha, f2 - omega, f2, f2 + omega, f2 + alpha, f2 + width, sample_rate / 2] # Frequency steps the overall filter

    # Create filters
    filter_1 = firwin2(numtaps=num_taps, freq=freq_1, gain=gains, fs=sample_rate, window=window_type) # Create filter 1
    filter_2 = firwin2(numtaps=num_taps, freq=freq_2, gain=gains, fs=sample_rate, window=window_type) # Create fitler 2
    filter_overall = firwin2(numtaps=num_taps, freq=freq, gain=gains_overall, fs=sample_rate, window=window_type) # Create overall filter
    
    return filter_1, filter_2, filter_overall



def applyFIRFilters(filter_1, filter_2, filter_overall, samples):
    """Pass data through two cascaded FIR filters, and a single overall filter and return the result after each filter"""

    half_filtered = lfilter(filter_1, 1, samples)
    full_filtered = lfilter(filter_2, 1, half_filtered)
    overall_filtered = lfilter(filter_overall, 1, samples)

    return half_filtered, full_filtered, overall_filtered


