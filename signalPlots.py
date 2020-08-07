"""
    signalPlots.py
    Contains all the plotting functions for ENEL420-S2 Assignment 1,
    including supporting functions.

    Authors: Matt Blake   (58979250),
             Reweti Davis (23200856).
    Last Modified: 07/08/2020
"""

from scipy.signal import freqz, lfilter, firwin, remez, firwin2, convolve
from scipy.fft import fft
import matplotlib.pyplot as plt
import numpy as np


#
# Unfiltered Plots
#
def plotECG(samples, time):
    """Plot a time domain graph of the ECG data"""

    plt.figure()
    plt.plot(time, samples)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV)")
    plt.suptitle("Time domain ECG signal")
    plt.xlim(time[0], time[-1]) # Limit the x axis to locations with data points



def plotECGSpectrum(frequency, frequency_data):
    """Calculate and plot the frequency spectrum of the ECG"""

    plt.figure()
    plt.plot(frequency, 20 * np.log10(abs(frequency_data)))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.suptitle("Frequency Spectrum of the ECG signal")
    plt.xlim(frequency[0], frequency[-1]/2) # Limit the x axis to locations with data points



#
# IIR Notch Plots
#
def plotIIRNotchECG(samples, time):
    """Plot a time domain graph of the IIR notch filtered ECG data"""

    plt.figure()
    plt.plot(time, samples)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV)")
    plt.suptitle("Time domain IIR Notch Filtered ECG signal")
    plt.xlim(time[0], time[-1]) # Limit the x axis to locations with data points



def plotIIRNotchECGSpectrum(notch_frequency, notch_freq_data):
    """Calculate and plot the frequency spectrum of the ECG after filtering with an IIR notch filter"""

    plt.figure()
    plt.plot(notch_frequency, 20 * np.log10(abs(notch_freq_data)))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.suptitle("Frequency Spectrum of the IIR Notch Filtered ECG signal")
    plt.xlim(notch_frequency[0], notch_frequency[-1]/2)  # Limit the x axis to locations with data points



def plotIIRNotchFilterResponse(numuerator, denominator):
    """Plot the frequency response of the window filter"""

    w, h = freqz(numuerator, denominator) # Calculate the frequency response
    w_scaled = 1024 * w / (2 * np.pi)
    plt.figure()
    plt.plot(w_scaled, 20 * np.log10(abs(h))) #Plot in dB vs Hz
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.suptitle("Frequency Spectrum of the Notch Filtered ECG Signal")
    plt.xlim(w_scaled[0], w_scaled[-1]) # Limit the x axis from 0 to Nyquist frequency


#
# Window Filtered Plots
#
def plotWindowedECG(samples, time):
    """Plot a time domain graph of the window filtered ECG data"""

    plt.figure()
    plt.plot(time, samples)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV)")
    plt.suptitle("Time Domain Window Filtered ECG Signal")
    plt.xlim(time[0], time[-1]) # Limit the x axis to locations with data points



def plotWindowedECGSpectrum(frequency, frequency_data):
    """Calculate and plot the indow filtered ECG frequency spectrum"""

    plt.figure()
    plt.plot(frequency, 20 * np.log10(abs(frequency_data)))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.suptitle("Frequency Spectrum of the Window Filtered ECG Signal")
    plt.xlim(frequency[0], frequency[-1]/2) # Limit the x axis from 0 to Nyquist frequency



def plotWindowFilterResponse(filter_array):
    """Plot the frequency response of the window filter"""

    w, h = freqz(filter_array) # Calculate the frequency response
    w_scaled = 1024 * w / (2 * np.pi)
    plt.figure()
    plt.plot(w_scaled, 20 * np.log10(abs(h))) #Plot in dB vs Hz
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.suptitle("Window Filter Frequency Response")
    plt.xlim(w_scaled[0], w_scaled[-1]) # Limit the x axis from 0 to nyquist frequency