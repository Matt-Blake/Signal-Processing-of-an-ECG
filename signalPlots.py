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

    # Plot ECG
    plt.figure()
    plt.plot(time, samples)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV)")
    plt.suptitle("Time domain ECG signal")
    plt.xlim(time[0], time[-1]) # Limit the x axis to locations with data points



def plotECGSpectrum(frequency, frequency_data):
    """Calculate and plot the frequency spectrum of the ECG"""

    # Plot ECG Spectrum
    plt.figure()
    plt.plot(frequency, 20 * np.log10(abs(frequency_data)))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.suptitle("Frequency Spectrum of the ECG signal")
    plt.xlim(frequency[0], frequency[-1] / 2) # Limit the x axis to locations with data points



#
# IIR Notch Plots
#
def plotNotchedECG(samples, time):
    """Plot a time domain graph of the ECG data"""

    # Plot ECG
    plt.figure()
    plt.plot(time, samples)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV)")
    plt.suptitle("Time domain notch filtered ECG signal")
    plt.xlim(time[0], time[-1]) # Limit the x axis to locations with data points



def plotIIRNotchFilterResponse(numuerator, denominator):
    """Plot the frequency response of the window filter."""

    w, h = freqz(numuerator, denominator)
    w_scaled = 1024 * w / (2 * np.pi)
    plt.figure()
    plt.plot(w_scaled, 20 * np.log10(abs(h))) #Plot in dB vs Hz
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.suptitle("Frequency Spectrum of the notch Filtered ECG Signal")
    plt.xlim(w_scaled[0], w_scaled[-1]) # Limit the x axis from 0 to nyquist frequency


#
# Window Filtered Plots
#
def plotWindowedECG(samples, time):
    """Plot the ECG which has filtered by a window filter."""

    # Plot Windowed ECG
    plt.figure()
    plt.plot(time, samples)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV)")
    plt.suptitle("Time Domain Window Filtered ECG Signal")
    plt.xlim(time[0], time[-1]) # Limit the x axis to locations with data points



def plotWindowedECGSpectrum(frequency, frequency_data):
    """Calculate and plot the ECG frequency spectrum."""

    # Plot window filtered ECG Spectrum
    plt.figure()
    plt.plot(frequency, 20 * np.log10(abs(frequency_data)))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.suptitle("Frequency Spectrum of the Window Filtered ECG Signal")
    plt.xlim(frequency[0], frequency[-1]/2) # Limit the x axis from 0 to nyquist frequency



def plotWindowFilterResponse(filter_array):
    """Plot the frequency response of the window filter."""
    w, h = freqz(filter_array)
    w_scaled = 1024 * w / (2 * np.pi)
    plt.figure()
    plt.plot(w_scaled, 20 * np.log10(abs(h))) #Plot in dB vs Hz
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.suptitle("Window Filter Frequency Response")
    plt.xlim(w_scaled[0], w_scaled[-1]) # Limit the x axis from 0 to nyquist frequency

