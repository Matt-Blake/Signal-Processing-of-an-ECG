"""
    signalPlots.py
    Contains all the plotting functions for ENEL420-20S2 Assignment 1,
    including supporting functions.

    Authors: Matt Blake   (58979250)
             Reweti Davis (23200856)
             Group Number: 18
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

    ECG = plt.figure()
    plt.plot(time, samples)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV)")
    plt.suptitle("Time domain ECG signal")
    plt.xlim(time[0], time[-1]) # Limit the x axis to locations with data points

    return ECG



def plotECGSpectrum(frequency, frequency_data):
    """Calculate and plot the frequency spectrum of the ECG"""

    ECGSpectrum = plt.figure()
    plt.plot(frequency, 20 * np.log10(abs(frequency_data)))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.suptitle("Frequency Spectrum of the ECG signal")
    plt.xlim(frequency[0], frequency[-1]/2) # Limit the x axis to locations with data points

    return ECGSpectrum



#
# IIR Notch Plots
#
def plotIIRNotchECG(samples, time):
    """Plot a time domain graph of the IIR notch filtered ECG data"""

    IIRNotchECG = plt.figure()
    plt.plot(time, samples)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV)")
    plt.suptitle("Time domain IIR Notch Filtered ECG signal")
    plt.xlim(time[0], time[-1]) # Limit the x axis to locations with data points

    return IIRNotchECG



def plotIIRNotchECGSpectrum(notch_frequency, notch_freq_data):
    """Calculate and plot the frequency spectrum of the ECG after filtering with an IIR notch filter"""

    IIRNotchECGSpectrum = plt.figure()
    plt.plot(notch_frequency, 20 * np.log10(abs(notch_freq_data)))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.suptitle("Frequency Spectrum of the IIR Notch Filtered ECG signal")
    plt.xlim(notch_frequency[0], notch_frequency[-1]/2)  # Limit the x axis to locations with data points

    return IIRNotchECGSpectrum



def plotIIRNotchFilterResponse(numuerator, denominator, f_samp):
    """Plot and return the frequency response (magnitude and phase) of the IIR notch filter"""

    # Calculate the frequency response
    freq, response = freqz(numuerator, denominator, fs=f_samp)

    # Create plot
    IIRNotchFilterResponse, (IIR_ax1, IIR_ax2) = plt.subplots(2, 1)
    plt.suptitle("IIR Notch Filter Frequency Response")
    plt.xlabel("Frequency (Hz)")

    # Plot magnitude response
    IIR_ax1.plot(freq, 20 * np.log10(abs(response))) # Plot magnitude in dB vs Hz
    IIR_ax1.set_ylabel("Amplitude (dB)")
    IIR_ax1.set_xlim(freq[0], freq[-1])  # Limit the x axis from 0 to Nyquist frequency

    # Plot phase response
    IIR_ax2.plot(freq, np.angle(response, deg=True)) # Plot phase in dB vs degrees
    IIR_ax2.set_ylabel("Phase (°)")
    IIR_ax2.set_xlim(freq[0], freq[-1])  # Limit the x axis from 0 to Nyquist frequency

    return IIRNotchFilterResponse



#
# Window Filtered Plots
#
def plotWindowedECG(samples, time):
    """Plot a time domain graph of the window filtered ECG data"""

    WindowedECG = plt.figure()
    plt.plot(time, samples)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV)")
    plt.suptitle("Time Domain Window Filtered ECG Signal")
    plt.xlim(time[0], time[-1]) # Limit the x axis to locations with data points

    return WindowedECG



def plotWindowedECGSpectrum(frequency, frequency_data):
    """Calculate and plot the window filtered ECG frequency spectrum"""

    WindowedECGSpectrum = plt.figure()
    plt.plot(frequency, 20 * np.log10(abs(frequency_data)))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.suptitle("Frequency Spectrum of the Window Filtered ECG Signal")
    plt.xlim(frequency[0], frequency[-1]/2) # Limit the x axis from 0 to Nyquist frequency

    return WindowedECGSpectrum



def plotWindowFilterResponse(filter_array, f_samp):
     """Plot and return the frequency response (magnitude and phase) of the window filter"""

     # Calculate the frequency response
     freq, response = freqz(filter_array, fs=f_samp)

     # Create plot
     WindowFilterResponse, (window_ax1, window_ax2) = plt.subplots(2, 1)
     plt.suptitle("Window Filter Frequency Response")
     plt.xlabel("Frequency (Hz)")

     # Plot magnitude response
     window_ax1.plot(freq, 20 * np.log10(abs(response)))  # Plot magnitude in dB vs Hz
     window_ax1.set_ylabel("Amplitude (dB)")
     window_ax1.set_xlim(freq[0], freq[-1])  # Limit the x axis from 0 to Nyquist frequency

     # Plot phase response
     window_ax2.plot(freq, np.angle(response, deg=True))  # Plot phase in dB vs degrees
     window_ax2.set_ylabel("Phase (°)")
     window_ax2.set_xlim(freq[0], freq[-1])  # Limit the x axis from 0 to Nyquist frequency

     return WindowFilterResponse