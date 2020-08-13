"""
    signalPlots.py
    Contains all the plotting functions for ENEL420-20S2 Assignment 1,
    including supporting functions.

    Authors: Matt Blake   (58979250)
             Reweti Davis (23200856)
             Group Number: 18
    Last Modified: 07/08/2020
"""

# Imported libraries
from scipy.signal import freqz, lfilter, firwin, remez, firwin2, convolve
from scipy.fft import fft
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Global variables
TIME_LINE_WIDTH = 0.5 # The linewidth of the time domain plots
SPECTRUM_LINE_WIDTH = 0.2 # The linewidth of the signal spectrum plots
FILTER_LINE_WIDTH = 1.7 # The linewidth of filter response plots


# Functions

#
# Unfiltered Plots
#
def plotECG(samples, time):
    """Plot a time domain graph of the ECG data"""

    ECG = plt.figure()
    plt.plot(time, samples, linewidth=TIME_LINE_WIDTH)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.suptitle("Time domain ECG signal")
    plt.xlim(time[0], time[-1]) # Limit the x axis to locations with data points

    return ECG



def plotECGSpectrum(frequency, frequency_data):
    """Calculate and plot the frequency spectrum of the ECG"""

    ECGSpectrum = plt.figure()
    plt.plot(frequency, 20 * np.log10(abs(frequency_data)), linewidth=SPECTRUM_LINE_WIDTH)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.suptitle("Frequency Spectrum of the ECG signal")
    plt.xlim(frequency[0], frequency[-1]/2) # Limit the x axis to locations with data points

    return ECGSpectrum



#
# IIR Notch Plots
#
def plotIIRPoleZero(cutoffs, notch_width, f_samp):
    """Plot a pole-zero diagram of an IIR notch filter"""

    # Define unit circle parameters
    circle_centre = (0, 0)  # The x and y coordinates for the unit circle to be centre at
    circle_radius = 1  # The unit circle has a radius of 1 by definition

    # Create figure
    circle_fig, axis = plt.subplots(figsize=(8, 8))  # Create plot
    plt.xlim([circle_centre[0] - 1, circle_centre[0] + 1])
    plt.ylim([circle_centre[1] - 1, circle_centre[1] + 1])

    # Define pole/zero magnitudes
    zeros_magnitude = 1  # Place the zeros on the unit circle for maximum attenuation
    poles_magnitude = 1 - np.pi * (notch_width/f_samp)  # Calculate the optimal magnitude for the pole pairs

    # Create real and imaginary axis on graph
    real_axis = plt.Line2D([circle_centre[0] - 2, circle_centre[0] + 2], [0, 0], color='black')
    imag_axis = plt.Line2D([0, 0], [[circle_centre[1] - 2, circle_centre[1] + 2]], color='black')

    # Plot unit circle
    circle = plt.Circle(circle_centre, circle_radius, color ='black', fill=False, label='Unit circle') # Create circle
    axis = circle_fig.gca() # Create plot axis
    axis.add_artist(circle) # Add unit circle to figure
    axis.add_artist(real_axis)
    axis.add_artist(imag_axis)

    # Plot poles and zeros
    for cutoff in cutoffs: # Iterate through each cutoff frequency

        # Calculate position of poles/zeros
        angle = 2 * np.pi * cutoff/f_samp # Calculate the pole/zero angle for cutoff frequency
        zero_x_postion = zeros_magnitude * np.cos(angle) # Calculate zero position in real (x) axis
        zero_y_postion = zeros_magnitude * np.sin(angle) # Calculate zero position in imaginary (y) axis
        pole_x_postion = poles_magnitude * np.cos(angle) # Calculate pole position in real (x) axis
        pole_y_postion = poles_magnitude * np.sin(angle) # Calculate pole position in imaginary (y) axis

        # Plot lines between the origin and poles/zeros
        zero_line = plt.Line2D([circle_centre[0], zero_x_postion], [circle_centre[1], zero_y_postion], color='grey', linestyle='--')
        conjugate_zero_line = plt.Line2D([circle_centre[0], zero_x_postion], [circle_centre[1], -zero_y_postion], color='grey', linestyle='--')
        axis.add_artist(zero_line)
        axis.add_artist(conjugate_zero_line)

        # Plot conjugate pairs of poles and zeros
        axis.plot([pole_x_postion], [pole_y_postion], marker='x', markersize=8, markeredgewidth=2, color='red', label='pole')
        axis.plot([pole_x_postion], [-pole_y_postion], marker='x', markersize=8, markeredgewidth=2, color='red', label='pole')
        axis.plot([zero_x_postion], [zero_y_postion], marker='o', color='blue', label='zero')
        axis.plot([zero_x_postion], [-zero_y_postion], marker='o', color='blue', label='zero')

    # Label figure
    plt.xlabel('Real{Z}')
    plt.ylabel('Imag{Z}')

    # Create legend
    zero_patch = mpatches.Patch(color='blue', hatch='o', label='Zeros')
    pole_patch = mpatches.Patch(color='red', hatch='x', label='Poles')
    plt.legend(handles=[zero_patch, pole_patch, circle], loc='upper left')

    return circle_fig



def plotIIRNotchECG(samples, time):
    """Plot a time domain graph of the IIR notch filtered ECG data"""

    IIRNotchECG = plt.figure()
    plt.plot(time, samples, linewidth=TIME_LINE_WIDTH)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.suptitle("Time domain IIR Notch Filtered ECG signal")
    plt.xlim(time[0], time[-1]) # Limit the x axis to locations with data points

    return IIRNotchECG



def plotIIRNotchECGSpectrum(notch_frequency, notch_freq_data):
    """Calculate and plot the frequency spectrum of the ECG after filtering with an IIR notch filter"""

    IIRNotchECGSpectrum = plt.figure()
    plt.plot(notch_frequency, 20 * np.log(abs(notch_freq_data)), linewidth=SPECTRUM_LINE_WIDTH)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.suptitle("Frequency Spectrum of the IIR Notch Filtered ECG signal")
    plt.xlim(notch_frequency[0], notch_frequency[-1]/2)  # Limit the x axis to locations with data points

    return IIRNotchECGSpectrum



def plotIIRNotchFilterResponse(numerator, denominator, f_samp):
    """Plot and return the frequency response (magnitude and phase) of the IIR notch filter"""

    # Calculate the frequency response
    freq, response = freqz(numerator, denominator, fs=f_samp)

    # Create plot
    IIRNotchFilterResponse, (IIR_ax1, IIR_ax2) = plt.subplots(2, 1)
    plt.suptitle("IIR Notch Filter Frequency Response")
    plt.xlabel("Frequency (Hz)")

    # Plot magnitude response
    IIR_ax1.plot(freq, 20 * np.log(abs(response))) # Plot magnitude in dB vs Hz
    IIR_ax1.set_ylabel("Amplitude (dB)")
    IIR_ax1.set_xlim(freq[0], freq[-1])  # Limit the x axis from 0 to Nyquist frequency

    # Plot phase response
    IIR_ax2.plot(freq, np.angle(response, deg=True), linewidth=FILTER_LINE_WIDTH) # Plot phase in dB vs degrees
    IIR_ax2.set_ylabel("Phase (°)")
    IIR_ax2.set_xlim(freq[0], freq[-1])  # Limit the x axis from 0 to Nyquist frequency

    return IIRNotchFilterResponse



#
# Window Filtered Plots
#
def plotWindowedECG(samples, time):
    """Plot a time domain graph of the window filtered ECG data"""

    WindowedECG = plt.figure()
    plt.plot(time, samples, linewidth=TIME_LINE_WIDTH)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.suptitle("Time Domain Window Filtered ECG Signal")
    plt.xlim(time[0], time[-1]) # Limit the x axis to locations with data points

    return WindowedECG



def plotWindowedECGSpectrum(frequency, frequency_data):
    """Calculate and plot the window filtered ECG frequency spectrum"""

    WindowedECGSpectrum = plt.figure()
    plt.plot(frequency, 20 * np.log10(abs(frequency_data)), linewidth=SPECTRUM_LINE_WIDTH)
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
     window_ax1.plot(freq, 20 * np.log10(abs(response)), linewidth=FILTER_LINE_WIDTH)  # Plot magnitude in dB vs Hz
     window_ax1.set_ylabel("Amplitude (dB)")
     window_ax1.set_xlim(freq[0], freq[-1])  # Limit the x axis from 0 to Nyquist frequency

     # Plot phase response
     window_ax2.plot(freq, np.unwrap(np.angle(response, deg=True)), linewidth=FILTER_LINE_WIDTH)  # Plot phase in dB vs degrees
     window_ax2.set_ylabel("Phase (°)")
     window_ax2.set_xlim(freq[0], freq[-1])  # Limit the x axis from 0 to Nyquist frequency

     return WindowFilterResponse


#
#Optimal filter plots
#
def plotOptimalECG(samples, time):
    """Plot a time domain graph of the window filtered ECG data"""

    OptimalECG = plt.figure()
    plt.plot(time, samples, linewidth=TIME_LINE_WIDTH)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.suptitle("Time Domain Optimal Filtered ECG Signal")
    plt.xlim(time[0], time[-1]) # Limit the x axis to locations with data points

    return OptimalECG



def plotOptimalECGSpectrum(frequency, frequency_data):
    """Calculate and plot the window filtered ECG frequency spectrum"""

    OptimalECGSpectrum = plt.figure()
    plt.plot(frequency, 20 * np.log10(abs(frequency_data)), linewidth=SPECTRUM_LINE_WIDTH)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.suptitle("Frequency Spectrum of the Optimal Filtered ECG Signal")
    plt.xlim(frequency[0], frequency[-1]/2) # Limit the x axis from 0 to Nyquist frequency

    return OptimalECGSpectrum



def plotOptimalFilterResponse(filter_array, f_samp):
     """Plot and return the frequency response (magnitude and phase) of the window filter"""

     # Calculate the frequency response
     freq, response = freqz(filter_array, fs=f_samp)

     # Create plot
     OptimalFilterResponse, (optimal_ax1, optimal_ax2) = plt.subplots(2, 1)
     plt.suptitle("Optimal Filter Frequency Response")
     plt.xlabel("Frequency (Hz)")

     # Plot magnitude response
     optimal_ax1.plot(freq, 20 * np.log10(abs(response)), linewidth=FILTER_LINE_WIDTH)  # Plot magnitude in dB vs Hz
     optimal_ax1.set_ylabel("Amplitude (dB)")
     optimal_ax1.set_xlim(freq[0], freq[-1])  # Limit the x axis from 0 to Nyquist frequency

     # Plot phase response
     optimal_ax2.plot(freq, np.unwrap(np.angle(response, deg=True)), linewidth=FILTER_LINE_WIDTH)  # Plot phase in dB vs degrees
     optimal_ax2.set_ylabel("Phase (°)")
     optimal_ax2.set_xlim(freq[0], freq[-1])  # Limit the x axis from 0 to Nyquist frequency

     return OptimalFilterResponse



#
#Frequency sampling filter plots
#
def plotFrequencySampledECG(samples, time):
    """Plot a time domain graph of the Frequency Sampling filtered ECG data"""

    freqSampledECG = plt.figure()
    plt.plot(time, samples, linewidth=TIME_LINE_WIDTH)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.suptitle("Time Domain Frequency Sampling Filtered ECG Signal")
    plt.xlim(time[0], time[-1]) # Limit the x axis to locations with data points

    return freqSampledECG



def plotFrequencySampledECGSpectrum(frequency, frequency_data):
    """Calculate and plot the Frequency Sampling filtered ECG frequency spectrum"""

    FreqECGSpectrum = plt.figure()
    plt.plot(frequency, 20 * np.log10(abs(frequency_data)), linewidth=SPECTRUM_LINE_WIDTH)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.suptitle("Frequency Spectrum of the Frequency Sampling Filtered ECG Signal")
    plt.xlim(frequency[0], frequency[-1]/2) # Limit the x axis from 0 to Nyquist frequency

    return FreqECGSpectrum



def plotFrequencySampledFilterResponse(filter_array, f_samp):
     """Plot and return the frequency response (magnitude and phase) of the Frequency Sampling filter"""

     # Calculate the frequency response
     freq, response = freqz(filter_array, fs=f_samp)

     # Create plot
     FreqFilterResponse, (freq_ax1, freq_ax2) = plt.subplots(2, 1)
     plt.suptitle("Frequency Sampling Filter Frequency Response")
     plt.xlabel("Frequency (Hz)")

     # Plot magnitude response
     freq_ax1.plot(freq, 20 * np.log10(abs(response)), linewidth=FILTER_LINE_WIDTH)  # Plot magnitude in dB vs Hz
     freq_ax1.set_ylabel("Amplitude (dB)")
     freq_ax1.set_xlim(freq[0], freq[-1])  # Limit the x axis from 0 to Nyquist frequency

     # Plot phase response
     freq_ax2.plot(freq, np.unwrap(np.angle(response, deg=True)), linewidth=FILTER_LINE_WIDTH)  # Plot phase in dB vs degrees
     freq_ax2.set_ylabel("Phase (°)")
     freq_ax2.set_xlim(freq[0], freq[-1])  # Limit the x axis from 0 to Nyquist frequency

     return FreqFilterResponse