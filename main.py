"""
    main.py
    The main module for ENEL420-S2 Assignment 1.
    This module imports ECG data from a .txt and
    creates IIR and FIR notch filters to reduce
    narrowband noise from the data. Figures of
    the data, filtered data and filter responses
    are produced.

    Authors: Matt Blake   (58979250),
             Reweti Davis (23200856).
    Last Modified: 07/08/2020
"""

# Imported libraries
from scipy.signal import freqz, lfilter, firwin, remez, firwin2, convolve
from scipy.fft import fft
import matplotlib.pyplot as plt
import numpy as np
from signalPlots import *


# Functions
def importData(filename):
    """Import data from a text file"""

    # Extract data from file
    data_file = open(filename, 'r') # Create the file object where data is stored
    data_string = data_file.read() # Read the data from the file object
    data_list = data_string.split() # Create a list of each sample from the singular data string

    # Convert data from strings to floats
    data = [] # Create array for results to be stored in
    for data_string in data_list:
        sample = float(data_string)
        data.append(sample)

    return data



def getTimeData(sample_rate, num_samples):
    """Create and return an array containing the time each sample is taken. This assumes equal sampling periods"""

    time = [] # Create an array for the results to be stored in
    for i in range(num_samples): # Iterate through each sample
        sample_time = i/sample_rate # Calculate the time this sample is taken
        time.append(sample_time) # Add time to results array

    return time



def calcFreqSpectrum(samples, sample_rate):
    """Compute and return the frequency spectrum of the input samples, for the specified sample rate."""

    freq_data = np.abs(fft(samples))
    freq = np.linspace(0, sample_rate, len(freq_data))
    return freq, freq_data



def computeNotchCoefficients(notch_freq, notch_width, sampling_freq):
    """Compute and return the optimal notch filter coefficients, based on the notch frequency, the 3 dB width of the
    notch and the sampling frequency"""

    rads_circle = 2 * np.pi # Define the number of radians in a circle (for unit circle analysis)

    # Calculate the locations of the zero conjugate pair
    zeros_magnitude = 1 # Place the zeros on the unit circle for maximum attenuation
    zeros_phase = notch_freq/sampling_freq * rads_circle # Calculate the optimal phase for the zero pairs\

    # Calculate the locations of the pole conjugate pair
    poles_magnitude = 1 - np.pi * (notch_width/sampling_freq)  # Calculate the optimal magnitude for the pole pairs
    poles_phase = notch_freq/sampling_freq * rads_circle  # Calculate the optimal phase for the pole pairs

    # Calculate feedfoward tap coefficients
    a0 = 1 * 1 # Calculate the zero delay term zero coefficent
    a1 = -(1 * zeros_magnitude * np.exp(1j * zeros_phase) + 1 * zeros_magnitude * np.exp(-1j * zeros_phase)) # Calculate the one delay term zero coefficent
    a2 = zeros_magnitude * np.exp(1j * zeros_phase) * zeros_magnitude * np.exp(-1j * zeros_phase) # Calculate the two delay term zero coefficent
    numerator = [np.real(a0), np.real(a1), np.real(a2)] # Store feedfoward coefficents in an array

    # Calculate feedback tap coefficients
    b0 = 1 * 1  # Calculate the zero delay term zero coefficent
    b1 = -(1 * poles_magnitude * np.exp(1j * poles_phase) + 1 * poles_magnitude * np.exp(-1j * poles_phase))  # Calculate the one delay term zero coefficent
    b2 = poles_magnitude * np.exp(1j * poles_phase) * poles_magnitude * np.exp(-1j * poles_phase)  # Calculate the two delay term zero coefficent
    denominator = [np.real(b0), np.real(b1), np.real(b2)]  # Store feedback coefficents in an array

    return numerator, denominator



def applyNotchFilters(notch_freq_1, notch_freq_2, notch_width, sample_rate, data):
    """Create notch filters, which are then applied to a data. This filtered data is returned"""

    # Create notch filters
    numerator_1, denominator_1 = computeNotchCoefficients(notch_freq_1, notch_width, sample_rate)  # Calculate notch filter coefficents for the first notch frequency
    numerator_2, denominator_2 = computeNotchCoefficients(notch_freq_2, notch_width, sample_rate)  # Calculate notch filter coefficents for the second notch frequency

    # Graph frequency response of filters
    plotIIRNotchFilterResponse(numerator_1, denominator_1) # Plot the frequency response of the first notch filter
    plotIIRNotchFilterResponse(numerator_2, denominator_2) # Plot the frequency response of the second notch filter
    #plotFilterSpectrum(filter_frequencies_1, filter_response_1) # Plot the frequency response of the first notch filter
    #plotFilterSpectrum(filter_frequencies_2, filter_response_2)  # Plot the frequency response of the second notch filter

    # Cascade filters and apply to data
    partially_filtered_data = lfilter(numerator_1, denominator_1, data)
    filtered_data = lfilter(numerator_2, denominator_2, partially_filtered_data)

    return filtered_data



def windowFilter(notches, sample_rate, notch_width):
    """Compute and return the bandstop  window filter array for the specified notches."""

    NUM_TAPS = 399 #Max number of taps allowed
    f1, f2 = notches
    width = notch_width / 2.0  #One sided 3dB bandwidth, in Hz
    ny = sample_rate / 2.0

    cutoff = [(f1 - width)/ny, (f1 + width)/ny, (f2 - width)/ny, (f2 + width)/ny]
    filter_array = firwin(numtaps=NUM_TAPS, cutoff=cutoff, window=('kaiser', 2))

    #Following lines are for interest, hp and lp, to erradicate excess noise in signal...
    # hp = firwin(numtaps=NUM_TAPS, cutoff=(10/ny), pass_zero=False)
    # lp = firwin(numtaps=NUM_TAPS, cutoff=(100/ny))
    # hectic = convolve(hp, filter_array)
    # filter_array = convolve(hectic, lp)
    
    return filter_array



def main():
    """Main function of ENEL420 Assignment 1"""
    plt.close('all')

    filename = 'enel420_grp_18.txt' # Location in project where ECG data is stored
    sample_rate = 1024  # Sample rate of data (Hz)
    
    cutoff = [57.755, 88.324] # Shifted f2 down by 0.5 Hz to improve response
    notch_width = 5 # 3 dB bandwidth of the notch filters (Hz)

    samples = importData(filename) # Import data from file
    base_time = getTimeData(sample_rate, len(samples))
    base_freq, base_freq_data = calcFreqSpectrum(samples, sample_rate)

    # Gather data from input files
    notched_samples = applyNotchFilters(cutoff[0], cutoff[1], notch_width, sample_rate, samples) # Calculate notch filter coefficents
    notch_time = getTimeData(sample_rate, len(notched_samples))

    # Create and apply filters to data
    window_filter = windowFilter(cutoff, sample_rate, notch_width)
    windowed_samples = convolve(samples, window_filter)
    win_time = getTimeData(sample_rate, len(windowed_samples))
    win_frequency, win_freq_data = calcFreqSpectrum(windowed_samples, sample_rate)

    # Plot figures of data and frequency responses
    plotECG(samples, base_time) # Plot a time domain graph of the ECG data
    plotECGSpectrum(base_freq, base_freq_data)
    plotNotchedECG(notched_samples, notch_time)
    #plotFilterSpectrum(filter_frequencies, filter_response) # Plot the frequency response of the notch filter
    plotWindowedECG(windowed_samples, win_time)
    plotWindowedECGSpectrum(win_frequency, win_freq_data)
    plotWindowFilterResponse(window_filter)
    plt.show()  # Display figures


if __name__ == "__main__":
    main()



