from scipy.signal import freqz, lfilter, firwin, remez, convolve
from scipy.fft import fft
import matplotlib.pyplot as plt
import numpy as np


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


def plotECG(samples, sample_rate):
    """Plot a time domain graph of the ECG data"""

    # Get timing data
    num_samples = len(samples) # Calculate the number of ECG sample
    time = getTimeData(sample_rate, num_samples) # Create an array containing the time each sample was taken

    # Plot ECG
    plt.figure(1)
    plt.plot(time, samples)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV)")
    plt.suptitle("Time domain ECG signal")
    plt.xlim(time[0], time[-1]) # Limit the x axis to locations with data points


def plotECGSpectrum(freq, freq_data):
    
    # Plot ECG
    plt.figure(2)
    plt.plot(freq, freq_data)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.suptitle("Frequency Spectrum of the ECG signal")
    plt.xlim(freq[0], freq[-1] / 2) # Limit the x axis to locations with data points


def calcFreqSpectrum(samples, sample_rate):
    freq_data = np.abs(fft(samples))
    freq = np.linspace(0, sample_rate/2, len(freq_data))
    return freq, freq_data


def computeNotchFilter(notch_freq, notch_width, sampling_freq):
    """Compute and return the optimal notch filter coefficents, based on the notch frequency, the 3 dB width of the
    notch and the sampling frequency"""

    # Calculate the locations of the zero conjugate pair
    zeros_magnitude = 1 # Place the zeros on the unit circle for maximum attenuation
    zeros_phase = notch_freq/sampling_freq * DEGREES_CIRCLE # Calculate the optimal phase for the zero pairs\

    # Calculate the locations of the pole conjugate pair
    zeros_magnitude = 1 - np.pi * (notch_width/sampling_freq)  # Calculate the optimal magnitude for the pole pairs
    zeros_phase = notch_freq / sampling_freq * DEGREES_CIRCLE  # Calculate the optimal phase for the pole pairs


def main():
    """Main function of ENEL420 Assignment 1"""
    plt.close('all')
    filename = 'enel420_grp_18.txt' # Location in project where ECG data is stored
    sample_rate = 1024  # Sample rate of data (Hz)

    samples = importData(filename) # Import data from file
    frequency, frequency_data = calcFreqSpectrum(samples, sample_rate)

    plotECG(samples, sample_rate) # Plot a time domain graph of the ECG data
    plotECGSpectrum(frequency, frequency_data)

    plt.show()  # Display figures


main()