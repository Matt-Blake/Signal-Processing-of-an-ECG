from scipy.signal import freqz, lfilter, firwin, remez, convolve
from scipy.fft import fft
import matplotlib.pyplot as plt
import numpy as np

DEGREES_CIRCLE = 360

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
    """Plot the frequency spectrum of ECG data"""

    # Plot spectrum
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


def computeNotchCoefficients(notch_freq, notch_width, sampling_freq):
    """Compute and return the optimal notch filter coefficients, based on the notch frequency, the 3 dB width of the
    notch and the sampling frequency"""

    # Calculate the locations of the zero conjugate pair
    zeros_magnitude = 1 # Place the zeros on the unit circle for maximum attenuation
    zeros_phase = notch_freq/sampling_freq * DEGREES_CIRCLE # Calculate the optimal phase for the zero pairs\

    # Calculate the locations of the pole conjugate pair
    poles_magnitude = 1 - np.pi * (notch_width/sampling_freq)  # Calculate the optimal magnitude for the pole pairs
    poles_phase = notch_freq / sampling_freq * DEGREES_CIRCLE * 1.08  # Calculate the optimal phase for the pole pairs

    # Calculate feedfoward tap coefficients
    a0 = 1 * 1 # Calculate the zero delay term zero coefficent
    a1 = 1 * zeros_magnitude * np.exp(complex(1j * zeros_phase) + 1 * zeros_magnitude * np.exp(-1j * zeros_phase)) # Calculate the one delay term zero coefficent
    a2 = zeros_magnitude * np.exp(1j * zeros_phase) * zeros_magnitude * np.exp(-1j * zeros_phase) # Calculate the two delay term zero coefficent
    #a = [a0, a1, a2] # Store feedfoward coefficents in an array
    a = [1, -1.91003, 1]

    # Calculate feedback tap coefficients
    b0 = 1 * 1  # Calculate the zero delay term zero coefficent
    b1 = 1 * poles_magnitude * np.exp(1j * poles_phase) + 1 * poles_magnitude * np.exp(-1j * poles_phase)  # Calculate the one delay term zero coefficent
    b2 = poles_magnitude * np.exp(1j * poles_phase) * poles_magnitude * np.exp(-1j * poles_phase)  # Calculate the two delay term zero coefficent
    #b = [b0, b1, b2]  # Store feedback coefficents in an array
    b = [1, -1.7922001, 0.969555]

    # Calculate IIR tap coefficents
    filter_frequencies, filter_response = freqz(a, b, fs=sampling_freq)

    return filter_frequencies, filter_response


def createNotchFilter(notch_freq_1, notch_freq_2, notch_width, sample_rate):
    """Create and return a notch filter, which filters out two frequency bands"""

    filter_frequencies_1, filter_response_1 = computeNotchCoefficients(notch_freq_1, notch_width, sample_rate)  # Calculate notch filter coefficents for the first notch frequency
    filter_frequencies_2, filter_response_2 = computeNotchCoefficients(notch_freq_2, notch_width, sample_rate)  # Calculate notch filter coefficents for the second notch frequency
    filter_response = convolve(filter_response_1, filter_response_2)  # Combine the two notch filters to create one overall filter
    print(len(filter_frequencies_1))
    print(len(filter_response_1))
    print(len(filter_response))

    return filter_frequencies_1, filter_response_1


def plotFilterSpectrum(freq, freq_data):
    """Plot the frequency spectrum of a filter response"""

    # Plot spectrum
    plt.figure(3)
    plt.plot(freq, freq_data)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.suptitle("Frequency Response of the Notch Filter")
    #plt.xlim(freq[0], freq[-1] / 2) # Limit the x axis to locations with data points


def main():
    """Main function of ENEL420 Assignment 1"""
    plt.close('all')

    filename = 'enel420_grp_18.txt' # Location in project where ECG data is stored
    sample_rate = 1024  # Sample rate of data (Hz)

    notch_freq_1 = 28.877
    notch_freq_2 = 44.412
    notch_width = 5 # 3 dB bandwidth of the notch filters (Hz)

    samples = importData(filename) # Import data from file
    frequency, frequency_data = calcFreqSpectrum(samples, sample_rate)

    plotECG(samples, sample_rate) # Plot a time domain graph of the ECG data
    plotECGSpectrum(frequency, frequency_data) # Plot the frequency spectrum of the ECG data
    filter_frequencies, filter_response = createNotchFilter(notch_freq_1, notch_freq_2, notch_width, sample_rate) # Calculate notch filter coefficents
    plotFilterSpectrum(filter_frequencies, filter_response) # Plot the frequency response of the notch filter

    plt.show()  # Display figures


main()