"""
    noise.py (Signal Processing of an ECG program)
    Contains all the noise variance computation functions for
    the Signal Processing of an ECG program.

    Authors: Matt Blake   (58979250)
             Reweti Davis (23200856)
    Last Modified: 27/12/2020
"""

# Imported libraries
from scipy.signal import freqz, lfilter, firwin, remez, firwin2, convolve
from scipy.fft import fft
import numpy as np
from configFiles import *

# Constants
VARIANCE_TEXT_1 = 'The mean power removed by the '  # The first section of the text string to print
VARIANCE_TEXT_2 = ' is {:.1f} pW\n' # The section section of the text string to print


# Functions
def saveNoisePowerData(noise_power_data:list, noise_power_output_filename:str):
    """Iterate through a list of filters, saving the noise power (variance) data"""

    outputfile = createClean(noise_power_output_filename)  # Create output file
    for filter_name, filter_noise_power in noise_power_data.items(): # Iterate through filters
        output_string = VARIANCE_TEXT_1 + filter_name + VARIANCE_TEXT_2.format(filter_noise_power) # The text to save
        outputfile.write(output_string) # Save the noise power (variance) data for that filter



def calculateVariance(data:list):
    """Calculates and returns the variance of a signal"""

    # Calculate the variance of the signal X using: variance = E[X^2] - E[X]^2
    expected_data_power = sum((np.square(data)))/len(data)  # Calculate E[X^2]
    power_of_expected_data = np.square(sum(data)/len(data))  # Calculate E[X]^2
    variance_data = expected_data_power - power_of_expected_data  # Calculate the variance of the data

    return variance_data



def calculateNoiseVariance(data:list, filtered_data:list):
    """"Calculate the variance of the noise by comparing the filtered and unfiltered data. The variance of the noise
    is approximated as the variance of the signal removed by the filter"""

    # Turn data arrays into numpy arrays so that mathematical operations can be performed
    np_data = np.array(data)
    np_filtered_data = np.array(filtered_data)

    # Calculate the variance of the removed noise by finding the variances of the filtered and unfiltered data
    data_variance = calculateVariance(np_data) # Calculate the variance of the unfiltered data
    filtered_data_variance = calculateVariance(np_filtered_data) # Calculat the variance of the filtered data
    noise_data_variance = data_variance - filtered_data_variance # Calculate the variance of the removed noise

    return noise_data_variance
