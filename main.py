"""
    main.py
    The main module for ENEL420-20S2 Assignment 1.
    This module imports ECG data from a .txt and
    creates IIR and FIR notch filters to reduce
    narrowband noise from the data. Figures of
    the data, filtered data and filter responses
    are produced. The noise power is calculated.
    All results are saved in the current directory.

    Authors: Matt Blake   (58979250)
             Reweti Davis (23200856)
    Group Number: 18
    Last Modified: 07/08/2020
"""

# Imported libraries
from scipy.signal import freqz, lfilter, firwin, remez, firwin2, convolve, kaiserord
from scipy.fft import fft
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from signalPlots import *


# Functions

#
# Get data information functions
#
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
    """Compute and return the frequency spectrum of the input samples, for the specified sample rate"""

    freq_data = np.abs(fft(samples)) # Apply FFT to data
    freq = np.linspace(0, sample_rate, len(freq_data)) # Create an array of frequencies to be plotted
    return freq, freq_data



#
# IIR filter functions
#
def computeIIRNotchCoefficients(notch_freq, notch_width, sampling_freq):
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



def calculateGainFactor(numerator, denominator, passband_freq):
    """Calculate and return the coefficent needed to normalise the passband gain of an IIR filter to unity"""

    # Initalise variables
    num_coeff = len(numerator) # Calculate number of tap coefficents
    numerator_sum = 0 # The sum of filter's numerator at the passband frequency
    denominator_sum = 0 # The sum of filter's denominator at the passband frequency

    # Calculate the value of the numerator and denominator by iterating through each tap coefficent
    for delay_index in range(len(numerator)):

        # Calculate the value of the numerator at tap
        numerator_coeff = numerator[delay_index] # Extract numerator tap coefficent
        numerator_sum += numerator_coeff * (np.exp(((num_coeff - delay_index - 1) * 1j * 2 * np.pi * passband_freq))) # Add transfer function value to numerator sum

        # Calculate the value of the denominator at tap
        denominator_coeff = denominator[delay_index] # Extract denominator tap coefficent
        denominator_sum += denominator_coeff * (np.exp(((num_coeff - delay_index - 1) * 1j * 2 * np.pi * passband_freq)))  # Add transfer function value to numerator sum

    # Calculate gain factor.
    gain_factor = denominator_sum/numerator_sum # At unity gain: gain_factor * numerator_sum/denominator_sum = 1
    real_gain_factor = np.real(gain_factor) # Take the real component of the gain factor

    return real_gain_factor



def createIIRNotchFilter(notch_freq, notch_width, passband_f, sample_rate):
    """Create and return the coefficents of an IIR notch filter"""

    numerator, denominator = computeIIRNotchCoefficients(notch_freq, notch_width, sample_rate)  # Calculate filter coefficents
    gain_factor = calculateGainFactor(numerator, denominator, passband_f) # Calculate gain factors needed to get unity gain in passband
    normalised_numerator = np.array(numerator) * gain_factor  # Normalise passband of filters to unity gain

    return normalised_numerator, denominator



def applyIIRNotchFilters(numerator_1, denominator_1, numerator_2, denominator_2, data):
    """Pass data through two cascaded IIR filters and return the result after each filter"""

    partially_filtered_data = lfilter(numerator_1, denominator_1, data) # Apply first filter to data
    filtered_data = lfilter(numerator_2, denominator_2, partially_filtered_data) # Apply second notch filter to data

    return partially_filtered_data, filtered_data



def combineFilters(numerator_1, denominator_1, numerator_2, denominator_2):
    """Tales the numerators and denominators of two filters and convolutes them to create an overall filter.
    The numerator and denominator of this filter are returned"""

    numerator = convolve(numerator_1, numerator_2)  # Create the overall numerator via convolution
    denominator = convolve(denominator_1, denominator_2)  # Create the overall denominator via convolution

    return numerator, denominator



#
# FIR Filter functions
#
def createWindowFilter(notches, sample_rate, notch_width, num_taps):
    """Compute and return the bandstop  window filter array for the specified notches. Adjusting the window type and band width changes attenuation."""

    window = ('kaiser', 2.5)
    f1, f2 = notches
    width = notch_width / 2.0 #One sided 3dB bandwidth, in Hz

    cutoff_1 = [(f1 - width), (f1 + width)]
    cutoff_2 = [(f2 - width), (f2 + width)]
    cutoff = [(f1 - width), (f1 + width), (f2 - width), (f2 + width)]


    filter_1 = firwin(numtaps=num_taps, cutoff=cutoff_1, window=window, fs=sample_rate)
    filter_2 = firwin(numtaps=num_taps, cutoff=cutoff_2, window=window, fs=sample_rate)
    filter_overall =  firwin(numtaps=num_taps, cutoff=cutoff, window=window, fs=sample_rate)
    
    return filter_1, filter_2, filter_overall



def createOptimalFilter(notches, sample_rate, notch_width, gains, num_taps):
    """Compute and return the bandstop  optimal filter arrays for the specified notches. Adjusting the window type and band width changes attenuation."""

    f1, f2 = notches
    width = notch_width / 2.0 #One sided 3dB bandwidth, in Hz - 5.0 / 1.3
    stop = 1000 #Stop band weighting - 12
    pass_ = 1 #Passband weighting - 1
    weight = [pass_, stop, pass_] #Isolated filter weighting
    weight_overall = [pass_, stop, pass_, stop, pass_] #Overall filter weighting
    gains_overall = [1, 0, 1, 0, 1] #Indicates stop and passband locations in the specified bands
    alpha = 0.001 #Minimal Spacing of notch to allow convergence

    band_1= [0,  f1 - width, f1 - alpha, f1 + alpha, f1 + width, sample_rate / 2] #Pad the stop band as the method doesnt convege well otherwise
    band_2= [0, f2 - width, f2 - alpha, f2 + alpha, f2 + width, sample_rate / 2]
    bands = [0,  f1 - width, f1 - alpha, f1 + alpha, f1 + width, f2 - width, f2 - alpha, f2 + alpha, f2 + width, sample_rate / 2] 

    filter_1 = remez(numtaps=num_taps, bands=band_1, desired=gains, fs=sample_rate, weight=weight)
    filter_2 = remez(numtaps=num_taps, bands=band_2, desired=gains, fs=sample_rate, weight=weight)
    filter_overall = remez(numtaps=num_taps, bands=bands, desired=gains_overall, fs=sample_rate, weight=weight_overall)
    
    return filter_1, filter_2, filter_overall



def createFreqSamplingFilter(notches, sample_rate, notch_width, gains, num_taps):
    """Compute and return the bandstop frequency sampling filter arrays for the specified notches. Adjusting the window type and band width changes attenuation."""

    f1, f2 = notches
    window_type = ('kaiser', 2.5)
    width = notch_width / 2.0 #One sided 3dB bandwidth, in Hz
    alpha = width - 0.01 #Added transition points to narrow the band further
    omega = width - 0.1 
    #gain = np.power(10, np.array(gains)/20.0)
    gains = [1, 1, 0, 0, 0, 0, 0, 1, 1]
    gains_overall = [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1]

    freq_1 = [0, f1 - width, f1 - alpha, f1 - omega, f1, f1 + omega, f1 + alpha, f1 + width, sample_rate / 2]
    freq_2 = [0, f2 - width, f2 - alpha, f2 - omega, f2, f2 + omega, f2 + alpha, f2 + width, sample_rate / 2]
    freq = [0, f1 - width, f1 - alpha, f1 - omega, f1, f1 + omega, f1 + alpha, f1 + width, f2 - width, f2 - alpha, f2 - omega, f2, f2 + omega, f2 + alpha, f2 + width, sample_rate / 2]

    filter_1 = firwin2(numtaps=num_taps, freq=freq_1, gain=gains, fs=sample_rate, window=window_type)
    filter_2 = firwin2(numtaps=num_taps, freq=freq_2, gain=gains, fs=sample_rate, window=window_type)
    filter_overall = firwin2(numtaps=num_taps, freq=freq, gain=gains_overall, fs=sample_rate, window=window_type)
    
    return filter_1, filter_2, filter_overall



def applyFIRFilters(filter_1, filter_2, filter_overall, samples):
    """Pass data through two cascaded FIR filters and return the result after each filter"""

    half_filtered = lfilter(filter_1, 1, samples)
    full_filtered = lfilter(filter_2, 1, half_filtered)
    overall_filtered = lfilter(filter_overall, 1, samples)

    return half_filtered, full_filtered, overall_filtered



#
# Noise Power (variance) calculations
#
def calculateVariance(data):
    """Calculates and returns the variance of a signal"""

    # Calculate the variance of the signal X using: variance = E[X^2] - E[X]^2
    expected_data_power = sum((np.square(data)))/len(data)  # Calculate E[X^2]
    power_of_expected_data = np.square(sum(data)/len(data))  # Calculate E[X]^2
    variance_data = expected_data_power - power_of_expected_data  # Calculate the variance of the data

    return variance_data



def calculateNoiseVariance(data, filtered_data):
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



#
# File functions
#
def createClean(filename, directory=False):
    """Create a file/folder at the target location and returns the path to this if it is a folder or a the file ready
    for reading and writing if it is a file.
    Deletes a previously created file if it exists, so that a new file can be written cleanly"""

    # Remove old file
    if os.path.exists(filename): # Check if an output path already exists
        if directory == True: # If a folder is to be created
            shutil.rmtree(filename) # Remove previous output folder, so the figures can be cleanly saved
        else: # If a file is to be created
            os.remove(filename) # Remove the previous file, so the file can be cleanly saved

    # Create file
    if directory == True: # If a folder is to be created
        output = os.path.join(filename)  # The output folder for the figures to be saved
        os.mkdir(output) # Create output folder
    else: # If a file is to be created
        output = open(filename, "w+") # Create an open the file for reading and writing

    return output



def saveFigures(figures, figures_location, figure_names):
    """Save a list of figures as the corresponding name"""

    output_path = createClean(figures_location, True) # Create a clear output path for figures to be stored in

    # Iterate through each figure saving it as the corresponding name
    for i in range(len(figures)):
        plt.figure(figures[i].number) # Set the figure as the current figure
        plt.savefig(output_path + '/' + figure_names[i]) # Save the current figure



def saveNoisePowerData(noise_power_data, noise_power_output_filename):
    """Iterate through a list of filters, saving the noise power (variance) data"""

    # Create text to write and file to write to
    variance_text_1 = 'The mean power removed by the '  # The first section of the text string to print
    variance_text_2 = ' is {:.1f} pW\n' # The section section of the text string to print
    outputfile = createClean(noise_power_output_filename)  # Create output file

    # Write data to file
    for filter_name, filter_noise_power in noise_power_data.items(): # Iterate through filters
        output_string = variance_text_1 + filter_name + variance_text_2.format(filter_noise_power) # The text to save
        outputfile.write(output_string) # Save the noise power (variance) data for that filter



def main():
    """Main function of ENEL420 Assignment 1"""

    # Define filenames
    filename = 'enel420_grp_18.txt' # Location in project where ECG data is stored
    figures_filename = 'Group_18_Figures' # Folder to save created figure images to
    noise_power_output_filename = 'Group_18_Noise_Power_(Variance)_Data_from_Created_Filters.txt' # File to save calculated noise power data
    figure_names = ['ECG_Time_Plot.png', 'ECG_Freq_Plot.png', 'IIR_Pole_Zero_Plot.png', 'IIR_Notched_ECG_Time_Plot.png',
                    'IIR_Notched_Freq_Plot.png', 'IIR_Frequency_Response.png', 'Windowed_ECG_Time_Plot.png',
                    'Windowed_Freq_Plot.png', 'Windowed_Frequency_Response.png', 'Optimal_ECG_Time_Plot.png',
                    'Optimal_Freq_Plot.png', 'Optimal_Frequency_Response.png', 'Freq_Sampled_ECG_Time_Plot.png',
                    'Freq_Sampled_Freq_Plot.png', 'Freq_Sampled_Frequency_Response.png']  # The names that each figure should be saved as

    # Define filter and data parameters
    sample_rate = 1024  # Sample rate of data (Hz)
    cutoff = [57.755, 88.824] # Frequencies to attenuate (Hz), which were calculated based on previous graphical analysis
    passband_f = [10, 10] # Passband frequencies (Hz) used to calculate the gain factor
    notch_width = 5 # 3 dB bandwidth of the notch filters (Hz)
    optimal_gains = [1, 0, 1]
    freq_gains = [1, 1, 0, 1, 1]
    num_FIR_taps = 399 # The number for each FIR filter

    # Gather data from input files
    samples = importData(filename) # Import data from file
    base_time = getTimeData(sample_rate, len(samples)) # Create a time array based on imported data
    base_freq, base_freq_data = calcFreqSpectrum(samples, sample_rate) # Calculate the frequency spectrum of the data

    # Create IIR Notch filters and use them to filter the ECG data
    notch_num_1, notch_denom_1 = createIIRNotchFilter(cutoff[0], notch_width, passband_f[0], sample_rate) # Calculate the first notch filter's coefficents
    notch_num_2, notch_denom_2 = createIIRNotchFilter(cutoff[1], notch_width, passband_f[1], sample_rate) # Calculate the second notch filter's coefficents
    half_notched_samples, notched_samples = applyIIRNotchFilters(notch_num_1, notch_denom_1, notch_num_2, notch_denom_2, samples) # Apply cascaded notch filters to data
    notch_time = getTimeData(sample_rate, len(notched_samples)) # Create a time array based on notch filtered data
    notch_frequency, notch_freq_data = calcFreqSpectrum(notched_samples, sample_rate) # Calculate frequency of the IIR filtered ECG data
    notched_numerator, notched_denominator = combineFilters(notch_num_1, notch_denom_1, notch_num_2, notch_denom_2)  # Combine the two IIR notch filters

    # Create and apply FIR filters to data
    window_filter_1, window_filter_2, window_filter_overall = createWindowFilter(cutoff, sample_rate, notch_width, num_FIR_taps) # Calculate window filter coefficents
    half_windowed_samples, full_windowed_samples, overall_windowed_samples = applyFIRFilters(window_filter_1, window_filter_2, window_filter_overall, samples) # Apply window filter to data
    win_time = getTimeData(sample_rate, len(full_windowed_samples)) # Create a time array based on window filtered data
    win_frequency, win_freq_data = calcFreqSpectrum(overall_windowed_samples, sample_rate) # Calculate frequency of the window IIR filtered ECG data

    optimal_filter_1, optimal_filter_2, optimal_filter_overall = createOptimalFilter(cutoff, sample_rate, notch_width, optimal_gains, num_FIR_taps)
    half_optimal_samples, full_optimal_samples, overall_optimal_samples = applyFIRFilters(optimal_filter_1, optimal_filter_2, optimal_filter_overall, samples)
    opt_time = getTimeData(sample_rate, len(full_optimal_samples)) # Create a time array based on optimal filtered data
    opt_frequency, opt_freq_data = calcFreqSpectrum(overall_optimal_samples, sample_rate) # Calculate frequency of the window IIR filtered ECG data
    
    freq_sampling_filter_1, freq_sampling_filter_2, freq_filter_overall  = createFreqSamplingFilter(cutoff, sample_rate, notch_width, freq_gains, num_FIR_taps)
    half_freq_samples, full_freq_samples, overall_freq_samples = applyFIRFilters(freq_sampling_filter_1, freq_sampling_filter_2, freq_filter_overall, samples)
    freq_sampling_time = getTimeData(sample_rate, len(full_freq_samples)) # Create a time array based on optimal filtered data
    freq_s_frequency, freq_s_freq_data = calcFreqSpectrum(overall_freq_samples, sample_rate) # Calculate frequency of the window IIR filtered ECG data

    # Plot unfiltered data
    ECG = plotECG(samples, base_time) # Plot a time domain graph of the ECG data
    ECGSpectrum = plotECGSpectrum(base_freq, base_freq_data) # Plot the frequency spectrum of the ECG data

    # Plot IIR notch filtered data
    IIRPoleZero = plotIIRPoleZero(cutoff, notch_width, sample_rate) # Plot a pole-zero plot of the created IIR notch filter
    IIRNotchECG = plotIIRNotchECG(notched_samples, notch_time) # Plot a time domain graph of the IIR notch filtered ECG data
    IIRNotchECGSpectrum = plotIIRNotchECGSpectrum(notch_frequency, notch_freq_data) # Plot the frequency spectrum of the IIR notch filtered ECG data
    IIRNotchFilterResponse = plotIIRNotchFilterResponse(notched_numerator, notched_denominator, sample_rate) # Plot the frequency response of the notch filter

    # Plot window filtered data
    WindowedECG = plotWindowedECG(overall_windowed_samples, win_time) # Plot a time domain graph of the window filtered ECG data
    WindowedECGSpectrum = plotWindowedECGSpectrum(win_frequency, win_freq_data) # Plot the frequency spectrum of the window filtered ECG data
    WindowFilterResponse = plotWindowFilterResponse(window_filter_overall, sample_rate) # Plot the frequency response of the window filter

    #Plot optimal filtered data
    OptimalECG = plotOptimalECG(overall_optimal_samples, opt_time) # Plot a time domain graph of the window filtered ECG data
    OptimalECGSpectrum = plotOptimalECGSpectrum(opt_frequency, opt_freq_data) # Plot the frequency spectrum of the window filtered ECG data
    OptimalFilterResponse = plotOptimalFilterResponse(optimal_filter_overall, sample_rate) # Plot the frequency response of the window filter

    #Plot Frequency Sampling filtered data
    FrequencySamplingECG = plotFrequencySampledECG(overall_freq_samples, freq_sampling_time) # Plot a time domain graph of the window filtered ECG data
    FrequencySamplingECGSpectrum = plotFrequencySampledECGSpectrum(freq_s_frequency, freq_s_freq_data) # Plot the frequency spectrum of the window filtered ECG data
    FrequencySamplingFilterResponse = plotFrequencySampledFilterResponse(freq_filter_overall, sample_rate) # Plot the frequency response of the window filter

    # Save figures
    figures = [ECG, ECGSpectrum, IIRPoleZero, IIRNotchECG, IIRNotchECGSpectrum, IIRNotchFilterResponse, WindowedECG,
               WindowedECGSpectrum, WindowFilterResponse, OptimalECG, OptimalECGSpectrum, OptimalFilterResponse,
               FrequencySamplingECG, FrequencySamplingECGSpectrum, FrequencySamplingFilterResponse] # The figures to save, which must be in the same order as figure_names
    saveFigures(figures, figures_filename, figure_names) # Save the figures to an output folder in the current directory

    # Calculate the variance of IIR filtered data
    notched_noise_variance = calculateNoiseVariance(samples, notched_samples)  # Calculate the variance of the noise removed by the IIR notch filters
    first_notched_noise_variance = calculateNoiseVariance(samples, half_notched_samples)  # Calculate the variance of the noise removed by the first IIR notch filter
    second_notched_noise_variance = calculateNoiseVariance(half_notched_samples, notched_samples)  # Calculate the variance of the noise removed by the second IIR notch filter

    # Calculate the variance of window filtered data
    window_noise_variance = calculateNoiseVariance(samples, overall_windowed_samples)  # Calculate the variance of the noise removed by the 
    first_window_noise_variance = calculateNoiseVariance(samples, half_windowed_samples)  # Calculate the variance of the noise removed by the 
    second_window_noise_variance = calculateNoiseVariance(half_windowed_samples, full_windowed_samples)  # Calculate the variance of the noise removed by the 

    # Calculate the variance of optimal filtered data
    optimal_noise_variance = calculateNoiseVariance(samples, overall_optimal_samples)  # Calculate the variance of the noise removed by the 
    first_optimal_noise_variance = calculateNoiseVariance(samples, half_optimal_samples)  # Calculate the variance of the noise removed by the 
    second_optimal_noise_variance = calculateNoiseVariance(half_optimal_samples, full_optimal_samples)  # Calculate the variance of the noise removed by the 

    # Calculate the variance of frequency sampling filtered data
    freq_sampling_noise_variance = calculateNoiseVariance(samples, overall_freq_samples)  # Calculate the variance of the noise removed by the 
    first_freq_sampling_noise_variance = calculateNoiseVariance(samples, half_freq_samples)  # Calculate the variance of the noise removed by the 
    second_freq_sampling_noise_variance = calculateNoiseVariance(half_freq_samples, full_freq_samples)  # Calculate the variance of the noise removed by the

    # Save noise power to a .txt file
    noise_power_data = {'IIR notch filters': notched_noise_variance,
                        'first IIR notch filter': first_notched_noise_variance,
                        'second IIR notch filter': second_notched_noise_variance,
                        'FIR Window filters': window_noise_variance,
                        'first window filter': first_window_noise_variance,
                        'second window filter': second_window_noise_variance,
                        'FIR Optimal filters': optimal_noise_variance,
                        'first optimal filter': first_optimal_noise_variance,
                        'second optimal filter': second_optimal_noise_variance,
                        'FIR Frequency Sampling filters': freq_sampling_noise_variance,
                        'first frequency sampling filter': first_freq_sampling_noise_variance,
                        'second frequency sampling filter': second_freq_sampling_noise_variance
                        }  # Create a dictionary of the filter name and its noise power
    saveNoisePowerData(noise_power_data, noise_power_output_filename)  # Save the data about each filter to a file
    #plt.show()



# Run program if called
if __name__ == '__main__':
    main()
