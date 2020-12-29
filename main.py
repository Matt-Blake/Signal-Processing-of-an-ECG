"""
    main.py (Signal Processing of an ECG program)
    The main module for the Signal Processing of an ECG program.
    This module imports ECG data from a .txt and creates IIR and
    FIR notch filters to reduce narrowband noise from the data.
    Figures of the data, filtered data and filter responses are
    produced. The noise power is calculated. All results are
    saved in the current directory. This was first designed for
    ENEL420 Assignment 1.

    Authors: Matt Blake   (58979250)
             Reweti Davis (23200856)
    Last Modified: 27/12/2020
"""


"""Todo
    # Make main smaller (extract repeated variance cals to function)
    # suss the importing libraries better (no need to import * everything)
    # Add -> to configFiles final function
    # Check if -> plt.figure in signalPlots.py needs to be changed
"""

# Imported libraries
from scipy.signal import freqz, lfilter, firwin, remez, firwin2, convolve, kaiserord
from scipy.fft import fft
import numpy as np
from config import *
from signalPlots import *
from IIR import *
from FIR import *
from noise import *
from configFiles import *
from config import *



def main():
    """Main function of the Signal Processing of an ECG program."""

    # Gather data from input files
    samples = importData(DATA_FILENAME) # Import data from file
    base_time = getTimeData(SAMPLE_RATE, len(samples)) # Create a time array based on imported data
    base_freq, base_freq_data = calcFreqSpectrum(samples, SAMPLE_RATE) # Calculate the frequency spectrum of the data

    # Create IIR Notch filters and use them to filter the ECG data
    notch_num_1, notch_denom_1 = createIIRNotchFilter(CUTOFF_FREQS[0], NOTCH_WIDTH, PASSBAND_FREQ, SAMPLE_RATE) # Calculate the first notch filter's coefficents
    notch_num_2, notch_denom_2 = createIIRNotchFilter(CUTOFF_FREQS[1], NOTCH_WIDTH, PASSBAND_FREQ, SAMPLE_RATE) # Calculate the second notch filter's coefficents
    half_notched_samples, notched_samples = applyIIRNotchFilters(notch_num_1, notch_denom_1, notch_num_2, notch_denom_2, samples) # Apply cascaded notch filters to data
    notch_time = getTimeData(SAMPLE_RATE, len(notched_samples)) # Create a time array based on notch filtered data
    notch_frequency, notch_freq_data = calcFreqSpectrum(notched_samples, SAMPLE_RATE) # Calculate frequency of the IIR filtered ECG data
    notched_numerator, notched_denominator = combineFilters(notch_num_1, notch_denom_1, notch_num_2, notch_denom_2)  # Combine the two IIR notch filters

    # Create and apply FIR filters to data
    window_filter_1, window_filter_2, window_filter_overall = createWindowFilters(CUTOFF_FREQS, SAMPLE_RATE, NOTCH_WIDTH, NUM_FIR_TAPS) # Calculate window filter coefficents
    half_windowed_samples, full_windowed_samples, overall_windowed_samples = applyFIRFilters(window_filter_1, window_filter_2, window_filter_overall, samples) # Apply window filter to data
    win_time = getTimeData(SAMPLE_RATE, len(full_windowed_samples)) # Create a time array based on window filtered data
    win_frequency, win_freq_data = calcFreqSpectrum(overall_windowed_samples, SAMPLE_RATE) # Calculate frequency of the window IIR filtered ECG data

    optimal_filter_1, optimal_filter_2, optimal_filter_overall = createOptimalFilters(CUTOFF_FREQS, SAMPLE_RATE, NOTCH_WIDTH, NUM_FIR_TAPS)
    half_optimal_samples, full_optimal_samples, overall_optimal_samples = applyFIRFilters(optimal_filter_1, optimal_filter_2, optimal_filter_overall, samples)
    opt_time = getTimeData(SAMPLE_RATE, len(full_optimal_samples)) # Create a time array based on optimal filtered data
    opt_frequency, opt_freq_data = calcFreqSpectrum(overall_optimal_samples, SAMPLE_RATE) # Calculate frequency of the window IIR filtered ECG data
    
    freq_sampling_filter_1, freq_sampling_filter_2, freq_filter_overall  = createFreqSamplingFilters(CUTOFF_FREQS, SAMPLE_RATE, NOTCH_WIDTH, NUM_FIR_TAPS)
    half_freq_samples, full_freq_samples, overall_freq_samples = applyFIRFilters(freq_sampling_filter_1, freq_sampling_filter_2, freq_filter_overall, samples)
    freq_sampling_time = getTimeData(SAMPLE_RATE, len(full_freq_samples)) # Create a time array based on optimal filtered data
    freq_s_frequency, freq_s_freq_data = calcFreqSpectrum(overall_freq_samples, SAMPLE_RATE) # Calculate frequency of the window IIR filtered ECG data

    # Plot unfiltered data
    ECG = plotECG(samples, base_time) # Plot a time domain graph of the ECG data
    ECGSpectrum = plotECGSpectrum(base_freq, base_freq_data) # Plot the frequency spectrum of the ECG data

    # Plot IIR notch filtered data
    IIRPoleZero = plotIIRPoleZero(CUTOFF_FREQS, NOTCH_WIDTH, SAMPLE_RATE) # Plot a pole-zero plot of the created IIR notch filter
    IIRNotchECG = plotIIRNotchECG(notched_samples, notch_time) # Plot a time domain graph of the IIR notch filtered ECG data
    IIRNotchECGSpectrum = plotIIRNotchECGSpectrum(notch_frequency, notch_freq_data) # Plot the frequency spectrum of the IIR notch filtered ECG data
    IIRNotchFilterResponse = plotIIRNotchFilterResponse(notched_numerator, notched_denominator, SAMPLE_RATE) # Plot the frequency response of the notch filter

    # Plot window filtered data
    WindowedECG = plotWindowedECG(overall_windowed_samples, win_time) # Plot a time domain graph of the window filtered ECG data
    WindowedECGSpectrum = plotWindowedECGSpectrum(win_frequency, win_freq_data) # Plot the frequency spectrum of the window filtered ECG data
    WindowFilterResponse = plotWindowFilterResponse(window_filter_overall, SAMPLE_RATE) # Plot the frequency response of the window filter

    #Plot optimal filtered data
    OptimalECG = plotOptimalECG(overall_optimal_samples, opt_time) # Plot a time domain graph of the optimal filtered ECG data
    OptimalECGSpectrum = plotOptimalECGSpectrum(opt_frequency, opt_freq_data) # Plot the frequency spectrum of the optimal filtered ECG data
    OptimalFilterResponse = plotOptimalFilterResponse(optimal_filter_overall, SAMPLE_RATE) # Plot the frequency response of the optimal filter

    #Plot Frequency Sampling filtered data
    FrequencySamplingECG = plotFrequencySampledECG(overall_freq_samples, freq_sampling_time) # Plot a time domain graph of the frequency sampling filtered ECG data
    FrequencySamplingECGSpectrum = plotFrequencySampledECGSpectrum(freq_s_frequency, freq_s_freq_data) # Plot the frequency spectrum of the frequency sampling filtered ECG data
    FrequencySamplingFilterResponse = plotFrequencySampledFilterResponse(freq_filter_overall, SAMPLE_RATE) # Plot the frequency response of the frequency sampling filter

    # Save figures
    figures = [ECG, ECGSpectrum, IIRPoleZero, IIRNotchECG, IIRNotchECGSpectrum, IIRNotchFilterResponse, WindowedECG,
               WindowedECGSpectrum, WindowFilterResponse, OptimalECG, OptimalECGSpectrum, OptimalFilterResponse,
               FrequencySamplingECG, FrequencySamplingECGSpectrum, FrequencySamplingFilterResponse] # The figures to save, which must be in the same order as figure_names
    saveFigures(figures, FIGURES_FOLDER_NAME, FIGURE_NAMES) # Save the figures to an output folder in the current directory

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
    noise_power_data = {FILTER_NAMES[0]: notched_noise_variance,
                        FILTER_NAMES[1]: first_notched_noise_variance,
                        FILTER_NAMES[2]: second_notched_noise_variance,
                        FILTER_NAMES[3]: window_noise_variance,
                        FILTER_NAMES[4]: first_window_noise_variance,
                        FILTER_NAMES[5]: second_window_noise_variance,
                        FILTER_NAMES[6]: optimal_noise_variance,
                        FILTER_NAMES[7]: first_optimal_noise_variance,
                        FILTER_NAMES[8]: second_optimal_noise_variance,
                        FILTER_NAMES[9]: freq_sampling_noise_variance,
                        FILTER_NAMES[10]: first_freq_sampling_noise_variance,
                        FILTER_NAMES[11]: second_freq_sampling_noise_variance
                        }  # Create a dictionary of the filter name and its noise power
    saveNoisePowerData(noise_power_data, NOISE_POWER_OUTPUT_FILENAME)  # Save the data about each filter to a file



# Run program if called
if __name__ == '__main__':
    main()
