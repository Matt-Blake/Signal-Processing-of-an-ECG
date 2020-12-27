"""
    IIR.py (Signal Processing of an ECG program)
    Contains all the IIR filter computation functions for
    the Signal Processing of an ECG program.

    Authors: Matt Blake   (58979250)
             Reweti Davis (23200856)
    Last Modified: 27/12/2020
"""

# Imported libraries
from scipy.signal import freqz, lfilter, firwin, remez, firwin2, convolve
from scipy.fft import fft
import numpy as np


# Functions
def computeIIRNotchCoefficients(notch_freq:list, notch_width:float, sampling_freq:float) -> tuple:
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



def calculateGainFactor(numerator:list, denominator:list, passband_freq:float) -> float:
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



def createIIRNotchFilter(notch_freq:list, notch_width:float, passband_f:float, sample_rate:float) -> tuple:
    """Create and return the coefficents of an IIR notch filter"""

    numerator, denominator = computeIIRNotchCoefficients(notch_freq, notch_width, sample_rate)  # Calculate filter coefficents
    gain_factor = calculateGainFactor(numerator, denominator, passband_f) # Calculate gain factors needed to get unity gain in passband
    normalised_numerator = np.array(numerator) * gain_factor  # Normalise passband of filters to unity gain

    return normalised_numerator, denominator



def applyIIRNotchFilters(numerator_1:list, denominator_1:list, numerator_2:list, denominator_2:list, data:list) -> tuple:
    """Pass data through two cascaded IIR filters and return the result after each filter"""

    partially_filtered_data = lfilter(numerator_1, denominator_1, data) # Apply first filter to data
    filtered_data = lfilter(numerator_2, denominator_2, partially_filtered_data) # Apply second notch filter to data

    return partially_filtered_data, filtered_data



def combineFilters(numerator_1:list, denominator_1:list, numerator_2:list, denominator_2:list) -> tuple:
    """Tales the numerators and denominators of two filters and convolutes them to create an overall filter.
    The numerator and denominator of this filter are returned"""

    numerator = convolve(numerator_1, numerator_2)  # Create the overall numerator via convolution
    denominator = convolve(denominator_1, denominator_2)  # Create the overall denominator via convolution

    return numerator, denominator

