"""
    config.py (Signal Processing of an ECG program)
    Used by the user to configure all inputs/outputs for the Create Preprocessed
    Dataset program. This was designed as a module so these constants could be
    seen by all the modules in the program.

    Authors: Matt Blake   (58979250)
             Reweti Davis (23200856)
    Last Modified: 27/12/2020
"""

# Define input signal parameters
DATA_FILENAME = 'enel420_grp_18.txt' # Location in project where ECG data is stored
SAMPLE_RATE = 1024  # Sample rate of data (Hz)

# Define output file parameters
FIGURES_FOLDER_NAME = 'Group_18_Figures' # Folder to save created figure images to
FIGURE_NAMES = ['ECG_Time_Plot.png', 'ECG_Freq_Plot.png', 'IIR_Pole_Zero_Plot.png', 'IIR_Notched_ECG_Time_Plot.png',
                'IIR_Notched_Freq_Plot.png', 'IIR_Frequency_Response.png', 'Windowed_ECG_Time_Plot.png',
                'Windowed_Freq_Plot.png', 'Windowed_Frequency_Response.png', 'Optimal_ECG_Time_Plot.png',
                'Optimal_Freq_Plot.png', 'Optimal_Frequency_Response.png', 'Freq_Sampled_ECG_Time_Plot.png',
                'Freq_Sampled_Freq_Plot.png', 'Freq_Sampled_Frequency_Response.png'] # The names (in order) that each figure should be saved as
NOISE_POWER_OUTPUT_FILENAME = 'Group_18_Noise_Power_(Variance)_Data_from_Created_Filters.txt' # File to save calculated noise power data
FILTER_NAMES = ['IIR notch filters', 'first IIR notch filter', 'second IIR notch filter', 'FIR Window filters',
                'first window filter', 'second window filter', 'FIR Optimal filters', 'first optimal filter',
                'second optimal filter', 'FIR Frequency Sampling filters', 'first frequency sampling filter',
                'second frequency sampling filter'] # The filter names (in order), so that variance/noise data can be saved

# Define general filter parameters
CUTOFF_FREQS = [57.755, 88.824] # Frequencies to attenuate (Hz), which were calculated based on previous graphical analysis
PASSBAND_FREQ = 10 # Passband frequency (Hz) used to calculate the gain factor
NOTCH_WIDTH = 5 # 3 dB bandwidth of the notch filters (Hz)
NUM_FIR_TAPS = 399 # The number for each FIR filter
FIR_WINDOW_TYPE = 'kaiser' # The type of window to use for the FIR filters (i.e. 'kaiser', 'blackman')

# Define optimal method parameters
OPTIMAL_PASS_WEIGHT = 1 # Passband weighting for the optimal filters
OPTIMAL_STOP_WEIGHT = 100000 # Stopband weighting for the optimal filters
OPTIMAL_WEIGHTS_SINGLE = [OPTIMAL_PASS_WEIGHT, OPTIMAL_STOP_WEIGHT, OPTIMAL_PASS_WEIGHT] # filter weighting
OPTIMAL_WEIGHTS_OVERALL = [OPTIMAL_PASS_WEIGHT, OPTIMAL_STOP_WEIGHT, OPTIMAL_PASS_WEIGHT, OPTIMAL_STOP_WEIGHT, OPTIMAL_PASS_WEIGHT] # Overall filter weighting
OPTIMAL_GAINS_SINGLE = [1, 0, 1] # The stopband and passband locations gains for the single stopband filters
OPTIMAL_GAINS_OVERALL = [1, 0, 1, 0, 1] # The stopband and passband locations gains for the overall filter
OPTIMAL_NOTCH_SPACING = 0.001 # Spacing of stop band notch to allow convergence when using the optimal method

# Define frequency sampling parameters
FREQ_SAMP_GAINS_SINGLE = [1, 1, 0, 0, 0, 0, 0, 1, 1] # The stopband and passband locations gains for the single stopband filters
FREQ_SAMP_GAINS_OVERALL = [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1]  # The stopband and passband locations gains for the overall filter
FREQ_SAMP_WIDTH = 0.1 # Frequency sampling width for stopband and passband
FREQ_SAMP_TRANSITION_WIDTH = 0.01 # Frequency sampling width which defines width of the transition band 

# Define signal plot parameters
TIME_LINE_WIDTH = 0.5 # The linewidth of the time domain plots
SPECTRUM_LINE_WIDTH = 0.2 # The linewidth of the signal spectrum plots
FILTER_LINE_WIDTH = 1.7 # The linewidth of filter response plots
POLE_ZERO_FIG_SIZE = (6, 6) # The x and y sizes of pole-zero plots produced
POLE_ZERO_MARKER_SIZE = 8 # The size of the markers used to indicate poles/zeros on a pole-zero plot
POLE_ZERO_MARKER_WIDTH = 2 # The edge width of the markers used to indicate poles/zeros on a pole-zero plot