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
NOISE_POWER_OUTPUT_FILENAME = 'Group_18_Noise_Power_(Variance)_Data_from_Created_Filters.txt' # File to save calculated noise power data
FIGURE_NAMES = ['ECG_Time_Plot.png', 'ECG_Freq_Plot.png', 'IIR_Pole_Zero_Plot.png', 'IIR_Notched_ECG_Time_Plot.png',
                'IIR_Notched_Freq_Plot.png', 'IIR_Frequency_Response.png', 'Windowed_ECG_Time_Plot.png',
                'Windowed_Freq_Plot.png', 'Windowed_Frequency_Response.png', 'Optimal_ECG_Time_Plot.png',
                'Optimal_Freq_Plot.png', 'Optimal_Frequency_Response.png', 'Freq_Sampled_ECG_Time_Plot.png',
                'Freq_Sampled_Freq_Plot.png', 'Freq_Sampled_Frequency_Response.png']  # The names that each figure should be saved as

# Define filter parameters
CUTOFF_FREQS = [57.755, 88.824] # Frequencies to attenuate (Hz), which were calculated based on previous graphical analysis
PASSBAND_FREQ = 10 # Passband frequency (Hz) used to calculate the gain factor
NOTCH_WIDTH = 5 # 3 dB bandwidth of the notch filters (Hz)
NUM_FIR_TAPS = 399 # The number for each FIR filter
FIR_WINDOW_TYPE = 'kaiser' # The type of window to use for the FIR filters (i.e. 'kaiser', 'blackman')

# Signal plot parameters
TIME_LINE_WIDTH = 0.5 # The linewidth of the time domain plots
SPECTRUM_LINE_WIDTH = 0.2 # The linewidth of the signal spectrum plots
FILTER_LINE_WIDTH = 1.7 # The linewidth of filter response plots