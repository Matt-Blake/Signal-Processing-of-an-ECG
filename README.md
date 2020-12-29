# Signal Processing of an ECG

## Description
The program uses digital filtering to remove narrowband noise from an ECG. This is done with an IIR filter (designed using pole placement) and 3 FIR filters (designed with the window method, optimal method and frequency-sampling) and the results are compared and graphed. This program was originally designed as part of ENEL420 Assignment 1.

## Author
+ Matt Blake
+ Reweti Davis

## License
Added an [MIT License](LICENSE)

## Configuration
The behavior of this program is controlled by changing constants defined in the config.py file.

## Inputs
This program requires an ECG signal with its amplitude values separated by spaces. This location and properties of this data is set by changing the following config.py constants:
```python
DATA_FILENAME = 'enel420_grp_18.txt'
SAMPLE_RATE = 1024
```

## Outputs
This program produces graphs (stored as .png images) of the time and frequency domain signals before and after filtering. Filter responses and pole-zero plots are also produced. Variance data is also calculated and saved in a .txt file, so the noise removed by each signal can be compared. These output filenames, including the directory name for figures, is set by changing the following config.py constants:
```python
FIGURES_FOLDER_NAME = 'Group_18_Figures'
FIGURE_NAMES = ['ECG_Time_Plot.png', 'ECG_Freq_Plot.png', ...]
NOISE_POWER_OUTPUT_FILENAME = 'Noise_Power_Variance_Data.txt'
FILTER_NAMES = ['IIR notch filters', 'first IIR notch filter', ...]
```
The figure names and filter names must have the same position in the above lists as they are utilised in the code. 

## Known Issues
There are currently no known issues

### Contact
If you encounter any issues or questions with the preprocessing, please contact 
Matt Blake by email at matt.blake.mb@gmail.com