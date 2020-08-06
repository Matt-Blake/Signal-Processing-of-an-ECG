from scipy.signal import freqz, lfilter, firwin, remez, convolve
import matplotlib.pyplot as plt

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
    num_samples = len(samples) # Calculate the number of ECG samples
    time = getTimeData(sample_rate, num_samples) # Create an array containing the time each sample was taken

    # Plot ECG
    plt.figure(1)
    plt.plot(time, samples)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV)")
    plt.suptitle("Time domain ECG signal")
    plt.xlim(time[0], time[-1]) # Limit the x axis to locations with data points

def plotECGSpectrum(samples):
    return 0

def main():
    """Main function of ENEL420 Assignment 1"""

    filename = 'enel420_grp_18.txt' # Location in project where ECG data is stored
    sample_rate = 1024  # Sample rate of data (Hz)

    samples = importData(filename) # Import data from file
    plotECG(samples, sample_rate) # Plot a time domain graph of the ECG data

    plt.show()  # Display


    plt.show()



main()