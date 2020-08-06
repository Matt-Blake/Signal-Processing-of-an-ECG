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


def main():
    """Main function of ENEL420 Assignment 1"""

    filename = 'enel420_grp_18.txt' # Location in project where data is stored

    samples = importData(filename) # Import data from file
    plt.plot(samples)
    plt.show()


main()