"""
    configFiles.py
    Contains all tfile formatting and configuration files functions for ENEL420-20S2 Assignment 1.

    Authors: Matt Blake   (58979250)
             Reweti Davis (23200856)
             Group Number: 18
    Last Modified: 14/08/2020
"""
import os
import shutil

#
# File functions
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




