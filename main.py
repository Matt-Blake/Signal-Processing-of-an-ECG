from scipy.signal import freqz, lfilter, firwin, remez, convolve

filename = 'enel420_grp_18.txt'

data = open(filename, 'r')
XDDD = data.read()
XD = XDDD.split()

for sample in XD:
    print(float(sample))

