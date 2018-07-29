"""
Week01 of the Astro MOOC!
"""
from statistics import mean
from statistics import median
import numpy as np
import csv
import random

# need a function to strip " from the data files as they are now
def clean_file(csv_file):
    """
        > Part One
        Given a CSV file, read the data into a nested list
        Input: String corresponding to comma-separated  CSV file
        Output: Lists of lists consisting of the fields in the CSV file
        """
    my_list = []
    with open(csv_file, newline='') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',', quotechar=" ")
        for row in file_reader:
            my_list.append(row)

    """
    > Part Two
    Input: Nested list csv_table and a string file_name
    Action: Write fields in csv_table into a comma-separated CSV file with the name file_name
    Mutates output: Yes
    """
    with open(csv_file, 'w', newline='') as csvfile:
        my_csv_writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
        for row in my_list:
            row2 = []
            for item in row:
                a = item.lstrip('"')
                b = a.rstrip('"')
                row2.append(b)
            my_csv_writer.writerow(row2)

clean_file('/Users/RAhmed/Astro_MOOC/data.csv')
clean_file('/Users/RAhmed/Astro_MOOC/data1copy.csv')
clean_file('/Users/RAhmed/Astro_MOOC/data2copy.csv')
clean_file('/Users/RAhmed/Astro_MOOC/data3copy.csv')
# ok, now all files good to go!
# showing off mean and median function from statistics
fluxes = [23.3, 42.1, 2.0, -3.2, 55.6]
m = mean(fluxes)
n = median(fluxes)
# print(m, n)

# showing classic csv function in Python
data = []
for line in open('/Users/RAhmed/Astro_MOOC/data.csv'):
    data.append(line.strip().split(','))

# print(data)

# turning csv into float
data = []
for line in open('/Users/RAhmed/Astro_MOOC/data.csv'):
    row = []
    for item in line.strip().split(','):
        row.append(float(item))
        data.append(row)

# print(data)

# numpy can do (e.g.) float conversion in one go, no need to iterate!
# and you get a matrix-like output
data = []
for line in open('data.csv'):
    data.append(line.strip().split(','))

data = np.asarray(data, float)
# print(data)

# numpy is also easier/better for loading csv files, already an array it seems?
data = np.loadtxt('data.csv', delimiter=',')
# print(data)

# showing what numpy can do
# more numpy examples
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
# Element-wise multiplication
print(a*2)
# Element-wise summation
print(a + b)
# Element-wise product
print(a*b)
a = np.array([[1,2,3], [4,5,6]])  # 2x3 array
# Print first row of a:
print(a[0,:])
# Print second column of a:
print(a[:,1])


# > task: write a calc_stats function that outputs mean and median rounded to 1dp of data csv file
def calc_stats(csvfile):
    """
    > Input:
    a csv file
    > Output:
    mean and median rounded to 1dp of the csvfile
    > Idea:
    Only use numpy for everything. N.B. statistics module functions can't do (e.g.)
    mean over more than a single row
    > Mutates input?:
    No
    """
    data = np.loadtxt(csvfile, delimiter=',')
    m = np.mean(data)
    m = np.ma.round(m, decimals=1)
    n = np.median(data)
    n = np.ma.round(n, decimals=1)
    return m, n


z = calc_stats('/Users/RAhmed/Astro_MOOC/data.csv')
# print("output of calc_stats:", z)





# > task: write a mean_data_Sets function that reads in a list of CSV files and
#  returns an array of the mean of each cell in the data files.
def mean_datasets(csv_file_list):
    """
    > Input:
    a list of csv files
    > Output:
    an array of means of each cell in list (same dimensions as files in the input list)
    > Idea:
    From internet, it looks much easier to use list first, then turn that into an np array
    > Mutates input?:
    No
    """
    # find dimensions of the data files in the list
    search = np.loadtxt(csv_file_list[0], delimiter=',')

    # find rows and columns
    dims = search.shape

    # create empty array
    output = np.zeros(dims)

    # now need to iterate across each file adding them to 'output' array
    for file in csv_file_list:
        data = np.loadtxt(file, delimiter=',')
        output += data

    # divide output by length of csv_file_list
    output = output/len(csv_file_list)
    output = np.round_(output, decimals=1)
    return output


csv_file_list = ['/Users/RAhmed/Astro_MOOC/data1copy.csv', '/Users/RAhmed/Astro_MOOC/data2copy.csv', '/Users/RAhmed/Astro_MOOC/data2copy.csv']
z = mean_datasets(csv_file_list)
# print(z)

# > task: need to open a fits file that stores astro images, that uses the astropy module
from astropy.io import fits
import matplotlib.pyplot as plt
hdulist = fits.open('/Users/RAhmed/Astro_MOOC/814wmos.fits')
# hdulist.info()


# > task: now do all the opening as a function, and return the coordinates of the brightest pixel
def load_fits(fits_file):
    """
    > Input:
    A FITS astronomical file
    > Output:
    Coordinates of the brightest point in the image
    > Idea:
    Basically you have to look up how Astropy works, else makes no sense:
    http://www.astropy.org/astropy-tutorials/FITS-images.html
    Also, does the horrible thing the Astro MOOC does, of importing modules inside the function
    > Mutates input?:
    No
    """
    import numpy as np
    from astropy.io import fits
    hdulist = fits.open(fits_file)
    image_data = hdulist[0].data
    #hdulist.info()
    return np.unravel_index(hdulist[0].data.argmax(), image_data.shape)

bright = load_fits('814wmos.fits')
# print(bright)

# You can also confirm your result visually:
hdulist = fits.open('814wmos.fits')
data = hdulist[0].data

# Plot the 2D image data
plt.imshow(data.T, cmap=plt.cm.viridis)
plt.colorbar()
# plt.show()


import numpy as np
from astropy.io import fits


# > task: now do all the opening as a function, and return the coordinates of the brightest pixel
def mean_fits(fits_file_list):
    """
    > Input:
    Takes a list of FITS files as an argument, reads them in, and returns the mean image
    data of the FITS files.
    > Output:
    The mean of the centre position; all images have same centre
    > Idea:
    Basically you have to look up how Astropy works, else makes no sense:
    http://www.astropy.org/astropy-tutorials/FITS-images.html
    > Mutates input?:
    No
    > Issues: when you open all the FITS files, I don't actually close the files
    """
    # from astropy.io import fits
    # create zero array same dimension as the files
    search = fits.open(fits_file_list[0])
    image_data = search[0].data

    # find rows and columns
    dims = image_data.shape

    # create empty array
    output = np.zeros(dims)

    # iterate through files and add to output array
    for fits_file in fits_file_list:
        fits_data = fits.open(fits_file)
        output += fits_data[0].data
        # ^ issue here is that file isn't actually closed. See median_fits function below

    # divide output by length of fits_file_list
    output = output / len(fits_file_list)
    return output


fits_file_list = ['/Users/RAhmed/Astro_MOOC/fits_images_all/image0.fits', '/Users/RAhmed/Astro_MOOC/fits_images_all/image1.fits', '/Users/RAhmed/Astro_MOOC/fits_images_all/image2.fits']
# VVV N.B. the way yu can determine the position in the array that is printed VVV
z = mean_fits(fits_file_list)[100,100]
print(z)


# showing the median function in statistics module
from statistics import median
fluxes = [17.3, 70.1, 22.3, 16.2, 20.7]
m = median(fluxes)
print(m)
# without statistics module you would have to sort list and then find mid point dependieng whether odd or even length


# task:
def list_stats(list0):
    """
    > Input: takes a single list of numbers
    > Output: returns tupe of mean, median
    > Idea: do not use statistics function
    > Mutates input?: no
    """
    list0.sort()
    sum = 0
    for num in list0:
        sum += num
    mean = sum/len(list0)
    if not len(list0) % 2 == 0:
        mid = len(list0) // 2
        median = list0[mid]
    else:
        mid = len(list0) // 2
        median = (list0[mid - 1] + list0[mid]) / 2

    return median, mean

list0 = [1965, 1963, 2002, 2006]
z = list_stats(list0)
# print(z)


# timing my code above
import time
start = time.perf_counter()
# potentially slow computation
list0 = [1965, 1963, 2002, 2006]
z = list_stats(list0)
print(z)
end = time.perf_counter() - start
print("time for my code: ",end)
# ^ uses a function called time.perf_counter, that uses accurate device clock


# a race between "manual" mean and numpy mean
import time, numpy as np
n = 10**7
data = np.random.randn(n)
# manual
start = time.perf_counter()
mean = sum(data)/len(data)
seconds = time.perf_counter() - start
print('That took {:.2f} seconds.'.format(seconds))

#numpy
start = time.perf_counter()
mean = np.mean(data)
seconds = time.perf_counter() - start
print('That took {:.2f} seconds.'.format(seconds))


# > task: create a timestat function
import numpy as np
import statistics
import time


def time_stat(func, size, ntrials):
    """
    > Input: three arguments:
    the func function we're timing,
    the size of the random array to test,
    and the number of experiments to perform.
    > Output:
    Return the average running time for the func function.
    > Idea: different random array for each trial
    > Mutates input:
    No.
    """
    cum_time = 0
    for n in range(ntrials):
        # the time to generate the random array should not be included
        data = np.random.rand(size)
        # modify this function to time func with ntrials times using a new random array each time
        start = time.perf_counter()
        res = func(data)
        seconds = time.perf_counter() - start
        cum_time += seconds

    # return the average run time
    return cum_time/ntrials

# test here
print('{:.6f}s for statistics.mean'.format(time_stat(statistics.mean, 10**5, 10)))
print('{:.6f}s for np.mean'.format(time_stat(np.mean, 10**5, 1000)))


# investigate memory usage
import sys

# normal Python
# note that the Astro MOOC's Grok uses half the memory here (and other IDEs similar to this)
# Probably above is due to Grok using 32-bit architecture.
a = 3
b = 3.123
c = [a, b]
d = []
for obj in [a, b, c, d]:
    print(obj, sys.getsizeof(obj))

# in numpy
import sys
import numpy as np

a = np.array([])
b = np.array([1, 2, 3])
c = np.zeros(10**6)

# here the sys part uses mor ememory than just the numpy part, as it also contains nump metadata
for obj in [a, b, c]:
    print('sys:', sys.getsizeof(obj), 'np:', obj.nbytes)

# this part is about getting just the object nbytes
a = np.zeros(5, dtype=np.int32)
b = np.zeros(5, dtype=np.float64)

for obj in [a, b]:
    print('nbytes         :', obj.nbytes)
    print('size x itemsize:', obj.size*obj.itemsize)

# A FITS 200*200 array of type float32 (=4bytes) is this size:
print(200*200*4/1024, "kBs")
# A FITS 10000*10000 array of type float32 (=4bytes) is this size:
print(10000*10000*4/1024/1000, "mBs")

# > task: write median_fits function:
def median_fits(fits_file_list):
    """
    > Input: list of FITS filenames, loads them into a NumPy array, and calculates the median image
    (where each pixel is the median of that pixel over every FITS file).
    > Output: tuple of median NumPy array, time for function to run, memory (in kB) used to store all the FITS files
    in the NumPy array in memory.
    > Note: The running time should include loading the FITS files and calculating the median.
    Also, will use NEW TECHNIQUE of numpy dstack, for 3D stack!
    """
    # wrapper for timing start
    start = time.perf_counter()

    # initially just get the FITS data you want from the files, and put then in a list
    fits_data = []
    for fits_file in fits_file_list:
        with fits.open(fits_file) as hdulist:
            fits_data.append(hdulist[0].data)

    # now need to create a 3D-array of all FITS data. Uses the numpy dstack command, convert to median
    # data_stack = np.dstack(fits_data)
    # ^ v combining commands to be more meomory efficient
    median_stack = np.median(np.dstack(fits_data), axis=2)

    # now need to calculate memory used in kBs (only concentrating on largest part)
    memory = median_stack.nbytes / 1024

    # stop timing
    seconds = time.perf_counter() - start

    # function output
    return median_stack, seconds, memory

"""
Some VERY NEW, GREAT of outputing results are given in the Astro MOOC, e.g.:

result = median_fits(['image0.fits', 'image1.fits'])
# ^ this would output the whole stack!
# v this says print result [0] for pixel [100,100]
print(result[0][100, 100], result[1], result[2])
"""

# > task: write median_bins_fits function:
# create a values_list for testing
values_list = [random.gauss(1000, 200) for _ in range(500)]


def median_bins(values_list, numbins):
    """
    > Idea: this is an internal helper function that just works for a list of values
    Create median_bins to calculate the mean, standard deviation and the bins (steps 1-6 of the below)
    1. Calculate their mean and standard deviation, mu and sigma;
    2. Set the bounds: minval =  mu - sigma and maxval = mu + sigma. Any value >= maxval is ignored;
    3. Set the bin width: width = 2*sigma/numbins;
    4. Make an ignore bin for counting value < minval;zeros
    5. Make  bins for counting values in minval and maxval, e.g. the first bin is minval <= value < minval + width;
    6. Count the number of values that fall into each bin;
    > Input: list of numbers and a desired number of bins
    > Output: i.a.w. the algorithm, returns mean, sd, number of values less than minval, and bin count array
    >Mutates input?: No
    """
    mu = np.mean(values_list)
    sigma = np.std(values_list)
    minval = mu - sigma
    maxval = mu + sigma
    bin_width = 2*sigma/numbins
    ignore_vals = 0
    bins = np.zeros(numbins)

    for value in values_list:
        if value < minval:
            ignore_vals += 1
        elif value < maxval:
            somenum = int((value - minval)/bin_width)
            bins[somenum] += 1
        # value > maxval is ignored

    return mu, sigma, ignore_vals, bins

z = median_bins([1, 1, 3, 2, 2, 6], 3)
# print(z)

# > task: write median_bins_fits function:
def median_approx(values_list, numbins):
    """
    THIS FUNCTION DOESN'T WORK PROPERLY - DOESN'T WORK ABOVE A CERTAIN LIST SIZE
    HAVEN'T DEBUGGED TO FIND OUT WHY SINCE AN ANSWER IS GIVEN
    > Idea: uses median_bins helper function, above
    Create median_bins to calculate the mean, standard deviation and the bins (additional steps 7-8 of the below)
    7. Sum these counts until total >= (N + 1)/2. Remember to start from the ignore bin;
    8. Return the midpoint of the bin that exceeded (N + 1)/2.
    > Input: list of numbers and a desired number of bins
    > Output: i.a.w. the algorithm, returns mean, sd, number of values less than minval, and bin count array
    >Mutates input?: No
    """
    # call median_bins function: really nice technique here
    mu, sigma, ignore_vals, bins = median_bins(values_list, numbins)
    bin_width = 2 * sigma / numbins

    # now get to midpoint of ... Adjust to include ignore_vals
    N = ignore_vals + np.sum(bins)
    midpoint = (N + 1)/2
    count = ignore_vals
    for b, bincount in enumerate(bins):
        while count < midpoint:
            count += bincount

    # find midpoint of bin in countB
    median = (mu - sigma) + (b - 0.5) * bin_width
    return median

z = median_bins([1, 5, 7, 7, 3, 6, 1], 4)
print("{}", z)
# zz = median_approx([1, 1, 3, 2, 2, 6], 3)
zz = median_approx([1, 5, 7, 7, 3, 6, 1], 4)  # <= THIS DOESN'T WORK
print(zz)
print("Cheese")


def median_approx2(values, B):
    """
    Largely used model answer idea of 'enumerate', and (b+0.5) not (b-0.5)
    """
    # Call median_bins to calculate the mean, std,
    # and bins for the input values
    mean, std, left_bin, bins = median_bins(values, B)

    # Position of the middle element
    N = len(values)
    mid = (N + 1) / 2

    count = left_bin
    for b, bincount in enumerate(bins):
        count += bincount
        if count >= mid:
            # Stop when the cumulative count exceeds the midpoint
            break

    width = 2 * std / B
    median = mean - std + width * (b + 0.5)
    return median

zz = median_approx2([1, 5, 7, 7, 3, 6, 1, 1], 4)
# print(zz)

"""
Helper function: given in the course. Really interesting np array functions.
Basically, outputs 2 arrays: one of SDs of each cellm, one of means across each cell
"""
def running_stats(filenames):
  '''Calculates the running mean and stdev for a list of FITS files using Welford's method.'''
  n = 0
  for filename in filenames:
    hdulist = fits.open(filename)
    data = hdulist[0].data
    if n == 0:
      mean = np.zeros_like(data)
      s = np.zeros_like(data)

    n += 1
    delta = data - mean
    mean += delta/n
    s += delta*(data - mean)
    hdulist.close()

  s /= n - 1
  np.sqrt(s, s)

  if n < 2:
    return mean, None
  else:
    return mean, s

filenames = ['/Users/RAhmed/Astro_MOOC/fits_images_all/image0.fits', '/Users/RAhmed/Astro_MOOC/fits_images_all/image1.fits', '/Users/RAhmed/Astro_MOOC/fits_images_all/image2.fits']
z = running_stats(filenames)
print(z[0].shape)
print(z[1].shape)


def median_bins_fits(filenames, numbins):
    """
    > Input: list of FITS image files, and the number of bins you want to you bin approximation technique to stack
    FITS pictures
    > Output:
    - an array of all cell means (mu), another of all cell SDs (sigma); these are from the helper function
    - an array of 'ignore numbers (in each cell) for numbers less than min_val (needed for these algorithm)
    - a 3D(!) 'bins' array that counts how many in each bin. Bins are given by 'numbins' input that user chooses
    - ^ the above is is to find the median bin in the subsequent median_approx_fits function, in similar way to
    previous functions here in week one
    > Idea: call a (given on course) helper function to work out 'running' means and SDs of each cell array as arrays
    are called in iteratively
    > Mutates inputs: no
    """
    # use helper function ot get mean and sd of each cell for all arrays
    mu, sigma = running_stats(filenames)

    # now need to create a bins arrays of same size as the two above
    dims = mu.shape

    # Initialise bins
    ignore = np.zeros(dims)
    # now the clever part - some new numpy features. Create a 3D numpy array size dim[0]*dim[1]*bumbins
    bins = np.zeros((dims[0], dims[1], numbins))
    bin_width = 2 * sigma / numbins

    # Loop over all FITS files
    for filename in filenames:
        hdulist = fits.open(filename)
        data = hdulist[0].data

        # Loop over every point in the 2D array
        for i in range(dims[0]):
            for j in range(dims[1]):
                value = data[i, j]
                min_val = mu[i, j] - sigma[i, j]
                max_val = mu[i, j] + sigma[i, j]

                if value < min_val:
                    ignore[i, j] += 1

                elif value >= min_val and value < max_val:
                    bin = int((value - (min_val)) / bin_width[i, j])
                    bins[i, j, bin] += 1

    return mu, sigma, ignore, bins


def median_approx_fits(filenames, numbins):
    """
    > Input: list of FITS image files, and the number of bins you want to you bin approximation technique to stack
    FITS pictures for. Through the median_bins_fits function called in this function at beginning,
    this gives the following arrays:
    - mu: the average number of readings in that cell over all arrays
    - sigma: the sd of readings in that cell over all arrays
    - ignore: the number of readings lower than the min_val in that cell over all arrays
    - bins: a 3D array, where the depth is given by 'numbins' chosen by user
    (You need to also count in the ignore values when finding median.)
    > Output:
    - an array called 'median' that has the median number of readings in that cell over all arrays.
    > Idea:
    1. The maximum number of total counts over all bins in the 3D bins array is same as the number of files.
    2. Given your actual total count in a cell, you need 'N', the number of counts
    > Mutates inputs: no
    """
    mu, sigma, ignore, bins = median_bins_fits(filenames, numbins)
    # ^ actually you might not need all of the above

    # find dimension of arrays in the FITS files
    dims = mu.shape

    # create your 'median' output array
    median = np.zeros(dims)

    # need to find midpoint in each of the 3D array for each cell, to 'count' to the median
    N = len(filenames)
    midpoint = (N + 1) / 2

    # use ability of numpy to operate across whole arrays. bin_width is an array
    bin_width = 2 * sigma / numbins

    # 3D-iteration over each (i,j,bin cell) in bins array
    for i in range(dims[0]):
        for j in range(dims[1]):
            count = ignore[i, j]
            for b, bincount in enumerate(bins[i, j]):
                count += bincount
                if count >= midpoint:
                    # Stop when the cumulative count exceeds the midpoint
                    break

            median[i, j] = mu[i, j] - sigma[i, j] + bin_width[i, j] * (b + 0.5)

    return median


filenames = ['/Users/RAhmed/Astro_MOOC/fits_images_all/image0.fits', '/Users/RAhmed/Astro_MOOC/fits_images_all/image1.fits', '/Users/RAhmed/Astro_MOOC/fits_images_all/image2.fits']
z = median_bins_fits(filenames, 5)
a = median_approx_fits(filenames, 5)
print(a[100, 100])







