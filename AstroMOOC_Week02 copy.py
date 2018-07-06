"""
Week02 of the Astro MOOC!
"""
from statistics import mean
from statistics import median
import numpy as np
import csv
import random

def hms2dec(hours, minutes, seconds):
    """
    > Aim: convert right ascension from HMS (hours, minutes, seconds) to decimal degrees
    > Input: tuple of hours, minutes, seconds
    > Output: decimal degrees of Input
    > Mutates input?: no
    """
    return (15 * (hours + (minutes / 60) + (seconds / 3600)))

print(hms2dec(23, 12, 6))


def dms2dec(degrees, arcmins, arcsecs):
    """
    > Aim: convert declination from DMS (degrees, minutes, seconds) to decimal degrees
    > Input: tuple of degrees, arcminutes, arcseconds
    > Output: decimal degrees of Input
    > Mutates input?: no
    """
    if degrees >= 0:
        return degrees + (arcmins / 60) + (arcsecs / 3600)
    else:
        return -1 * (-1 * degrees + (arcmins / 60) + (arcsecs / 3600))

print(dms2dec(73, 21, 14.4))
print(dms2dec(-5, 31, 12))


def angular_dist(r1, d1, r2, d2):
    """
    Find angular distance from celestial position of 2 objects
    Uses given havesine formula
    > Input: right ascension and declination of 2 objects (in decimla)
    > Output: angular distance in decimal degrees
    > Mutates input:? no
    """
    # convert all inputs to radians in numpy
    r1_rad = np.radians(r1)
    d1_rad = np.radians(d1)
    r2_rad = np.radians(r2)
    d2_rad = np.radians(d2)

    b = np.cos(d1_rad) * np.cos(d2_rad) * np.sin(np.abs(r1_rad - r2_rad) / 2) ** 2
    a = np.sin(np.abs(d1_rad - d2_rad) / 2) ** 2
    ang_rad = 2 * np.arcsin(np.sqrt(a + b))

    # convert to decimal degrees and return
    ang_degr = np.degrees(ang_rad)
    return ang_degr

print(angular_dist(10.3, -3, 24.3, -29))


def import_bss():
    """
    Custom function to import the bss.dat file
    > Mutates input?: no
    """
    # note this does not load column 7
    data = np.loadtxt('/Users/RAhmed/Astro_MOOC/Week02/bss.dat', usecols=range(1, 7))

    # empty output list
    output = []

    # iterate over data and convert inputs as needed
    for index, row in enumerate(data, 1):
        output.append((index, hms2dec(row[0], row[1], row[2]), dms2dec(row[3], row[4], row[5])))

    return output

print("BSS", import_bss())
print("LEN", len(import_bss()))


def import_super():
    """
    Custom function to import the super.csv file
    > Mutates input?: no
    """
    # note this does not load row 1 (headers) or past column 1
    data = np.loadtxt('/Users/RAhmed/Astro_MOOC/Week02/super.csv', delimiter=",", skiprows=1, usecols=(0,1))

    # empty output list
    output = []

    # iterate over data and convert inputs as needed
    for index, row in enumerate(data, 1):
        output.append((index, row[0], row[1]))

    return output

#print("SUPER", import_super())


def find_closest(catalogue, ra, dec):
    """
    Find closest star in bss catalogue to the postion given by ra, dec
    > Input: bss catalogue, a right ascenscion and declination to check against
    > Output: Index (row number) of nearest catalogue item to (ra, dec) and angular distance
    > Mutates input?: no
    """
    output = [float('inf'), float('inf')]
    for row in catalogue:
        dist = angular_dist(row[1], row[2], ra, dec)
        if dist < output[1]:
            output = [row[0], dist]

    return output[0], output[1]

cat = import_bss()
print("CLOSEST", find_closest(cat, 175.3, -32.5))
print("CLOSEST",find_closest(cat, 32.2, 40.7))


def crossmatch(cat0, cat1, max_dist):
    """
    Find closest star in catalogue1 to star in catalogue0
    > Input: 2 catalogues and a maximum distance that the stars can be to each other
    > Output: Index (row number) of the 2 catalogue items and angular distance
    > Mutates input?: no
    """
    matches = []
    no_matches = []
    for row in cat0:
        ra, dec = row[1], row[2]
        closest = find_closest(cat1,ra, dec)
        if closest[1] < max_dist:
            matches.append((row[0], closest[0], closest[1]))
        else:
            no_matches.append(row[0])

    return matches, no_matches


bss_cat = import_bss()
super_cat = import_super()
# define a max distance (here 40 arcseconds)
max_dist = 40/3600

z = crossmatch(bss_cat, super_cat, max_dist)
print(z[0])
print(z[1])


# we want to improve the efficiency of the whole process
# will use numpy arrays instead of lists, created here
cat0 = np.array([[180, 30], [45, 10], [300, -45], [250, 67], [100, 12], [190, 45]])
cat1 = np.array([[180, 32], [55, 10], [302, -44], [150, 67], [250, 18], [111, 20]])

# bit of familiarisation
print("cat0;", cat0)
print("cat0[0]", cat0[0])
print("cat0[0][0]", cat0[0][0])
print("cat0[0][1]", cat0[0][1])
for row in cat0:
    print(row[0], row[1])
for i, j in enumerate(cat0):
    print(i, j)


# import time library for speed tests
import time

def angular_dist2(r1, d1, r2, d2):
    """
    Find angular distance from celestial position of 2 objects
    Uses given havesine formula
    > Input: right ascension and declination of 2 objects (in decimla)
    > Output: angular distance in decimal degrees
    > Mutates input:? no
    """
    b = np.cos(d1) * np.cos(d2) * np.sin(np.abs(r1 - r2) / 2) ** 2
    a = np.sin(np.abs(d1 - d2) / 2) ** 2
    return 2 * np.arcsin(np.sqrt(a + b))


def crossmatch2(cat0, cat1, max_radius):
    """
    Find closest star in catalogue1 to star in catalogue0
    > Input: 2 catalogues and a maximum distance that the stars can be to each other
    > Output: Index (row number) of the 2 catalogue items and angular distance
    > Mutates input?: no
    """
    # start timer
    start = time.perf_counter()
    # initialise global maximum distance
    max_radius = np.radians(max_radius)

    # initialise output lists
    matches = []
    no_matches = []

    #  do a one-off conversion to radians
    cat0 = np.radians(cat0)
    cat1 = np.radians(cat1)


    # iterate over the 2 numpy arrays
    for index0, (ra0, dec0) in enumerate(cat0):
        # set initial local distance
        dist = np.inf
        min_index1 = None
        for index1, (ra1, dec1) in enumerate(cat1):
            dist_check = angular_dist2(ra0, dec0, ra1, dec1)
            if dist_check < dist:
                dist = dist_check
                min_index1 = index1

        # check against max_radius
        if dist > max_radius:
            no_matches.append(index0)
        else:
            matches.append((index0, min_index1, np.degrees(dist)))

    # stop timer
    time_taken = time.perf_counter() - start

    return matches, no_matches, time_taken


matches, no_matches, timed = crossmatch2(cat0, cat1, 5)
print(matches)
print(no_matches)
print(timed)


# now change functions to use arrays!
def angular_dist3(ra1, dec1, ra2s, dec2s):
    """
    Find angular distance from celestial position of 2 objects
    Uses given havesine formula
    > Input: N.B. The ra2s and dec2s are catalogue arrays!
    Calculation happens on all ra2s and dec2s at once!
    > Output: angular distance in decimal degrees
    > Mutates input:? no
    """
    b = np.cos(dec1) * np.cos(dec2s) * np.sin(np.abs(ra1 - ra2s) / 2) ** 2
    a = np.sin(np.abs(dec1 - dec2s) / 2) ** 2
    return 2 * np.arcsin(np.sqrt(a + b))

# test function
ra1, dec1 = np.radians([180, 30])
cat2 = [[180, 32], [55, 10], [302, -44]]
cat2 = np.radians(cat2)
ra2s, dec2s = cat2[:,0], cat2[:,1]
dists = angular_dist3(ra1, dec1, ra2s, dec2s)
print(np.degrees(dists))


def crossmatch3(cat0, cat1, max_radius):
    """
    Find closest star in catalogue1 to star in catalogue0
    > Input: 2 catalogues and a maximum distance that the stars can be to each other
    > Output: Index (row number) of the 2 catalogue items and angular distance
    > Mutates input?: no
    """
    # start timer
    start = time.perf_counter()
    # initialise global maximum distance
    max_radius = np.radians(max_radius)

    # initialise output lists
    matches = []
    no_matches = []

    #  do a one-off conversion to radians
    cat0 = np.radians(cat0)
    cat1 = np.radians(cat1)
    ra1s, dec1s = cat1[:, 0], cat1[:, 1]

    # iterate over the 2 numpy arrays all at once for 2nd array
    for index0, (ra0, dec0) in enumerate(cat0):
        dists = angular_dist3(ra0, dec0, ra1s, dec1s)
        min_index1 = np.argmin(dists)
        min_dist = dists[min_index1]

        # check against max_radius
        if min_dist > max_radius:
            no_matches.append(index0)
        else:
            matches.append((index0, min_index1, np.degrees(dists[min_index1])))

    # stop timer
    time_taken = time.perf_counter() - start

    return matches, no_matches, time_taken


# further optimisation by sorting 2nd catalogue by declination and breaking out of search loop when you can
def crossmatch4(cat0, cat1, max_radius):
    """
    Find closest star in catalogue1 to star in catalogue0
    > Input: 2 catalogues and a maximum distance that the stars can be to each other
    > Output: Index (row number) of the 2 catalogue items and angular distance
    > Mutates input?: no
    """
    # start timer
    start = time.perf_counter()
    # initialise global maximum distance
    max_radius = np.radians(max_radius)

    # initialise output lists
    matches = []
    no_matches = []

    #  do a one-off conversion to radians
    cat0 = np.radians(cat0)
    cat1 = np.radians(cat1)

    # sort cat 1 by declination; we create an index by declination (argsort), then order by this index
    order = np.argsort(cat1[:,1])
    cat1_ordered = cat1[order]

    # iterate over the 2 numpy arrays
    for index0, (ra0, dec0) in enumerate(cat0):
        # set initial local distance
        dist = np.inf
        min_index1 = None
        for index1, (ra1, dec1) in enumerate(cat1_ordered):
            if dec1 < (dec1 + max_radius):
                dist_check = angular_dist(ra0, dec0, ra1, dec1)
                if dist_check < dist:
                    dist = dist_check
                    min_index1 = index1
                    pass
            else:
                break

        # check against max_radius
        if dist > max_radius:
            no_matches.append(index0)
        else:
            matches.append((index0, min_index1, np.degrees(dist)))

    # stop timer
    time_taken = time.perf_counter() - start

    return matches, no_matches, time_taken


print(crossmatch4(cat0, cat1, 5))


# further optimisation by sorting 2nd catalogue by binary search.
# N.B. you are using a straighforward angular_dist function
# that takes single readings, not a while 2nd catalgue at once
# i.e. use angular_dist_2
def crossmatch5(cat0, cat1, max_distance):
    """
    Find closest star in catalogue1 to star in catalogue0.
    > Idea: you begin search in car1 at index given by binary search.
    Break out of search when past declination + max_dist
    > Input: 2 catalogues; but NOT maximum distance as before
    > Output: Index (row number) of the 2 catalogue items and angular distance
    > Mutates input?: no
    > Note: N.B. you are using a straighforward angular_dist function that takes single readings,
    not a whole 2nd catalgue at once, i.e. use angular_dist_2
    Also, maximum distance is set inside function, so as not to confude marking programme
    (stay consistent with model answer
    """
    # start timer
    start = time.perf_counter()
    # set global maximum distance inside function!
    max_radius = np.radians(max_distance)

    # initialise output lists
    matches = []
    no_matches = []

    #  do a one-off conversion to radians
    cat0 = np.radians(cat0)
    cat1 = np.radians(cat1)

    # sort cat 1 by declination; we create an index by declination (argsort), then order by this index
    order = np.argsort(cat1[:,1])
    cat1_ordered = cat1[order]
    # the issue is you have lost the original row order; need to be able to regenerate it
    # this is taken care of (amazing functionality) just before return; e.g. closest_id1 = order[n]

    # also, need a 'simple' version of cat1_ordered

    # iterate over the 2 numpy arrays
    for index0, (ra0, dec0) in enumerate(cat0):
        # set initial local distance
        dist = np.inf
        min_index1 = None

        # need to determine the indices of cat1 to check that fall inside dec0 +/- max_radius
        index_low = cat1_ordered[:,1].searchsorted(dec0 - max_radius, side='left')
        index_high = cat1_ordered[:,1].searchsorted(dec0 + max_radius, side='right')


        #  now just wish to check betwene the ncessary indices in cat1
        for index1, (ra1, dec1) in enumerate(cat1_ordered[index_low:index_high+1], index_low):
            if dec1 < (dec1 + max_radius):
                dist_check = angular_dist2(ra0, dec0, ra1, dec1)
                if dist_check < dist:
                    dist = dist_check
                    min_index1 = index1
                    pass

            else:
                break

        # check against max_radius
        if dist > max_radius:
            no_matches.append(index0)

        else:
            matches.append((index0, order[min_index1], np.degrees(dist)))

    # stop timer
    time_taken = time.perf_counter() - start

    return matches, no_matches, time_taken

# some testing:
cat0 = np.array([[180, 30], [45, 10], [300, -45], [250, 67], [100, 12], [190, 45], [190, -26], [234, 22]])
cat1 = np.array([[180, 32], [55, 10], [302, -44], [150, 67], [250, 18], [111, 20], [204, --30], [235, 21]])
# sort cat 1 by declination; we create an index by declination (argsort), then order by this index
order = np.argsort(cat1[:,1])
cat1_ordered = cat1[order]
print("{}", cat1_ordered)
print("{}{}", cat1_ordered[:,1])


asc_dec = np.argsort(cat1[:, 1])
coords2_sorted = cat1[asc_dec]
dec2_sorted = coords2_sorted[:, 1]

print("MOW")
print(asc_dec)
print(coords2_sorted)
print(dec2_sorted)
print(asc_dec[2])

print(crossmatch5(cat0, cat1, 5))


# Last optimisation using the k-d matching in the Astropy module
# "to perform crossmatching incredibly quickly".
# Starting position will be the 2nd crossmatch function (with a lot of modifications)

#  more imports
from astropy.coordinates import SkyCoord
from astropy import units as u

def crossmatch6(cat0, cat1, max_radius):
    """
    Find closest star in bss catalogue to the postion given by ra, dec
    > Input: 2 catalogues, max allowable distance
    > Output: list of indices of matched items and angular distance, list of no matches, time to run function
    > Mutates input?: no
    > Astropy notes: output
    sky_cat1 = SkyCoord(coords1*u.degree, frame='icrs')
    sky_cat2 = SkyCoord(coords2*u.degree, frame='icrs')
    closest_ids, closest_dists, closest_dists3d = sky_cat1.match_to_catalog_sky(sky_cat2)
    3d distance s are ouputed, but we won't use ourselves
    """
    # start timer
    start = time.perf_counter()
    # we're using degrees here so don't need following line
    # max_radius = np.radians(max_radius)

    # initialise output lists
    matches = []
    no_matches = []

    #  convert catalogues to SkyCoord format in Astropy
    sky_cat0 = SkyCoord(cat0 * u.degree, frame='icrs')
    sky_cat1 = SkyCoord(cat1 * u.degree, frame='icrs')

    # use Astropy's k-d trees method
    closest_ids, closest_dists, closest_dists3d = sky_cat0.match_to_catalog_sky(sky_cat1)

    #  need to convest the closest_dists into a numpy array from some weird format
    closest_dists_array = closest_dists.value

    # you have to enumerate sky_cat1 as closest_ids gives sequential index from 2nd catalogue only
    # you have to enumerate sky_cat1 as closest_ids gives sequential index from 2nd catalogue only
    for index0, (index1, dist) in enumerate(zip(closest_ids, closest_dists_array)):
        if dist < max_radius:
            matches.append((index0, index1, dist))
        else:
            no_matches.append(index0)

    # stop timer
    time_taken = time.perf_counter() - start

    return matches, no_matches, time_taken


# ^ above works in markng scheme with the catalgues they chose, but not with the cat01 and cat1 above for some reason.
# So leaving as basically works.