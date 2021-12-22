import numpy as np
from scipy.stats import kstest

def _listify(string):
    """ Turn a single line (as a string) in the data file into a list. """

    lst = []
    stg = string.rstrip()[1:len(string)-2]
    i = 0
    while i < len(stg):
        j = 0
        while i+j < len(stg) and stg[i+j] != ',':
            j += 1
        lst.append(float(stg[i: i+j]))
        j += 1
        i = i+j

    return lst


def _anti_textify(name):
    """ Return a list of lists that were converted from line by line from the
    data file named name. """

    f = open(name, 'r')
    lst = []
    line = f.readline()
    while line != '':
        lst.append(_listify(line))
        line = f.readline()
    f.close()

    return lst


def _distances(sortedlst):
    """ Return the differences/distances between consecutive elements in a
    sorted list of real numbers. """

    dist = []
    for i in range(len(sortedlst)-1):
        dist.append(sortedlst[i+1] - sortedlst[i])

    return dist


def _get_distances(data_file_name, pad1=2, pad2=2):
    """ Return a big list of all consecutive distances from the file with name
    data_file_name, from all rows/samples, after removing pad1 points from the
    left, and pad2 points from the right of each sample. """

    endpts = _anti_textify(data_file_name)
    dists = []
    size = len(endpts)
    for i in range(size):
        sortedlst = endpts[i]
        dist = _distances(sortedlst)
        dists += dist[pad1: len(dist)-pad2]

    return dists


def _get_distances_separate(data_file_name, pad1=2, pad2=2):
    """ Return a list of lists, each contain all consecutive distances from each
    row/sample from the file with name data_file_name after removing pad1 points
    from the left and pad2 points from the right of each sample. The difference
    between this function and _get_distances is that this function returns
    distances from each sample as separate lists, rather than in a single list.
    """

    endpts = _anti_textify(data_file_name)
    dists = []
    size = len(endpts)
    for i in range(size):
        sortedlst = endpts[i]
        dist = _distances(sortedlst)
        dists. append(dist[pad1:len(dist)-pad2])

    return dists


def _jump_pairs(data_file_name, jump, pad1=2, pad2=2):
    """ Return two numpy array each containing points in the point fields given
    by the file with name data_file_name. The two lists have the same length.
    For each index i of the lists, the corresponding two entries from both lists
    are points in the point field that is a gap of jump apart. (After removing
    pad1 points from the left, and pad2 points from the right of each sample.)
    """

    dists = _get_distances_separate(data_file_name, pad1=pad1, pad2=pad2)
    lefties, righties = [], []
    for dist_lst in dists:
        for i in range(len(dist_lst) - jump):
            lefties.append(dist_lst[i])
            righties.append(dist_lst[i + jump])
    lefties, righties = np.array(lefties), np.array(righties)

    return lefties, righties


def ks_distance(file1, file2, pad1=2, pad2=2):
    """ Perform Kolmogorov-Smirnov tests on the distance distributions of point
    field data from files with name file1 and file2, after removing pad1 points
    from the left, and pad2 points from the right of each sample in both data
    sets. """

    x = _get_distances(file1, pad1=pad1, pad2=pad2)
    y = _get_distances(file2, pad1=pad1, pad2=pad2)
    x = np.array(x)
    y = np.array(y)
    x = x/np.mean(x)
    y = y/np.mean(y)

    print(kstest(x, y))
    print('\n')


def ks_jump(file1, file2, jump, pad1=2, pad2=2):
    """ Perform Kolmogorov-Smirnov tests on the jump-jump ratio distributions of
    point field data from files with name file1 and file2, after removing pad1
    points from the left, and pad2 points from the right of each sample in both
    data sets. """

    lefties, righties = _jump_pairs(file1, jump, pad1=pad1, pad2=pad2)
    data1 = lefties/righties
    lefties, righties = _jump_pairs(file2, jump, pad1=pad1, pad2=pad2)
    data2 = lefties/righties

    print(kstest(data1, data2))
    print('\n')
