import glob
import itertools

import scipy.misc
import scipy.ndimage


def iterate_files():
    return itertools.cycle(glob.glob('data/*.jpg'))

def load_original(filename):
    return scipy.ndimage.imread(filename, mode='RGB')

def load_seed(filename, original, zoom):
    target_shape = (original.shape[0]//zoom, original.shape[1]//zoom)
    return scipy.misc.imresize(original, target_shape, interp='bilinear')
