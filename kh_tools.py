import numpy as np
# http://scikit-image.org/docs/dev/auto_examples/filters/plot_denoise.html
from skimage.util import random_noise


def get_noisy_data(data):
    """Apply random noise to images data
    
    Arguments:
        data {np.array} -- Image data.
    
    Returns:
        np.array -- Output images.
    """

    lst_noisy = []
    sigma = 0.155
    for image in data:
        noisy = random_noise(image, var=sigma ** 2)
        lst_noisy.append(noisy)
    return np.array(lst_noisy)
