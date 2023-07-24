from PIL import Image 
import numpy as np

def bmp_to_np(image_name):
    img = Image.open(image_name)
    # convert the image to an array
    image_array = np.array(img)
    return image_array


def hist_equalize(image_name):
    image_array = bmp_to_np(image_name)
    # make the histogram
    hist, bins = np.histogram(image_array.flatten(), bins=256, range=(0, 255))

    # get the cumulative distribution function of the histogram and normalize it to the [0, 255] range
    cdf = hist.cumsum()
    normalized_cdf = cdf*255/cdf[-1]
    
    # equalize the histogram using the normalized cdf
    equalized_dist = np.interp(image_array.flatten(), bins[:-1], normalized_cdf)

    # reshape the image
    equalized_image = np.reshape(equalized_dist, image_array.shape)
    equalized_image = np.uint8(equalized_image)
    return equalized_image


def save_numpy_img(image_array, image_name):
    image = Image.fromarray(image_array)
    image.save(image_name)
    return 
