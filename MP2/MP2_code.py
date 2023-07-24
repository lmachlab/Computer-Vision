from PIL import Image 
import numpy as np

def bmp_to_np(image_name):
    img = Image.open(image_name)
    # convert the image to an array
    image_array = np.array(img)
    return image_array

def dilation(image_name, SE):
    image_array = bmp_to_np(image_name)
    return dilation_helper(image_array, SE)


def dilation_helper(image_array, SE):
    height, width = image_array.shape
    SE_height, SE_width = SE.shape


    pad_height = SE_height // 2
    pad_width = SE_width // 2

    dilated_image = np.zeros((height, width), dtype=np.uint8)


    for i in range(height):
        for j in range(width):
            # Check if the current pixel is a foreground pixel (i.e., white)
            if image_array[i, j] > 0:
                # Loop through each pixel in the kernel
                for ki in range(SE_height):
                    for kj in range(SE_width):
                        # Calculate corresponding pixel coordinates in the input image
                        ni = i - pad_height + ki
                        nj = j - pad_width + kj

                        # Check if the corresponding pixel is within the image boundaries
                        if 0 <= ni < height and 0 <= nj < width:
                            # Set the corresponding pixel in the dilated image to white
                            dilated_image[ni, nj] = 255

    return dilated_image

def erosion(image_name, SE):
    image_array = bmp_to_np(image_name)
    return erosion_helper(image_array, SE)

def erosion_helper(image_array, SE):
    
    height, width = image_array.shape
    SE_height, SE_width = SE.shape

    eroded_image = np.zeros((height, width), dtype=np.uint8)


    for i in range(height):
        for j in range(width):
            # get region of the image based on SE
            img_se = image_array[i:i+SE_height, j:j+SE_width]
            # see if SE is a subset of the image
            np.resize(SE, img_se.shape)
            if SE.size == img_se.size:
                sub_result = np.logical_and(img_se, SE)
            else:
                sub_result = np.logical_and(img_se, np.ones_like(img_se))
            #update value in eroded image
            if np.min(sub_result) != 0:
                eroded_image[i, j] = 255

    return eroded_image

def opening(image_name, SE_e, SE_d):
    image_array = bmp_to_np(image_name)
    eroded_image = erosion_helper(image_array, SE_e)
    dilated_image = dilation_helper(eroded_image, SE_d)
    return dilated_image

def closing(image_name, SE_e, SE_d):
    image_array = bmp_to_np(image_name)
    dilated_image = dilation_helper(image_array, SE_d)
    eroded_image = erosion_helper(dilated_image, SE_e)
    return eroded_image

def boundary(image_name, SE):
    image_array = bmp_to_np(image_name)
    height, width = image_array.shape
    eroded_image = erosion_helper(image_array, SE)
    boundary = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if image_array[i][j] == True:
                if eroded_image[i][j] == False:
                    boundary[i][j] = 255
    return boundary

    

def save_numpy_img(image_array, image_name):
    image = Image.fromarray(image_array)
    image.save(image_name)
    return 




