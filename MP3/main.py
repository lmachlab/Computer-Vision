import MP3_code
from matplotlib import pyplot as plt

image_name = 'moon.bmp'
equalized_image_name = 'equalized_moon.bmp'

equalized_image = MP3_code.hist_equalize(image_name)
MP3_code.save_numpy_img(equalized_image, equalized_image_name)

image_array = MP3_code.bmp_to_np(image_name)

plt.hist(image_array.flatten(), bins = [x for x in range(0, 256)]) 
plt.title("pre-equalization histogram") 
plt.savefig('pre-equalization.png')

plt.hist(equalized_image.flatten(), bins=[x for x in range(0, 256)])
plt.title("equalized histogram")
plt.savefig('equalized.png')