import MP2_code
import numpy as np

SE_1 = np.ones((3, 3))
SE_2 = np.ones((5, 5))
SE_3 = np.ones((2, 2))
SE_4 = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])


## dilate gun
dilated_gun_1 = MP2_code.dilation('gun.bmp', SE_1)
MP2_code.save_numpy_img(dilated_gun_1, 'dilated_gun_1.bmp')

dilated_gun_2 = MP2_code.dilation('gun.bmp', SE_3)
MP2_code.save_numpy_img(dilated_gun_2, 'dilated_gun_2.bmp')

dilated_gun_3 = MP2_code.dilation('gun.bmp', SE_4)
MP2_code.save_numpy_img(dilated_gun_3, 'dilated_gun_3.bmp')

## erode gun
eroded_gun_1 = MP2_code.erosion('gun.bmp', SE_1)
MP2_code.save_numpy_img(eroded_gun_1, 'eroded_gun_1.bmp')

eroded_gun_2 = MP2_code.erosion('gun.bmp', SE_3)
MP2_code.save_numpy_img(eroded_gun_2, 'eroded_gun_2.bmp')

## open gun
opened_gun_1 = MP2_code.opening('gun.bmp', SE_1, SE_2)
MP2_code.save_numpy_img(opened_gun_1, 'opened_gun_1.bmp')

opened_gun_2 = MP2_code.opening('gun.bmp', SE_2, SE_1)
MP2_code.save_numpy_img(opened_gun_2, 'opened_gun_2.bmp')

opened_gun_3 = MP2_code.opening('gun.bmp', SE_1, SE_1)
MP2_code.save_numpy_img(opened_gun_3, 'opened_gun_3.bmp')

##close gun
closed_gun_1 = MP2_code.closing('gun.bmp', SE_1, SE_2)
MP2_code.save_numpy_img(closed_gun_1, 'closed_gun_1.bmp')

closed_gun_2 = MP2_code.closing('gun.bmp', SE_1, SE_1)
MP2_code.save_numpy_img(closed_gun_2, 'closed_gun_2.bmp')

closed_gun_3 = MP2_code.closing('gun.bmp', SE_3, SE_2)
MP2_code.save_numpy_img(closed_gun_3, 'closed_gun_3.bmp')

## gun boundary
gun_boundary_1 = MP2_code.boundary('gun.bmp', SE_1)
MP2_code.save_numpy_img(gun_boundary_1, 'gun_boundary_1.bmp')

gun_boundary_2 = MP2_code.boundary('gun.bmp', SE_2)
MP2_code.save_numpy_img(gun_boundary_2, 'gun_boundary_2.bmp')

gun_boundary_3 = MP2_code.boundary('gun.bmp', SE_3)
MP2_code.save_numpy_img(gun_boundary_3, 'gun_boundary_3.bmp')

gun_boundary_4 = MP2_code.boundary('gun.bmp', SE_4)
MP2_code.save_numpy_img(gun_boundary_4, 'gun_boundary_4.bmp')


## dilate palm
dilated_palm_1 = MP2_code.dilation('palm.bmp', SE_1)
MP2_code.save_numpy_img(dilated_palm_1, 'dilated_palm_1.bmp')

dilated_palm_2 = MP2_code.dilation('palm.bmp', SE_2)
MP2_code.save_numpy_img(dilated_palm_2, 'dilated_palm_2.bmp')

## erode palm
eroded_palm_1 = MP2_code.erosion('palm.bmp', SE_1)
MP2_code.save_numpy_img(eroded_palm_1, 'eroded_palm_1.bmp')

eroded_palm_2 = MP2_code.erosion('palm.bmp', SE_2)
MP2_code.save_numpy_img(eroded_palm_2, 'eroded_palm_2.bmp')

## open palm
opened_palm_1 = MP2_code.opening('palm.bmp', SE_3, SE_1)
MP2_code.save_numpy_img(opened_palm_1, 'opened_palm_1.bmp')

##close palm
closed_palm_1 = MP2_code.closing('palm.bmp', SE_1, SE_1)
MP2_code.save_numpy_img(closed_palm_1, 'closed_palm_1.bmp')

## palm boundary
palm_boundary_1 = MP2_code.boundary('palm.bmp', SE_3)
MP2_code.save_numpy_img(palm_boundary_1, 'palm_boundary_1.bmp')

palm_boundary_2 = MP2_code.boundary('palm.bmp', SE_2)
MP2_code.save_numpy_img(palm_boundary_2, 'palm_boundary_2.bmp')



