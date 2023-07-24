import numpy as np
from MP6_code import HoughTransform
import cv2


test1_hough = HoughTransform('test.bmp', output_name='test1_lines.png', threshold=92)
test2_hough = HoughTransform('test2.bmp', output_name='test2_lines.png', threshold=70)
input_hough = HoughTransform('input.bmp', output_name='input_lines.png', threshold=60)
test1_small = HoughTransform('test.bmp', output_name='test1_lines_small.png', threshold=50, num_rhos=100, num_thetas=100)