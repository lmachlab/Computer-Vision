from PIL import Image 
from scipy import ndimage
from scipy.ndimage import convolve
import numpy as np
import cv2
import matplotlib.pyplot as plt

class CannyEdgeDetection:
    def __init__(self, image_name, output_name, kernel_size=5, sigma=0.5, high_ratio=0.21, low_ratio=0.11, min_val=50, max_val=200):
        self.image_name = image_name
        self.output_name = output_name
        self.smoothed = self.smooth_image(self.image_name, kernel_size, sigma)
        self.mag = self.calculate_gradient(self.smoothed)[0]
        self.angle = self.calculate_gradient(self.smoothed)[1]
        self.suppressed = self.perform_suppression(self.mag, self.angle)
        self.threshold = self.double_threshold(self.suppressed, high_ratio, low_ratio)
        self.edges = self.track_edge(self.threshold, min_val, max_val)
        self.official_edges = self.official_canny(self.image_name)

    def official_canny(self, image_name):
        img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        blur_img = cv2.GaussianBlur(img, (3, 3), 0.5)
        edge_img = cv2.Canny(blur_img, threshold1=100, threshold2=200)
        return edge_img
    

    def smooth_image(self, image_name, kernel_size, sigma):
        # creates the gaussian kernel and apply the filter to the image
        image_array = np.array(Image.open(image_name).convert('L'))

        kernel = np.zeros((kernel_size,kernel_size))

        half = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                x = i - half
                y = j - half
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel /= kernel.sum()

        smoothed = convolve(image_array, kernel)
        return smoothed

    def calculate_gradient(self, smoothed_array):
        # calculate the gradient (the magnitude and the angle of the gradient)
        smoothed_array = np.float64(smoothed_array)
        sobel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
        gradient_x = convolve(smoothed_array, sobel_x)
        gradient_y = convolve(smoothed_array, sobel_y)
        mag = np.sqrt(gradient_x**2 + gradient_y**2)
        mag = mag / mag.max() * 255
        angle = np.arctan2(gradient_y, gradient_x) * 180 / np.pi
        return mag, angle

    def perform_suppression(self, mag, angle):
        # carry out the non-maximum supression given the gradient
        suppressed = np.zeros_like(mag)
        height,width = mag.shape
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                theta = angle[i, j]
                if (0 <= theta < 22.5) or (157.5 <= theta <= 180):
                    a = mag[i, j+1]
                    b = mag[i, j-1]
                elif (22.5 <= theta < 67.5):
                    a = mag[i+1, j+1]
                    b = mag[i-1, j-1]
                elif (67.5 <= theta < 112.5):
                    a = mag[i+1, j]
                    b = mag[i-1, j]
                elif (112.5 <= theta < 157.5):
                    a = mag[i-1, j+1]
                    b = mag[i+1, j-1]
                if mag[i, j] >= a and mag[i, j] >= b:
                    suppressed[i, j] = mag[i, j]
        return suppressed

    def double_threshold(self, suppressed, high_ratio, low_ratio):
        # double thresholding step (high and low threshold)
        low_threshold = low_ratio * np.max(suppressed)
        high_threshold = high_ratio * np.max(suppressed)
    
        height, width = suppressed.shape
        threshold = np.zeros_like(suppressed)
        for i in range(height):
            for j in range(width):
                if suppressed[i][j] >= high_threshold:
                    threshold[i][j] = 255
                elif suppressed[i][j] >= low_threshold:
                    threshold[i][j] = 75
        return threshold
    
    def track_edge(self, threshold_array, min_val, max_val):
        height, width = threshold_array.shape
        edges = np.zeros_like(threshold_array)
        for i in range(1, height-1):
            for j in range(1, width):
                if threshold_array[i][j] <= min_val:
                    edges[i][j] = 0
                elif threshold_array[i][j] >= max_val:
                    edges[i][j] = 255
                else:
                    if((threshold_array[i-1][j-1] >= max_val) or (threshold_array[i-1][j] >= max_val) or (threshold_array[i-1][j+1] >= max_val)
                        or (threshold_array[i][j-1] >= max_val) or (threshold_array[i][j+1] >= max_val) or 
                        (threshold_array[i+1][j-1] >= max_val) or (threshold_array[i+1][j] >= max_val) or (threshold_array[i+1][j+1] >= max_val)):
                        edges[i][j] = 255
                    else:
                        edges[i][j] = 0
        return edges
    
    def plot_results(self):
    
        plots = plt.figure(figsize=(12, 12))
        subplot1 = plots.add_subplot(2, 3, 1)
        subplot1.imshow(cv2.imread(self.image_name, cv2.IMREAD_GRAYSCALE), cmap='gray')

        subplot2 = plots.add_subplot(2, 3, 2)
        subplot2.imshow(self.smoothed, cmap="gray")

        subplot3 = plots.add_subplot(2, 3, 3)
        subplot3.imshow(self.mag, cmap="gray")

        subplot4 = plots.add_subplot(2, 3, 4)
        subplot4.imshow(self.suppressed, cmap="gray")

        subplot5 = plots.add_subplot(2, 3, 5)
        subplot5.imshow(self.threshold, cmap="gray")

        subplot6 = plots.add_subplot(2, 3, 6)
        subplot6.imshow(self.edges, cmap="gray")

        subplot1.title.set_text(self.image_name)
        subplot2.title.set_text("Image Smoothed")
        subplot3.title.set_text("Image Gradient")
        subplot4.title.set_text("After Non-Maximum Supression")
        subplot5.title.set_text("After Double-Thresholding")
        subplot6.title.set_text("Edges")
        plots.savefig(self.output_name, format='png')
        return 


    def save_numpy_img(self, image_array, image_name):
        image_array = image_array.astype(np.uint8)
        image = Image.fromarray(image_array)
        image.save(image_name)
        return 