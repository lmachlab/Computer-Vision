import cv2
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import plotly.io as pio

class HoughTransform:
    def __init__(self, image_name, output_name, num_thetas=180, num_rhos=180, threshold=50):
        self.image_name = image_name
        self.output_name = output_name
        self.image = cv2.imread(image_name)
        self.num_thetas = int(num_thetas)
        self.num_rhos = int(num_rhos)
        self.threshold = threshold
        self.edges = self.bmp_to_edges(image_name)
        self.lines = self.hough_lines(self.edges, self.num_thetas, self.num_rhos, self.threshold)

    def bmp_to_edges(self, image_name):
        img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        blur_img = cv2.GaussianBlur(img, (3, 3), 0.5)
        edge_img = cv2.Canny(blur_img, threshold1=100, threshold2=200)
        return edge_img

    def hough_lines(self, edge_img, num_thetas=180, num_rhos=180, threshold=50):
        # define the dimensions of the hough space
        theta_step = 180 / num_thetas
        rows, columns = edge_img.shape
        half_row = rows/2
        half_column = columns/2
        diag_dist = int(round(np.sqrt(edge_img.shape[0]**2 + edge_img.shape[1]**2)))
        rho_step =  2*diag_dist / num_rhos

        theta_values = np.arange(0, 180, theta_step)
        rho_values = np.arange(-diag_dist, diag_dist, rho_step)

        sin_theta = np.sin(np.radians(theta_values))
        cos_theta = np.cos(np.radians(theta_values))
        accumulator = np.zeros((len(rho_values), len(theta_values)))

        # start making plots
        plots = plt.figure(figsize=(12, 12))
        subplot1 = plots.add_subplot(1, 4, 1)
        subplot1.imshow(self.image)
        
        subplot2 = plots.add_subplot(1, 4, 2)
        subplot2.imshow(edge_img, cmap="gray")
        
        subplot3 = plots.add_subplot(1, 4, 3)
        subplot3.set_facecolor((0, 0, 0))
        
        subplot4 = plots.add_subplot(1, 4, 4)
        subplot4.imshow(self.image)


        # step through each pixel, if is in an edge, calculate all possible
        # thetas and rhos and add to accumulator
        
        for i in range(rows):
            for j in range(columns):
                if edge_img[i][j]:
                    hough_rho = []
                    hough_theta = []
                    for theta_i in range(len(theta_values)):
                        rho = (j-half_column)*cos_theta[theta_i] + (i-half_row)*sin_theta[theta_i]
                        theta = theta_values[theta_i]
                        rho_i = np.argmin(np.abs(rho_values-rho))
                        accumulator[rho_i][theta_i] += 1
                        hough_rho.append(rho)
                        hough_theta.append(theta)
                    subplot3.plot(hough_theta, hough_rho, color="white", alpha=0.05)
        
        # find points with most overlaps in this
        lines = []
        for i in range(len(rho_values)):
            for j in range(len(theta_values)):
                if accumulator[i][j] >= threshold:
                    rho = rho_values[i]
                    theta = theta_values[j]
                    a = cos_theta[j]
                    b = sin_theta[j]
                    x_1 = 5
                    x_2 = 245
                    x0 = a * rho + half_column #(rho_values[i] - diag_dist)
                    y0 = b * rho + half_row #(rho_values[i] - diag_dist)
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    lines.append([(x1, y1), (x2, y2)])
                    subplot3.plot([theta], [rho], marker='o', color="yellow")
                    subplot4.add_line(mlines.Line2D([x1, x2], [y1, y2]))

        subplot3.invert_yaxis()
        # subplot3.invert_xaxis()

        subplot1.title.set_text(self.image_name)
        subplot2.title.set_text("Image Edges")
        subplot3.title.set_text("Hough Space")
        subplot4.title.set_text("Detected Lines")
        plots.savefig(self.output_name, format='png')

        return lines

    
    def plot_lines(self, output_name):
        img = cv2.imread(self.image_name)

        for line in self.lines:
            x1, y1 = line[0]
            x2, y2 = line[1]
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Save the output image
        cv2.imshow(output_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return



    def save_numpy_img(self, image_array, image_name):
        image_array = image_array.astype(np.uint8)
        image = Image.fromarray(image_array)
        image.save(image_name)
        return 