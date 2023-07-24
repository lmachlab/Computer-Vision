from PIL import Image
import numpy as np
import cv2
import os

def save_as_video(path_to_jpg, output_name, frame_rate=25, frame_size=(96, 128)):
    img_files = sorted(os.listdir(path_to_jpg))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_name, fourcc, frame_rate, frame_size)
    for img_file in img_files:
        img_path = os.path.join(path_to_jpg, img_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, frame_size)
        out.write(img)
    out.release()
    cv2.destroyAllWindows()
    return

def video_tracking(metric, initial_index=[19, 50], frame_width=45, frame_height=55):
    previous = 'image_girl/0001.jpg'
    image = cv2.imread(previous)
    x = initial_index[1]
    y = initial_index[0]
    cv2.rectangle(image, (x, y), (x + frame_width, y + frame_height), (0, 255, 0), 2)
    if metric == 'SSD':
        cv2.imwrite('SSD_images/1.jpg', image)
    elif metric == 'CC':
        cv2.imwrite('CC_images/1.jpg', image)
    else:
        cv2.imwrite('NCC_images/1.jpg', image)

    previous_image = np.array(Image.open(previous))
    previous_frame = previous_image[initial_index[0]:initial_index[0]+frame_height, initial_index[1]:initial_index[1]+frame_width]
    

    for i in range(2, 501):
        if i//10 == 0:
            next = f'image_girl/000{i}.jpg'
        elif i//100 == 0:
            next = f'image_girl/00{i}.jpg'
        else:
            next = f'image_girl/0{i}.jpg'
        if metric == 'SSD':
            next_index = ssd_choose_frame(next, previous_frame)
            image = cv2.imread(next)
            x = next_index[1]
            y = next_index[0]
            cv2.rectangle(image, (x, y), (x + frame_width, y + frame_height), (0, 255, 0), 2)
            cv2.imwrite(f'SSD_images/{i}.jpg', image)
        elif metric == 'CC':
            next_index = cc_choose_frame(next, previous_frame)
            image = cv2.imread(next)
            x = next_index[1]
            y = next_index[0]
            cv2.rectangle(image, (x, y), (x + frame_width, y + frame_height), (0, 255, 0), 2)
            cv2.imwrite(f'CC_images/{i}.jpg', image)
        else:
            next_index = ncc_choose_frame(next, previous_frame)
            image = cv2.imread(next)
            x = next_index[1]
            y = next_index[0]
            cv2.rectangle(image, (x, y), (x + frame_width, y + frame_height), (0, 255, 0), 2)
            cv2.imwrite(f'NCC_images/{i}.jpg', image)
        next_image = np.array(Image.open(next))
        previous_frame = next_image[y:y+frame_height, x:x+frame_width]
    
    return

    

def ssd_choose_frame(image_name, previous_array, frame_width=45, frame_height=55):
    opened_image = np.array(Image.open(image_name))

    rows, columns, colors = np.shape(opened_image)

    min_ssd = float('inf')
    for i in range(rows-frame_height):
        for j in range(columns-frame_width):
            next_frame = opened_image[i:i+frame_height, j:j+frame_width]
            next_ssd = ssd_calc(next_frame, previous_array)
            if next_ssd < min_ssd:
                min_ssd = next_ssd
                min_index = [i, j]
    return min_index

def cc_choose_frame(image_name, previous_array, frame_width=45, frame_height=55):
    opened_image = np.array(Image.open(image_name))

    rows, columns, colors = np.shape(opened_image)

    max_ssd = 0
    for i in range(rows-frame_height):
        for j in range(columns-frame_width):
            next_frame = opened_image[i:i+frame_height, j:j+frame_width]
            next_ssd = cc_calc(next_frame, previous_array)
            if next_ssd > max_ssd:
                max_ssd = next_ssd
                max_index = [i, j]
    return max_index

def ncc_choose_frame(image_name, previous_array, frame_width=45, frame_height=55):
    opened_image = np.array(Image.open(image_name))

    rows, columns, colors = np.shape(opened_image)

    max_ssd = 0
    for i in range(rows-frame_height):
        for j in range(columns-frame_width):
            next_frame = opened_image[i:i+frame_height, j:j+frame_width]
            next_ssd = ncc_calc(next_frame, previous_array)
            if next_ssd > max_ssd:
                max_ssd = next_ssd
                max_index = [i, j]
    return max_index


def ssd_calc(previous_roi, next_roi):
    
    subtract = next_roi-previous_roi
    distance = subtract**2
    total = np.sum(distance)

    return total

def cc_calc(previous_roi, next_roi):

    multiply = next_roi*previous_roi
    correlation = np.sum(multiply)

    return correlation

def ncc_calc(previous_roi, next_roi):

    avg_previous = np.sum(previous_roi)/np.size(previous_roi)
    avg_next = np.sum(next_roi)/np.size(previous_roi)

    next_hat = next_roi - avg_next
    previous_hat = previous_roi - avg_previous

    numerator = np.sum(next_hat*previous_hat)

    next_sum = np.sum(next_hat**2)
    previous_sum = np.sum(previous_hat**2)
    denominator = np.sqrt(next_sum*previous_sum)
    normalized_cc = numerator/denominator 

    return normalized_cc

if __name__ == "__main__":
    video_tracking(metric='SSD')
    video_tracking(metric='CC')
    video_tracking(metric='NCC')



