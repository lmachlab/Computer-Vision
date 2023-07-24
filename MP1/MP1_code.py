from PIL import Image

def component_labeller(image_name):
    # open image
    img = Image.open(image_name)

    # convert the image to a matrix
    width, height = img.size
    img_matrix = []
    for y in range(height):
        row = []
        for x in range(width):
            pixel = img.getpixel((x, y))
            row.append(pixel)
        img_matrix.append(row)
    # relabel the connected components. white is 255
    i = 0
    label_value = 50
    connect_sets = []
    while i < height:
        
        j = 0
        while j < width:
            
            if img_matrix[i][j] == 255: #if the color is white 
                if img_matrix[i-1][j] == 0 and img_matrix[i][j-1] == 0: #if both neighbors are color black
                    img_matrix[i][j] = label_value
                    connect_sets.append({label_value})
                    label_value += 10
                elif img_matrix[i][j-1] != 0 and img_matrix[i-1][j] == 0: #if the left neighbor only is not black
                    img_matrix[i][j] = img_matrix[i][j-1]
                elif img_matrix[i][j-1] == 0 and img_matrix[i-1][j] != 0: #if the up neighbor only is not black
                    img_matrix[i][j] = img_matrix[i-1][j]
                else: #both the up and left are not black
                    up_label = img_matrix[i-1][j]
                    left_label = img_matrix[i][j-1]
                    img_matrix[i][j] = min(up_label, left_label)
                    for item in connect_sets:
                        if up_label in item:
                            up_index = connect_sets.index(item)
                        if left_label in item:
                            left_index = connect_sets.index(item)
                    if up_index != left_index:
                        new_set = connect_sets[up_index].union(connect_sets[left_index])
                        indices = sorted([left_index, up_index], reverse=True)
                        for index in indices:
                            del connect_sets[index]
                        connect_sets.append(new_set)
            j+=1
        i+=1
    return len(connect_sets)




print('the number of connected components in the face image is ', component_labeller('sample_photos/face.bmp'))
print('the number of connected components in the gun image is ', component_labeller('sample_photos/gun.bmp'))
print('the number of connected components in the test image is ', component_labeller('sample_photos/test.bmp'))


