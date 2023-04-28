import cv2
import numpy as np
import glob

image_path = "sample/sample.jpg"
DATASET_DIR = "sample"
image = cv2.imread(image_path)

IMG_EXTS = ['*.jpg', '*.jpeg', '*.png', '*.JPG']
IMAGE_PATHS =[]
[IMAGE_PATHS.extend(glob.glob(f'{DATASET_DIR}/**/'+ x, recursive=True)) for x in IMG_EXTS]

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 10, 290)
    return edges

def region(image):
    height = image.shape[0]
    width = image.shape[1]
    #isolate the gradients that correspond to the lane lines
    triangle = np.array([
                        [(550, 800), (1100,800), (800,390), (680, 390),]
                        ])
    #create a black image with the same dimensions as original image
    mask = np.zeros_like(image)
    #create a mask (triangle that isolates the region of interest in our image)
    cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask

def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    #make sure array not empty
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
        #draw lines on black image
            print(line)
            cv2.line(lines_image, (x1,y1), (x2,y2), (255, 0, 0), 10)
    return lines_image

def average(image, lines):
    left = []
    right = []

    # if lines is not None:
    for line in lines:
        # print(line)
        x1, y1, x2, y2 = line.reshape(4)
        #fit line to points, return slope and y-int
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        # print(parameters)
        slope = parameters[0]
        y_int = parameters[1]
        #lines on right have positive slope, and lines on the left have neg slope
        if slope < 0:
            left.append((slope, y_int))
        else:
            right.append((slope, y_int))
    left_fit_average = np.average(left, axis = 0)
    right_fit_average = np.average(right, axis=0)
    # print(left_fit_average, "left")
    # print(right_fit_average, "right")

    #takes average among all columns (column0: slope, column1: y_int)
    right_avg = np.average(right, axis=0)
    left_avg = np.average(left, axis=0)
    #create lines based on averages calculates
    left_line = make_points(image, left_avg)
    right_line = make_points(image, right_avg)
    return np.array([left_line, right_line])

def make_points(image, average):
    # print(average)
    slope, y_int = average
    y1 = image.shape[0]
    #how long we want our lines to be --> 3/5 the size of the image
    y2 = int(y1 * (3/6))
    # y2 = int(y1 * 0)
    #determine algebraically
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])

#'''##### DETECTING lane lines in image ######'''#

lane_image = np.copy(image)
edges = canny(lane_image)
isolated = region(edges)

cv2.imshow("edges", edges)
cv2.imshow("isolated", isolated)

#Drawing Lines: (order of params) --> region of interest, bin size (P, theta), min intersections needed, placeholder array, 
lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines = average(lane_image, lines)
black_lines = display_lines(lane_image, averaged_lines)
# #taking wighted sum of original image and lane lines image
lanes = cv2.addWeighted(lane_image, 0.8, black_lines, 1, 1)
cv2.imshow("black lines", black_lines)
cv2.imshow("line detection", lanes)

cv2.imwrite("output/sample.jpg", lanes)

cv2.waitKey(0)
cv2.destroyAllWindows