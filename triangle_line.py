import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "sample/sample.jpg"
image = cv2.imread(image_path)

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 10, 290)
    return edges

def region(image):
    height = image.shape[0]
    width = image.shape[1]
    #isolate the gradients that correspond to the lane lines
    #Edit in the array to find the area that fit with the lane
    triangle = np.array([
                        [(550, height), (1100,height), (800,390), (680, 390),]
                        ])
    #create a black image with the same dimensions as original image
    mask = np.zeros_like(image)
    #create a mask (triangle that isolates the region of interest in our image)
    cv2.fillPoly(mask, triangle, 255)
    #Comment this mask to see the shape
    # mask = cv2.bitwise_and(image, mask)
    return mask

#Draw Triangle lines

edges = canny(image)
white_line = region(edges)

# plt.imshow(edges, cmap='gray')
# plt.show()

cv2.imshow("edges", edges)
cv2.imshow("display", white_line)

cv2.waitKey(0)
cv2.destroyAllWindows