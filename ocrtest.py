import cv2
import pytesseract
from PIL import Image
from matplotlib import pyplot as plt

img_file = "example.jpg"
img = cv2.imread(img_file)

#Convert color image to grayscale
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

im_gray = grayscale(img)

def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)
    height, width  = im_data.shape[:2]
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()

#binarization
thresh, im_blackwhite = cv2.threshold(im_gray, 157, 255, cv2.THRESH_BINARY)

#Preprocessing step: reducing noise
def noise_removal(image):
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return (image)

im_noiseless = noise_removal(im_gray)

#Preprocessing step: removing extraneous borders
def remove_borders(image):
    contours, heiarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return (crop)

#saving preprocessed image
im_borderless = remove_borders(im_noiseless)
cv2.imwrite("borderless.jpg", im_borderless)


#Name (Last, First, Middle)
x,y,w,h = 32, 173, 264, 18
ROI = im_borderless[y:y+h, x:x+w]
name = pytesseract.image_to_string(ROI, lang='eng', config='--psm 6')
print("Name: " + name)

#Address
x,y,w,h = 32, 202, 264, 18
ROI = im_borderless[y:y+h, x:x+w]
street = pytesseract.image_to_string(ROI, lang='eng', config='--psm 6')

x,y,w,h = 32, 234, 214, 18
ROI = im_borderless[y:y+h, x:x+w]
city = pytesseract.image_to_string(ROI, lang='eng', config='--psm 6')

x,y,w,h = 265, 234, 31, 18
ROI = im_borderless[y:y+h, x:x+w]
state = pytesseract.image_to_string(ROI, lang='eng', config='--psm 6')

print("Address: " + street, city, state)