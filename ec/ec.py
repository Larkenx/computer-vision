from PIL import Image
from numpy import *
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import feature

inf = float('inf')
# Load in the scene
scene = Image.open('chamfer_scene.jpg')
# Load in the yield sign
search_object = Image.open('chamfer_object.png')
search_object.thumbnail((65,65), Image.ANTIALIAS)
# Perform canny edge detection on the yield sign
# array of the search object, converted to grayscale.
aso = array(search_object.convert('L'), dtype="double")
ascene = array(scene.convert('L'), dtype="double")

# Exercise 3
canny_scene = feature.canny(ascene, sigma=3)
canny_search_object = feature.canny(aso, sigma=3)

# I couldn't get these images to save. They turned into black images upon saving,
# so I saved them from the matplotlib pyplot interface.
Image.fromarray(uint8(canny_scene)*255).save('chamfer_scene_canny2.jpg')
Image.fromarray(uint8(canny_search_object)*255).save('chamfer_object_canny2.jpg')

# plt.imshow(canny_search_object, cmap=plt.cm.gray)
# plt.show()


def distance_transform(canny_img):
    d = empty_like(canny_img, dtype="double")
    # Initialize all pixels to 0 or inf
    for i in range(canny_img.shape[0]):
        for j in range(canny_img.shape[1]):
            if canny_img[i,j] > 0:
                d[i,j] = 0
            else:
                d[i,j] = -1

    # Exercise 3

    # forward pass: left,above from left to right
    for i in range(1, canny_img.shape[0]): # not doing the last row
        for j in range(1, canny_img.shape[1]): # not doing the last column
            if d[i,j] == -1:
                if d[i-1, j] >= 0: # above
                    d[i,j] = d[i-1, j] + 1
                elif [i, j-1] >= 0: # left
                    d[i,j] = d[i, j-1] + 1

    # backwards pass: right,below from left to right
    for i in range(canny_img.shape[0]-1, 0, -1): # not doing the last row
        for j in range(canny_img.shape[1]-2, 0, -1): # not doing the last column
            if d[i,j] == -1:
                if d[i+1, j] >= 0 and d[i+1,j] < d[i, j+1]: # below
                    d[i,j] = d[i+1, j] + 1
                elif [i, j+1] >= 0: # right
                    d[i,j] = d[i, j+1] + 1

    # print d.shape
    print d[:,:-1]
    return d

distance_transform(canny_scene)
# plt.imshow(distance_transform(canny_scene))#, cmap=plt.cm.jet)
# plt.show()
