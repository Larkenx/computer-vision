from scipy.cluster.vq import *
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve2d
from scipy.spatial.distance import euclidean
from numpy.random import randn
from pylab import *
from PIL import Image
from skimage import feature
import matplotlib.colors
import random

def prepare_save(f):
    f_max = double(amax(f))
    f_min = double(amin(f))
    # print f_max
    # print f_min
    for i in range(0, f.shape[0]):
        for j in range(0, f.shape[1]):
            if f[i,j] == f_max:
                f[i,j] = 255
            elif f[i,j] > 1:
                f[i,j] = 122.5
            elif f[i,j] == 1:
                f[i,j] = 50


    return uint8(f)

    # return uint8(( (f - f_min) / (f_max - f_min) * 255))

# Problem 1

# Canny edge detection
img = Image.open('line_original.jpg')
aimg = array(img.convert('L'), dtype="double")
canny_img = feature.canny(aimg, sigma=1)
Image.fromarray(uint8(canny_img)*255).save('line_canny.jpg')

# Hough Transform
diagonal = sqrt(canny_img.shape[0]**2 + canny_img.shape[1]**2)
bin_size = 1
H = zeros(((diagonal / bin_size)+1, 360 / bin_size))
print "Diagonal is: ", diagonal
for i in range(0, canny_img.shape[0]):
    for j in range(0, canny_img.shape[1]):
        edge_point = canny_img[i,j]
        if edge_point:
            for theta in range(0, 360 / bin_size):
                d = uint8((j*sin(theta) + i*cos(theta)) / bin_size)# divide by d bin size
                # print d
                H[d,theta] += 1

imshow(H)
show()
Image.fromarray(prepare_save(H)).save('line_hough.jpg')




