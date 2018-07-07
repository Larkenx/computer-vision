from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve2d
from scipy.spatial.distance import euclidean
from numpy import *
from pylab import *
from PIL import Image

img = array(Image.open('zebra.jpg').convert('L'))
def prepare_save(f):
    f_max = double(amax(f))
    f_min = double(amin(f))
    # print f_max
    # print f_min
    return uint8(( (f - f_min) / (f_max - f_min) * 255))

# Generating guassian filters
impulse = zeros((60, 60))
impulse[impulse.shape[0] / 2, impulse.shape[1] / 2] = 1
# imshow(impulse, cmap=cm.gray)
# show()

# Gaussian filters convolved with gradients
f1 = convolve2d(gaussian_filter(impulse, sigma=2), [[1, -1]], mode='same')
f2 = convolve2d(gaussian_filter(impulse, sigma=2), [[1], [-1]], mode='same')
f3 = convolve2d(gaussian_filter(impulse, sigma=4), [[1, -1]], mode='same')
f4 = convolve2d(gaussian_filter(impulse, sigma=4), [[1], [-1]], mode='same')
f5 = convolve2d(gaussian_filter(impulse, sigma=8), [[1, -1]], mode='same')
f6 = convolve2d(gaussian_filter(impulse, sigma=8), [[1], [-1]], mode='same')
f7 = gaussian_filter(impulse, sigma=8) - gaussian_filter(impulse, sigma=4)
f8 = gaussian_filter(impulse, sigma=4) - gaussian_filter(impulse, sigma=2)
filters = [f1, f2, f3, f4, f5, f6, f7, f8]

# Saving filters
i = 0
for f in filters:
    f_name = "filter" + str(i) + ".jpg"
    Image.fromarray(prepare_save(f)).save(f_name)
    i += 1

# Convolving all 8 filters with test image
activated_zebras = []

i = 0
for f in filters:
    f_name = "zebra_activation" + str(i) + ".jpg"
    activated_img = convolve2d(img, f, mode="same")
    Image.fromarray(prepare_save(activated_img)).save(f_name)
    activated_zebras.append(activated_img)
    i += 1

result = zeros((img.shape[0], img.shape[1], 8))
for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
        for k in range(0, 8):
            print activated_zebras[k][i,j]
            result[i, j] = [activated_zebras[k][i, j]]

print result
# # Exercise 3
# D = empty_like(img)
# center = (img.shape[0] / 2, img.shape[1] / 2)
# cp = [x[center[0], center[1]] for x in activated_zebras]
# for i in range(0, img.shape[0]):
#     for j in range(0, img.shape[1]):
#         p = [x[i,j] for x in activated_zebras]
#         D[i,j] = euclidean(cp,p)
#
#
# Image.fromarray(prepare_save(D)).save('zebra_texture_comparison.jpg')

