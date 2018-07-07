from scipy.cluster.vq import *
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve2d
from scipy.spatial.distance import euclidean
from numpy.random import randn
from pylab import *
from PIL import Image
import matplotlib.colors
import random
# Exercise 1
# Plotting a normal distribution of points
"""
class1 = 2.0 * randn(100, 2)
class2 = randn(100,2) + array([5,5])
class3 = randn(100,2) + array([5,0])
features = vstack((class1, class2, class3))
centroids, variance = kmeans(features, 3)
code, distance = vq(features, centroids)
figure()
ndx = where(code==0)[0]
plot(features[ndx,0],features[ndx,1],"*")
ndx = where(code==1)[0]
plot(features[ndx,0],features[ndx,1],"r.")
ndx = where(code==2)[0]
plot(features[ndx,0],features[ndx,1],"c.")
plot(centroids[:,0],centroids[:,1],"go")
axis("on")
show()
"""
# Exercise 2
# zebra = array(Image.open('zebra.jpg'))
# fish = array(Image.open('fish.jpg'))

# 2D array of RGB values
# zebra_rgb = zebra.reshape((zebra.shape[0]*zebra.shape[1], 3))
# fish_rgb = fish.reshape((fish.shape[0]*fish.shape[1], 3))

# Working with the zebra image
"""
k = 4
centroids,variance = kmeans(double(zebra_rgb), k)
code,distance = vq(zebra_rgb, centroids)
code_reshaped = code.reshape((zebra.shape[0],zebra.shape[1]))
colors = [[random.randint(0,255) for i in range(0,3)] for j in range(0, k)]

zclustered_img = zeros((code_reshaped.shape[0], code_reshaped.shape[1], 3))
for i in range(0, code_reshaped.shape[0]):
    for j in range(0, code_reshaped.shape[1]):
        clust_id = code_reshaped[i,j]
        zclustered_img[i,j] = colors[clust_id]


Image.fromarray(uint8(zclustered_img)).save('zebra_color_clustered_saved.jpg')
"""


# Working with the fish image. I didn't know what looked best,
# so I just saved all iterations for k=2,4,6,8,10. I think that
# 4 probably looks closest to the image on canvas :)
"""
for k in range(2,12,2):
    centroids,variance = kmeans(double(zebra_rgb), k)
    code,distance = vq(fish_rgb, centroids)
    code_reshaped = code.reshape((fish.shape[0],fish.shape[1]))
    colors = [[random.randint(0,255) for i in range(0,3)] for j in range(0, k)]

    fclustered_img = zeros((code_reshaped.shape[0], code_reshaped.shape[1], 3))
    for i in range(0, code_reshaped.shape[0]):
        for j in range(0, code_reshaped.shape[1]):
            clust_id = code_reshaped[i,j]
            fclustered_img[i,j] = colors[clust_id]

    Image.fromarray(uint8(fclustered_img)).save('fish_color_clustered_{}.jpg'.format(k))
"""

# Exercise 3
#

# Gaussian filters convolved with gradients
impulse = zeros((60, 60))
impulse[impulse.shape[0] / 2, impulse.shape[1] / 2] = 1

f1 = convolve2d(gaussian_filter(impulse, sigma=2), [[1, -1]], mode='same')
f2 = convolve2d(gaussian_filter(impulse, sigma=2), [[1], [-1]], mode='same')
f3 = convolve2d(gaussian_filter(impulse, sigma=4), [[1, -1]], mode='same')
f4 = convolve2d(gaussian_filter(impulse, sigma=4), [[1], [-1]], mode='same')
f5 = convolve2d(gaussian_filter(impulse, sigma=8), [[1, -1]], mode='same')
f6 = convolve2d(gaussian_filter(impulse, sigma=8), [[1], [-1]], mode='same')
f7 = gaussian_filter(impulse, sigma=8) - gaussian_filter(impulse, sigma=4)
f8 = gaussian_filter(impulse, sigma=4) - gaussian_filter(impulse, sigma=2)
filters = [f1, f2, f3, f4, f5, f6, f7, f8]

def produce_eight_dim(img):
    aimgs = []
    for f in filters:
        activated_img = convolve2d(img, f, mode="same")
        aimgs.append(activated_img)

    result = zeros((img.shape[0], img.shape[1], 8))
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            for k in range(0,8):
                result[i,j] = aimgs[k][i,j]

    return result


# Working with 8D zebra
"""
zebra = array(Image.open('zebra.jpg').convert('L'))
eight_z = produce_eight_dim(zebra)
features = eight_z.reshape((eight_z.shape[0]*eight_z.shape[1], 8))

for k in range(5, 20, 5):
    centroids,variance = kmeans(features, k)
    code,distance = vq(features, centroids)
    code_reshaped = code.reshape((zebra.shape[0],zebra.shape[1]))
    colors = [[random.randint(0,255) for i in range(0,3)] for j in range(0, k)]

    zclustered_img = zeros((code_reshaped.shape[0], code_reshaped.shape[1], 3))
    for i in range(0, code_reshaped.shape[0]):
        for j in range(0, code_reshaped.shape[1]):
            clust_id = code_reshaped[i,j]
            zclustered_img[i,j] = colors[clust_id]

    Image.fromarray(uint8(zclustered_img)).save('zebra_texture_clustering_{}.jpg'.format(k))
"""

# Working with 8D fish
"""
fish = array(Image.open('fish.jpg').convert('L'))
eight_z = produce_eight_dim(fish)
features = eight_z.reshape((eight_z.shape[0] * eight_z.shape[1], 8))

for k in range(5, 20, 5):
    centroids,variance = kmeans(features, k)
    code,distance = vq(features, centroids)
    code_reshaped = code.reshape((fish.shape[0],fish.shape[1]))
    colors = [[random.randint(0,255) for i in range(0,3)] for j in range(0, k)]

    zclustered_img = zeros((code_reshaped.shape[0], code_reshaped.shape[1], 3))
    for i in range(0, code_reshaped.shape[0]):
        for j in range(0, code_reshaped.shape[1]):
            clust_id = code_reshaped[i,j]
            zclustered_img[i,j] = colors[clust_id]

    Image.fromarray(uint8(zclustered_img)).save('fish_texture_clustering_{}.jpg'.format(k))
"""











