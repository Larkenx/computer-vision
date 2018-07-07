from PIL import Image
from scipy import signal
from pylab import *
from scipy.ndimage import filters
# load the image
img = Image.open('empire.jpg')
# Crop the image and save it
cropped_img = img.crop((100, 100, 400, 400)).resize((128, 128))
cropped_img.save('empire_cropped.jpg')
# Resize the cropped image and save it
cropped_img.rotate(45).save('empire_cropped_rotated.jpg')
# Save a grayscale version of the empire img
img.convert('L').save('empire_gray.jpg')

# Using matplotlib, we can now do some stuff with our image
original = array(img)
cropped = array(img.crop((100, 100, 400, 400)))
cropped_resized = array(cropped_img)


def grayscale(arr):
    result = np.zeros(shape=(arr.shape[0], arr.shape[1]))
    for i in range(0, arr.shape[0]):
        for j in range(0, arr.shape[1]):
            pix = arr[i,j]
            result[i,j] = average(pix)
            # result[i,j] = (pix[0] * 299/1000 + pix[1] * 587/1000 + pix[2] * 114/1000) / 3

    return result

"""
# Build grayscale image from array (returns array)
grayscaled_img_arr = grayscale(original)
# Convert into an image
grayscaled_img = Image.fromarray(grayscaled_img_arr).convert('RGB')
# Save image
grayscaled_img.save('empire_gray_yours.jpg')

# Show images
imshow(grayscaled_img_arr, cmap=matplotlib.cm.Greys_r)
imshow(original)
imshow(cropped)
imshow(cropped_resized)
show()
"""
# This function only works on 3x3 filters.
def static_correlate(img, filter):
    img = array(img)
    new_img = copy(img)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            new_pix = 0
            # Trim the edges
            if i == 0 or i == (img.shape[0] - 1) or j == 0 or j == (img.shape[1] - 1):
                new_img[i, j] = 255
                continue

            pix = img[i,j]
            u = -1
            v = -1
            for h in filter:
                for k in h:
                    new_pix += ((k * img[i+u, j+v])/9)
                    v += 1
                    if v == 2:
                        u += 1
                        v = -1
            # print new_pix
            new_img[i,j] = new_pix

    return new_img

def correlate(img, filter):
    k_size = (filter.shape[0] - 1)  / 2 # if (filter.shape[0] == 3) 2k + 1 = 3
    divisor = 0
    for i in filter:
        for j in i:
            divisor += j

    img = array(img)
    new_img = copy(img)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            new_pix = 0
            # Trim the edges
            if i < k_size or i >= (img.shape[0] - k_size) or j < k_size or j >= (img.shape[1] - k_size):
                new_img[i, j] = 255
                continue

            u = k_size * -1
            v = k_size * -1
            for h in filter:
                for k in h:
                    # print i+u, j+v
                    new_pix += ((k * img[i+u, j+v]) / (divisor))
                    v += 1
                    if v == k_size+1:
                        u += 1
                        v = k_size * -1
            # print new_pix
            new_img[i,j] = new_pix

    return new_img

def convolute(img, filter):
    new_filter = filter[::-1]
    return correlate(img, new_filter)

def sharpening_filter(img):
    return correlate(img, impulsef -  bf1)

## Box Filters
# 3x3
bf1 = ones(shape=(3,3))
# 10x10
bf2 = ones(shape=(5, 5))
#20x20
bf3 = ones(shape=(7,7))
# Impulse Filter
impulsef = array([[0, 0, 0], [0, 2, 0], [0, 0, 0]])
# Gaussian Filter
gaussianf = array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

img = Image.open('empire_gray.jpg')
# Image.fromarray(correlate(array(img), bf1)).show()
# Image.fromarray(correlate(array(img), impulsef)).show()
# Image.fromarray(sharpening_filter(array(img))).show()

# box_filtered_img = Image.fromarray(convolute(array(img), bf1))
# box_filtered_img.save('empire_filter1.jpg')
#
# impulse_filtered_img = Image.fromarray(convolute(img, impulsef))
# impulse_filtered_img.save('empire_filter2.jpg')
#
# gaussian_filtered_img = Image.fromarray(convolute(img, gaussianf))
# gaussian_filtered_img.save('empire_filter3.jpg')

"""
# Drawing the blurred images side by side with original
fig = figure()
fig.add_subplot(141, title='original')
# Showing original
toShow = img.convert('L')
imshow(toShow, cmap=matplotlib.cm.Greys_r)
# Showing gaussianf
toShow = convolute(img.convert('L'), bf1)
fig.add_subplot(142, title='box')
imshow(toShow, cmap=matplotlib.cm.Greys_r)
# Showing impulsef
toShow = convolute(img.convert('L'), gaussianf)
fig.add_subplot(143, title='gaussian')
imshow(toShow, cmap=matplotlib.cm.Greys_r)
# Showing boxf
toShow = convolute(img.convert('L'), impulsef)
fig.add_subplot(144, title='impulse')
imshow(toShow, cmap=matplotlib.cm.Greys_r)
show()

"""
# Using SciPy gaussian blur libraries
# img = array(img.convert('L'))
# gaussian_img = filters.gaussian_filter(img, 1)
# Image.fromarray(gaussian_img).show()
# gaussian_img = filters.gaussian_filter(img, 2)
# Image.fromarray(gaussian_img).show()
# gaussian_img = filters.gaussian_filter(img, 5)
# Image.fromarray(gaussian_img).show()
# The higher the standard deviation, the softer/blurrier the image
