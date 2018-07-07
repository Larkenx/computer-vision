from skimage.io import imread
from scipy.spatial.distance import euclidean
from scipy.cluster.vq import *
import matplotlib.pyplot as plt
from numpy.random import rand
from PIL import Image
from pylab import *
import sift
import operator
from shutil import copyfile

''' Commenting out problem one '''

"""
def appendimages(im1, im2):
    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]
    if rows1 < rows2:
        im1 = np.concatenate((im1, np.zeros((rows2 - rows1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2, np.zeros((rows1 - rows2, im2.shape[1]))), axis=0)
    # if none of these cases they are equal, no filling needed.
    return np.concatenate((im1, im2), axis=1)


def plot_matches(im1, im2, matches):
    im3 = appendimages(im1, im2)
    add = im1.shape[1]
    plt.imshow(im3)

    for i in range(len(matches)):
        rows = [matches[i][0][0], matches[i][1][0]]
        cos = [matches[i][0][1], matches[i][1][1] + add]
        plt.plot(cos, rows, 'k-', lw=1)


path = "/Users/larken/class/computer-vision/a6/"
img1 = array(Image.open(path + "cluttered_desk.png").convert('L'))
img2 = array(Image.open(path + "staple_remover_model.png").convert('L'))
img3 = array(Image.open(path + "elephant_model.png").convert('L'))

# sift.process_image("cluttered_desk.png", 'tmp1.sift')
# sift.process_image("staple_remover_model.png", 'tmp2.sift')
# sift.process_image("elephant_model.png", 'tmp3.sift')

'''Clustering features from all three images. Reading sift features from vlfeat CLI tool from book. '''
l1, d1 = sift.read_features_from_file('tmp1.sift')
l2, d2 = sift.read_features_from_file('tmp2.sift')
l3, d3 = sift.read_features_from_file('tmp3.sift')
k = 250
keypoints = vstack((l1, l2, l3))  # locations of keypoints on input images
features = vstack((d1, d2, d3))  # concatenated 128D vectors
centroids, variance = kmeans(features, k)  # cluster the descriptors/features
code, distance = vq(features, centroids)  # associate features with centroids
colors = [rand(3) for j in range(0, k)]
l1_codes = code[:l1.shape[0]]  # use mapping of code to keypoints to color circled features
l2_codes = code[l1.shape[0]:l2.shape[0] + l1.shape[0]]
l3_codes = code[l2.shape[0] + l1.shape[0]:]


def plot_features(im, locs, code, colors):
    '''Show image with features. input: im (image as array), locs (row, col, scale, orientation of each feature). '''

    def draw_circle(c, r, color):
        t = arange(0, 1.01, .01) * 2 * pi
        x = r * cos(t) + c[0]
        y = r * sin(t) + c[1]
        plot(x, y, color=color)

    imshow(im)
    axis("off")

    for i in range(0, len(locs)):
        draw_circle(locs[i][:2], locs[i][2], colors[code[i]])


'''Plotting features and coloring by cluster ID'''
# plot_features(img1, l1, l1_codes, colors)
# savefig("cluttered_desk_bow.png", bbox_inches=None)
# show()
#
# plot_features(img2, l2, l2_codes, colors)
# savefig("stapler_remover_model_bow.png", bbox_inches=None)
# show()
#
# plot_features(img3, l3, l3_codes, colors)
# savefig("elephant_model_bow.png", bbox_inches=None)
# show()

'''Matching via cluster ID'''
# matches = []
#
# # Go through every descriptor for cluttered desk
# for i in range(0, len(l1_codes)):
#     current = l1_codes[i]
#     current_loc = l1[i][:2] # x,y
#     # Go through every other descriptor in stapler remover
#     for j in range(0, len(l3_codes)):
#         if current == l3_codes[j]:
#             matches.append([current_loc,l3[j][:2]])
#
# plot_matches(img1, img3, matches)
# savefig("matching2.jpg", bbox_inches=None)
# show()

'''Unable to use cv2.findHomography library...'''
"""

# Problem Two: Image retrieval using Bag-of-words
categories = ["airplanes", "camera", "chair", "crab", "crocodile", "elephant", "headphone", "pizza", "soccer_ball", "starfish"]
image_to_sift = []
path = "/Users/larken/class/computer-vision/a6/object_categories/"

# Generating sift files for every image, stored under sift_features/*.sift
# for c in categories:
#     cat_path = path + c + '/'
#     for i in range(1, 10):
#         sift.process_image(cat_path + "image_000{0}.jpg".format(i), "sift_features/" + c + "_{0}.sift".format(i))
#     sift.process_image(cat_path + "image_0010.jpg", "sift_features/" + c + "_10.sift")

# Reading sift features from files to do k-means clustering. Only using the first image for each category
"""
all_kps, all_feats = [], []
sift_path = "/Users/larken/class/computer-vision/a6/sift_features/"
for c in categories:
    keypoints, descriptors = sift.read_features_from_file(sift_path + c + "_1.sift")
    all_kps.append(keypoints)
    all_feats.append(descriptors)

features = vstack(all_feats)
centroids, variance = kmeans(features, 200)
code, distance = vq(features, centroids)  # associate features with centroids
histograms = empty((100, 200))
h_index = -1
for cat in categories:
    for i in range(1,11):
        # print i
        # Read in the feature for the current image. We need to go through all n 128D vectors,
        # and compare each descriptor to all centroids. We take the argmin and update the histogram with the min
        file_name = "{0}{1}_{2}.sift".format(sift_path, cat, i)
        print file_name
        current_location, current_feats = sift.read_features_from_file(file_name)
        # Grab each feature from the images' features
        for f in current_feats:
            # Compare this feature to all centroids
            distance_mapping = [euclidean(f, c) for c in centroids] # mapping of distance to each centroid index
            closest_centroid = argmin(distance_mapping)
            histograms[h_index + i, closest_centroid] += 1 # update the count for this centroid

    h_index += 10

savetxt("complete_histograms.txt", histograms.astype(int))
"""

'''

# Now that we've generated the histogram and saved it to the disk as "complete_histograms.txt", we can easily load it
# into our program. We're ready to start querying images.

# The easiest way to go about this is to create a dictionary that maps filenames to their respective histogram
# The histograms are currently indexed alpha-numerically, but it would be easier to recall what files go with what
# if we can sort them by minimal Euclidean distance.
histograms = loadtxt("complete_histograms.txt")
image_file_paths = []
for c in categories:
    cat_path = path + c + '/'
    for i in range(1, 10):
        image_file_paths.append("{0}image_000{1}.jpg".format(cat_path, i))
    image_file_paths.append(cat_path + "image_0010.jpg")

image_histos = dict(zip(image_file_paths, histograms))

def query_image(imname, h):
    """Takes the image name of a query image and dictionary of image,histogram key-value pairs.
    Returns a list of the 5 best matches to the query image."""
    D = h.copy()
    query_hist = D[imname]
    # Set every value in the copied dictionary to be the distance between the query image and the image bank
    for k, v in D.iteritems():
        if k == imname:
            D[k] = float('inf') # if this is the exact image as the query image, we want to exclude it
        else:
            D[k] = euclidean(query_hist, v)

    sorted_D = sorted(D.items(), key=operator.itemgetter(1)) # http://stackoverflow.com/questions/613183/sort-a-python-dictionary-by-value
    # Sort them ascending (lower values mean more similar images based on histogram)
    return sorted_D

# query_image(image_file_paths[0], image_histos)
save_query_path = "/Users/larken/class/computer-vision/a6/queries/"
for i in range(0,100, 10):
    query_name = image_file_paths[i]
    current_category = categories[i / 10]
    matched_images = query_image(query_name, image_histos)
    matched_file_names = [i[0] for i in matched_images[:5]]
    f_index = 1
    for f in matched_file_names:
        copyfile(f, "{0}{1}_query_result{2}.jpg".format(save_query_path, current_category, f_index))
        f_index += 1

'''












