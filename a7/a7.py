from scipy.spatial.distance import euclidean
from scipy.cluster.vq import *
from PIL import Image
from pylab import *
import sift
import operator
from shutil import copyfile
from os import listdir
from os.path import isfile, join
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

# Problem Two: Image retrieval using Bag-of-words
categories = ["airplanes", "camera", "chair", "crab", "crocodile", "elephant", "headphone", "pizza", "soccer_ball", "starfish"]
image_to_sift = []
path = "/Users/larken/class/computer-vision/a7/object_categories/"
sift_path = "/Users/larken/class/computer-vision/a7/sift_features/"


# Generating sift files for every image, stored under sift_features/*.sift
'''
for c in categories:
    cat_path = path + c + '/'
    images = [f for f in listdir(cat_path) if isfile(join(cat_path, f))]
    suffix = 1
    for f in images:
        sift.process_image(join(cat_path, f), "{0}{1}_{2}.sift".format(sift_path, c, suffix))
        suffix += 1
'''
# Once we generate the sift data for all of the images, we don't need to do it again
sift_files = [f for f in listdir(sift_path) if isfile(join(sift_path, f))]
'''
# Reading sift features from files to do k-means clustering. Only using the first image for each category
all_kps, all_feats = [], []
# Grab sift features from the first of each category
for c in categories:
    keypoints, descriptors = sift.read_features_from_file(sift_path + c + "_1.sift")
    all_kps.append(keypoints)
    all_feats.append(descriptors)
# Do clustering
features = vstack(all_feats)
centroids, variance = kmeans(features, 200)
code, distance = vq(features, centroids)  # associate features with centroids
# Create histograms for every image
histograms = empty((len(sift_files), 200))
i = 0
for f in sift_files:
    current_location, current_feats = sift.read_features_from_file(sift_path + f)
    # Grab each feature from the images' features
    for f in current_feats:
        # Compare this feature to all centroids
        distance_mapping = [euclidean(f, c) for c in centroids] # mapping of distance to each centroid index
        closest_centroid = argmin(distance_mapping)
        histograms[i, closest_centroid] += 1 # update the count for this centroid

    i += 1

savetxt("complete_histograms.txt", histograms.astype(int))
'''

# histograms = loadtxt("complete_histograms.txt")
# cat_mapping = []
# for f in sift_files:
#     cat_name = f[0 : f.index('_')]
#     cat_mapping.append(cat_name)
#
# histo_cats = zip(cat_mapping, histograms)
#
# neighbors = KNeighborsClassifier(n_neighbors=3)
# for i in range(0, len(histograms)):
#     neighbors.fit(histograms, histograms[i])
#     print neighbors.kneighbors(histograms[i])










