from pylab import *
from numpy import *
from matplotlib import collections as mc
from PIL import Image
import sift
import cv2

def generate_sift_image(imname):
    im = array(Image.open(imname))
    removed_ext = imname[0:imname.index(".")]
    sift.process_image(imname, removed_ext + '.sift')
    l1,d1 = sift.read_features_from_file(removed_ext + '.sift')
    print d1.shape

    figure()
    gray()
    sift.plot_features(im, l1, circle=True)
    savefig(removed_ext + "_sift.png", bbox_inches="tight")
    show()

# Exercise 1
# generate_sift_image("box.png")
# generate_sift_image("box_in_scene.png")

# Exercise 2
def dist(v1, v2):
    """Finds the Euclidean distance between two vectors. """
    sum = 0
    for i in range(0, v1.shape[0]):
        sum += (v1[i] - v2[i])**2

    return sqrt(sum)

# Exercise 3
def pairwise_dist(im1, im2):
    """Generates a distance matrix of f1.shape[0] by f2.shape[0],
    comparing the pairwise distance between every feature from im1 to im2 """
    D = empty(shape=(im1.shape[0], im2.shape[0]))
    for i in range(0, im1.shape[0]):
        v1 = im1[i]
        for j in range(0, im2.shape[0]):
            v2 = im2[j]
            D[i,j] = dist(v1, v2)

    return D

# Exercise 4
def plot_local_feats(imname1, imname2):
    aim1 = array(Image.open(imname1).convert('L'))
    aim2 = array(Image.open(imname2).convert('L'))
    offset = aim2.shape[0] - aim1.shape[0]
    aim1 = concatenate((aim1, zeros((offset,aim1.shape[1]))), axis=0)
    # We have to add a black padding/border to make the two images
    # the same size so they can be plotted together

    sift.process_image(imname1, 'tmp1.sift')
    sift.process_image(imname2, 'tmp2.sift')
    l1, d1 = sift.read_features_from_file('tmp1.sift')
    l2, d2 = sift.read_features_from_file('tmp2.sift')
    print l1.shape
    print d1.shape
    print d1

    # Now we have the two features, we can build the distance matrix
    D = pairwise_dist(d1, d2)

    # for each feature in d1, find the minimum distance in d2 features that matches
    lines = empty((50, 2, 2)) # [ [[x1,y1], [x2,y2]], ...]
    li = 0
    min_list = empty((D.shape[0]))

    for i in range(0, D.shape[0]):
        min_list[i] = min(D[i])

    min_list = sort(min_list)

    for i in range(0, D.shape[0]):
        for j in range(0, D.shape[1]):
            if (D[i,j] in min_list[:50]):
                lines[li][0] = l1[i,:2]
                lines[li][1] = l2[j,:2]
                lines[li][1][0] += aim1.shape[1]
                li += 1


    fig, ax = subplots()
    ax.add_collection(mc.LineCollection(lines))
    plot_img = concatenate((aim1, aim2), axis=1)
    imshow(plot_img)
    axis("off")
    savefig("matching.png", bbox_inches=None, dpi=fig.dpi)
    show()

plot_local_feats("box.png", "box_in_scene.png")

def plot_circs_match(imname1, imname2, circle=False):

    def draw_circle(c,r):
        t = arange(0,1.01,.01)*2*pi
        x = r*cos(t) + c[0]
        y = r*sin(t) + c[1]
        plot(x,y,"b",linewidth=2)

    aim1 = array(Image.open(imname1).convert('L'))
    aim2 = array(Image.open(imname2).convert('L'))
    offset = aim2.shape[0] - aim1.shape[0]
    aim1 = concatenate((aim1, zeros((offset, aim1.shape[1]))), axis=0)

    sift.process_image(imname1, 'tmp1.sift')
    sift.process_image(imname2, 'tmp2.sift')
    l1, d1 = sift.read_features_from_file('tmp1.sift')
    l2, d2 = sift.read_features_from_file('tmp2.sift')

    # Now we have the two features, we can build the distance matrix
    D = pairwise_dist(d1, d2)

    # for each feature in d1, find the minimum distance in d2 features that matches
    lines = empty((50, 2, 2)) # [ [[x1,y1], [x2,y2]], ...]
    li = 0
    min_list = empty((D.shape[0]))

    for i in range(0, D.shape[0]):
        min_list[i] = min(D[i])

    min_list = sort(min_list)

    for i in range(0, D.shape[0]):
        for j in range(0, D.shape[1]):
            if (D[i,j] in min_list[:50]):
                lines[li][0] = l1[i,:2]
                lines[li][1] = l2[j,:2]
                lines[li][1][0] += aim1.shape[1]
                li += 1


    fig, ax = subplots()
    ax.add_collection(mc.LineCollection(lines))

    imshow(concatenate((aim1, aim2), axis=1))
    for p in l1:
        draw_circle(p[:2],p[2])

    for p in l2:
        draw_circle(p[:2] + [aim1.shape[1], 0], p[2])

    axis("off")
    savefig("matching_with_circles.png", bbox_inches="tight")
    show()

# plot_circs_match("box.png", "box_in_scene.png", circle=True)

# Exercise 5
def efive():
    img1 = cv2.imread('/Users/larken/class/computer-vision/a5/box.png',0) # queryImage
    img2 = cv2.imread('/Users/larken/class/computer-vision/a5/box_in_scene.png',0) # trainImage
    orb = cv2.SIFT()
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    print des1.shape
    print des2.shape

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)
    print matches

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:50], None)
    # Image.fromarray(uint8(img3)).save('matching2.jpg')
    # imshow(img3)
    # show()

# efive()

# Exercise 6
def esix():
    MIN_MATCH_COUNT = 10

    img1 = cv2.imread('/Users/larken/class/computer-vision/a5/box.png',0) # queryImage
    img2 = cv2.imread('/Users/larken/class/computer-vision/a5/box_in_scene.png',0) # trainImage
    orb = cv2.ORB()

    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

# esix()

