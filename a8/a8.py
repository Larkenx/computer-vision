from numpy import *
from os import listdir
from os.path import isfile, join
import os
import cv2
import matplotlib.pyplot as plt

project_dir = os.getcwd() + '/'

def draw_flow(im,flow,step=16):
    """ Plot optical flow at sample points
    spaced step pixels apart. """
    h,w = im.shape[:2]
    y,x = mgrid[step/2:h:step,step/2:w:step].reshape(2,-1)
    fx,fy = flow[y,x].T
    # create line endpoints
    lines = vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines = int32(lines)
    # create image and draw
    vis = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    for (x1,y1),(x2,y2) in lines:
        cv2.line(vis,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.circle(vis,(x1,y1),1,(0,255,0), -1)
    return vis

'''
# Generating optical flow Dog frames
dog_path = "/Users/larken/class/computer-vision/a8/dog/"
dog_frames = [f for f in listdir(dog_path)]
flow_dog_path = "/Users/larken/class/computer-vision/a8/flow_dog/"
first_dog_frame = cv2.imread(dog_path + dog_frames[0])
prev_gray = cv2.cvtColor(first_dog_frame, cv2.COLOR_BGR2GRAY)

for f in dog_frames:
    im = cv2.imread(dog_path + f)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray,gray,0.5,1,3,15,3,5,1)
    prev_gray = gray
    flow_viz = draw_flow(gray,flow)
    cv2.imwrite(flow_dog_path + f, flow_viz)
'''

# Generating optical flow Thomas frames
'''
thomas_path = "/Users/larken/class/computer-vision/a8/jpl_thomas/"
thomas_frames = [f for f in listdir(thomas_path) if not f == ".DS_Store"]
flow_thomas_path = "/Users/larken/class/computer-vision/a8/flow_thomas/"
first_thomas_frame = cv2.imread(thomas_path + thomas_frames[0])

prev_gray = cv2.cvtColor(first_thomas_frame, cv2.COLOR_BGR2GRAY)

for f in thomas_frames:
    im = cv2.imread(thomas_path + f)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray,gray,0.5,1,3,15,3,5,1)
    prev_gray = gray
    flow_viz = draw_flow(gray,flow)
    cv2.imwrite(flow_thomas_path + f, flow_viz)
'''

def xyt(sequence_dir):
    fnames = os.listdir(sequence_dir)
    label = int(sequence_dir.split('_')[1])
    # frames = sorted(fnames, key = lambda x: int(x.split('_')[1].split('.')[0]))
    return label, stack([cv2.imread(os.path.join(sequence_dir,f)) for f in fnames])

def get_HOG_of_frame(frame):
    """Extracts histogram of gradients for a single frame. Divide the frame into 5-by-5 spatial regions,
    and then count the number of pixels (in each region) belonging to each of 9 gradient orientation bins."""
    HOGS = []
    im = float32(frame) / 255.0
    gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle = uint8(angle)

    # Create 5x5 spatial regions
    v_offset = int(ceil(angle.shape[1] / 5))
    h_offset = int(ceil(angle.shape[0] / 5))
    for i in range(0, angle.shape[1], v_offset):
        for j in range(0, angle.shape[0], h_offset):
            x_window = i+v_offset
            y_window = j+h_offset
            if i+v_offset >= angle.shape[1]:
                x_window = angle.shape[1]
            if j+h_offset >= angle.shape[0]:
                y_window = angle.shape[0]
            region = angle[j:y_window, i:x_window]
            # For this region, calculate gradient directions and store in HOG
            HOG = zeros_like(arange(9), dtype=uint8)
            for dir in region.flatten():
                if dir > 180:
                    dir = dir % 180
                bin = int(floor(dir / 20))
                if bin == 9:
                    HOG[bin-1] += 1
                else:
                    HOG[bin] += 1

            HOGS.append(HOG)

    result = array(HOGS).flatten()
    return result

def video_HOG_average(sequence_dir):
    label, frames = xyt(sequence_dir)
    average_HOG = zeros_like(arange(255))
    for f in range(0, frames.shape[0], 6):
        # print "Processing frame {0}".format(f)
        res = get_HOG_of_frame(frames[f])
        averaged_HOG =+ res

    return label, averaged_HOG / (frames.shape[0] / 6)

jpl_path = project_dir + "jpl/"
sequence_directories = [jpl_path + dir for dir in listdir(jpl_path) if dir != '.DS_Store']

'''Averaging all of the HOG's from every video. Sampling every 6 frames.
I wrote this to the disk because it took forever!'''
# averaged_videos = []
# for i in range(0, len(sequence_directories)):
#     print "Processing directory {0}".format(sequence_directories[i])
#     label, avg_HOG = video_HOG_average(sequence_directories[i])
#     averaged_videos.append(avg_HOG)
# savetxt("averaged_video_hogs", averaged_videos)

'''Getting activity ID's from the videos'''
# activity_ids = []
# for i in range(0, len(sequence_directories)):
#     sequence_dir = sequence_directories[i]
#     label = int(sequence_dir.split('_')[1])
#     activity_ids.append(label)

# averaged_videos = array(averaged_videos)
# averaged_videos = loadtxt("averaged_video_hogs")
# labeled_avg_vids = zip(activity_ids, averaged_videos)

# labels, first_hog_avg = video_HOG_average(jpl_path + "1_1")
# savetxt("first_hog_avg", first_hog_avg)
# first_hog_avg = loadtxt("first_hog_avg")
# labels, sec_hog_avg = video_HOG_average(jpl_path + "2_2")
# savetxt("sec_hog_avg", sec_hog_avg)
# sec_hog_avg = loadtxt("sec_hog_avg")


def show_hog(hog, fname):
    x_offset = 50
    y_offset = 50
    i = 0
    j = 0
    for h in range(0,225,9):
        spatial_region = hog[h:h+9]
        center = (i + x_offset, j + y_offset)
        for theta in range(0,9):
            bin_count = spatial_region[theta]*5
            angle = theta*20 + 20
            endpoint = cos(angle)*bin_count + i + x_offset, sin(angle)*bin_count + j + y_offset
            print endpoint
            plt.plot([center[0], endpoint[0]], [center[1], endpoint[1]])
        i += 100
        if i >= 600:
            i = 0
            j += 100
    plt.savefig(fname)
    plt.show()

# Visualizing averaged HOG results
# show_hog(first_hog_avg, "1_1_hog")
# show_hog(sec_hog_avg, "2_2_hog")






