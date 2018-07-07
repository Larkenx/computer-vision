from numpy import *
import cv2
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

# setup video capture
cap = cv2.VideoCapture(0)
ret,im = cap.read()
prev_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

while True:
    ret,im = cap.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray,gray,0.5,1,3,15,3,5,1)
    prev_gray = gray
    flow_viz = draw_flow(gray,flow)
    cv2.imshow('Optical flow', flow_viz)
    frame = cv2.flip(flow_viz, 0)
    key = cv2.waitKey(33)
    if key == ord('q'):
        print key
        break

