# -*- coding: utf-8 -*-
"""Machine Learning Exercise 16 - Computer Vision Features - Emerson Ham
This exercise covers a variety of features commonly used for computer vision tracking
Since we use openCV, most of this code is derived from examples in the OpenCV documentation.
"""

#! pip install opencv-python opencv-contrib-python

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import urllib.request

# Code from https://www.pyimagesearch.com/2015/03/02/convert-url-to-image-with-python-and-opencv/
def url_to_image(url):
    # Downloads the image, convert it to a NumPy array, and then reads it into OpenCV's format

    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
    # return the image
    return image

# Download image
url = "https://www.goodracks.com/media/newSimplePicture.jpg"
img = url_to_image(url)

plt.imshow(img)
plt.show()

# Convert to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
plt.imshow(gray)
plt.show()

# Find the Harris Corners
harris = cv2.cornerHarris(gray,2,3,0.04)
harrisImg = np.copy(img)
threshold = 0.01*harris.max()
harrisImg[harris>threshold]=[255,0,0]

plt.imshow(harrisImg)
plt.show()

# Calculate the Shi-Tomasi corners with a maximum number of 30 corners and set them to corners
corners = cv2.goodFeaturesToTrack(gray,30,0.01,10)
corners = np.int0(corners)
stImg = np.copy(img)
for i in corners:
    x,y = i.ravel()
    cv2.circle(stImg,(x,y),3,255,-1)
plt.imshow(stImg)
plt.show()

# SIFT and SURF features may be patented
try:
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    siftImg=cv2.drawKeypoints(gray, kp, img)
    plt.imshow(siftImg)
except Exception as e:
    print(e)

try:
    surf = cv2.xfeatures2d.SURF_create(400)
    kp=surf.detect(gray, None)
    surfImg = cv2.drawKeypoints(img,kp,None,(255,0,0),4) # Will draw red dots at the keypoints
    plt.imshow(surfImg)
    
except Exception as e:
      print(e)

# Create a FAST feature detector and draw dots on fast features
fast = cv2.FastFeatureDetector_create()
# find and draw the keypoints
fastKp = fast.detect(img,None)
fastImg = cv2.drawKeypoints(img, fastKp, None, color=(255,0,0))

plt.imshow(fastImg)
plt.show()

# Create BRIEF descriptors
CenSurE = cv2.xfeatures2d.StarDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
censureKp = CenSurE.detect(img,None)
censureKp, briefDes = brief.compute(img, censureKp)

CenSurEImg =  cv2.drawKeypoints(img, censureKp, None, color=(255,0,0))
plt.imshow(CenSurEImg)
plt.show()

print(f"We have {len(censureKp)} key points each with a 32-dim vector resulting in a descriptor of shape {briefDes.shape}.")

# Create the ORB detector
orb = cv2.ORB_create()
orbKp = orb.detect(img,None)
orbKp, orbDes = orb.compute(img, orbKp)
orbImg = cv2.drawKeypoints(img, orbKp, None, color=(0,255,0), flags=0)

plt.imshow(orbImg)
plt.show()