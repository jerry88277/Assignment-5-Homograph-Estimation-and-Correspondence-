# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 16:03:25 2022

@author: Jerry

reference:
    https://github.com/rmislam/PythonSIFT
    https://github.com/adumrewal/SIFTImageSimilarity
    https://gist.github.com/tigercosmos/90a5664a3b698dc9a4c72bc0fcbd21f4
"""
import os
import cv2
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import pysift

import random
import time
from tqdm import tqdm

# In[] def
# Resize images to a similar dimension
# This helps improve accuracy and decreases unnecessarily high number of keypoints

def computeSIFT(image):
    return pysift.computeKeypointsAndDescriptors(image)

def computeSIFT_CV2(image):
    sift = cv2.xfeatures2d.SIFT_create()
    return sift.detectAndCompute(image, None)

def imageResizeTrain(image):
    maxD = 1024
    height,width = image.shape
    aspectRatio = width/height
    if aspectRatio < 1:
        newSize = (int(maxD*aspectRatio),maxD)
    else:
        newSize = (maxD,int(maxD/aspectRatio))
    image = cv2.resize(image,newSize)
    return image

def imageResizeTest(image):
    maxD = 1024
    if len(image.shape) == 2:
        height, width = image.shape
    if len(image.shape) == 3:
        height, width, channel = image.shape
    
    aspectRatio = width/height
    if aspectRatio < 1:
        newSize = (int(maxD*aspectRatio),maxD)
    else:
        newSize = (maxD,int(maxD/aspectRatio))
    image = cv2.resize(image,newSize)
    return image
    
def fetchKeypointFromFile(index):
    filepath = "data/keypoints/" + str(image_list[index].split('.')[0]) + ".pkl"
    keypoint = []
    file = open(filepath,'rb')
    deserializedKeypoints = pickle.load(file)
    type(deserializedKeypoints)
    file.close()
    for point in deserializedKeypoints:
        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], size=point[1], angle=point[2], response=point[3], octave=point[4], class_id=point[5])
        keypoint.append(temp)
    return keypoint

def fetchDescriptorFromFile(index):
    filepath = "data/descriptors/" + str(image_list[index].split('.')[0]) + ".pkl"
    file = open(filepath,'rb')
    descriptor = pickle.load(file)
    file.close()
    return descriptor

def calculateResultsFor(index_1, index_2):
    keypoint1 = fetchKeypointFromFile(index_1)
    descriptor1 = fetchDescriptorFromFile(index_1)
    keypoint2 = fetchKeypointFromFile(index_2)
    descriptor2 = fetchDescriptorFromFile(index_2)
    matches = calculateMatches(descriptor1, descriptor2)
    score = calculateScore(len(matches),len(keypoint1),len(keypoint2))
    plot = getPlotFor(index_1, index_2, keypoint1, keypoint2, matches)
    print(len(matches),len(keypoint1),len(keypoint2),len(descriptor1),len(descriptor2))
    print(score)
    fig, ax = plt.subplots(figsize = (20, 16))
    plt.imshow(plot), plt.show()
    plt.axis('off')
    plt.savefig(f'results/Own_SIFT_image_combined{index_1}_{index_2}.png')
    plt.close()
    
def getPlotFor(i,j,keypoint1,keypoint2,matches):
    image1 = imageResizeTest(cv2.imread("data/images/" + image_list[i]))
    image2 = imageResizeTest(cv2.imread("data/images/" + image_list[j]))
    return getPlot(image1,image2,keypoint1,keypoint2,matches)

def calculateScore(matches,keypoint1,keypoint2):
    return 100 * (matches/min(keypoint1,keypoint2))

def calculateMatches(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    topResults1 = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            topResults1.append([m])
            
    matches = bf.knnMatch(des2, des1, k=2)
    topResults2 = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            topResults2.append([m])
    
    topResults = []
    for match1 in topResults1:
        match1QueryIndex = match1[0].queryIdx
        match1TrainIndex = match1[0].trainIdx

        for match2 in topResults2:
            match2QueryIndex = match2[0].queryIdx
            match2TrainIndex = match2[0].trainIdx

            if (match1QueryIndex == match2TrainIndex) and (match1TrainIndex == match2QueryIndex):
                topResults.append(match1)
    return topResults

def getPlot(image1,image2,keypoint1,keypoint2,matches):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    matchPlot = cv2.drawMatchesKnn(image1, keypoint1, image2, keypoint2, matches, None, [255,255,255], flags=2)
    return matchPlot

def homography(pairs): # pairs = points
    rows = []
    for i in range(pairs.shape[0]):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)
        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)
    U, s, V = np.linalg.svd(rows)
    H = V[-1].reshape(3, 3)
    H = H/H[2, 2] # standardize to let w*H[2,2] = 1
    return H

def random_point(matches, k=4):
    idx = random.sample(range(len(matches)), k)
    point = [matches[i] for i in idx ]
    return np.array(point)

def get_error(points, H):
    num_points = len(points)
    all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
    all_p2 = points[:, 2:4]
    estimate_p2 = np.zeros((num_points, 2))
    for i in range(num_points):
        temp = np.dot(H, all_p1[i])
        estimate_p2[i] = (temp/temp[2])[0:2] # set index 2 to 1 and slice the index 0, 1
    # Compute error
    errors = np.linalg.norm(all_p2 - estimate_p2 , axis=1) ** 2

    return errors

def ransac(matches, threshold, iters):
    num_best_inliers = 0
    
    for i in range(iters):
        points = random_point(matches)
        H = homography(points)
        
        #  avoid dividing by zero 
        if np.linalg.matrix_rank(H) < 3:
            continue
            
        errors = get_error(matches, H)
        idx = np.where(errors < threshold)[0]
        inliers = matches[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_H = H.copy()
            
    print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
    return best_inliers, best_H

def plot_matches(matches, total_img):
    match_img = total_img.copy()
    offset = total_img.shape[1]/2
    fig, ax = plt.subplots(figsize = (20, 16))
    ax.set_aspect('equal')
    ax.imshow(np.array(match_img).astype('uint8')) #　RGB is integer type
    
    ax.plot(matches[:, 0], matches[:, 1], 'xr')
    ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr')
     
    ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]],
            'r', linewidth=0.5)

    plt.show()

def matcher(kp1, des1, img1, kp2, des2, img2, threshold):
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2) # des1 = descriptor1  des2 = descriptor2

    # Apply ratio test
    good = []
    for m,n in matches:
        print(f'm:{m.distance}')
        print(f'n:{n.distance}')
        if m.distance < threshold * n.distance:
            good.append([m])

    matches = []
    for pair in good:
        matches.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt))

    matches = np.array(matches)
    return matches

def matcher_2(kp1, des1, img1, kp2, des2, img2, threshold):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    topResults1 = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            topResults1.append([m])
            
    matches = bf.knnMatch(des2, des1, k=2)
    topResults2 = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            topResults2.append([m])
    
    topResults = []
    for match1 in topResults1:
        match1QueryIndex = match1[0].queryIdx
        match1TrainIndex = match1[0].trainIdx

        for match2 in topResults2:
            match2QueryIndex = match2[0].queryIdx
            match2TrainIndex = match2[0].trainIdx

            if (match1QueryIndex == match2TrainIndex) and (match1TrainIndex == match2QueryIndex):
                topResults.append(list(kp1[match1[0].queryIdx].pt + kp2[match1[0].trainIdx].pt))

    matches = np.array(topResults)
    
    return matches



def read_image(path):
    img = cv2.imread(path)
    img_gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_gray, img, img_rgb

def plot_sift(gray, rgb, kp):
    tmp = rgb.copy()
    img = cv2.drawKeypoints(gray, kp, tmp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img
    


# In[] process

# image_list = ["taj1.jpeg", "taj2.jpeg", "eiffel1.jpeg", "eiffel2.jpeg", "liberty1.jpeg",
#               "liberty2.jpeg", "robert1.jpeg", "tom1.jpeg", "ironman1.jpeg", "ironman2.jpeg",
#               "ironman3.png","darkknight1.jpeg","darkknight2.jpeg", "book1.jpeg", "book2.jpeg"]

image_list = ['book1.jpg', 'book2.jpg', 'book3.jpg', 'scene.jpg']

image_floder  = r'D:\NCKU\Course\Digital Image Processing And Computer vision\HW5\Homograph Estimation and Correspondence\SIFTImageSimilarity-master\data\images'

# We use grayscale images for generating keypoints
imagesBW = []
for imageName in image_list:
    # imagePath = "data/images/" + str(imageName)
    imagePath = os.path.join(image_floder, imageName)
    imagesBW.append(imageResizeTrain(cv2.imread(imagePath, 0)))

keypoints = []
descriptors = []

for index, image in tqdm(enumerate(imagesBW)):
    print("Starting for image: " + image_list[index])
    # keypointTemp, descriptorTemp = computeSIFT_CV2(image)
    keypointTemp, descriptorTemp = computeSIFT(image)
    keypoints.append(keypointTemp)
    descriptors.append(descriptorTemp)
    print("Ending for image: " + image_list[index])

    
for index, keypoint in tqdm(enumerate(keypoints)):
    deserializedKeypoints = []
    filepath = "data/keypoints/" + str(image_list[index].split('.')[0]) + ".pkl"
    for point in keypoint:
        temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
        deserializedKeypoints.append(temp)
    with open(filepath, 'wb') as fp:
        pickle.dump(deserializedKeypoints, fp)    

    
for index, descriptor in tqdm(enumerate(descriptors)):
    filepath = "data/descriptors/" + str(image_list[index].split('.')[0]) + ".pkl"
    with open(filepath, 'wb') as fp:
        pickle.dump(descriptor, fp)



# In[] result
calculateResultsFor(3, 0)
calculateResultsFor(3, 1)
calculateResultsFor(3, 2)

# In[] plot key points

images_rgb = []
images_gray = []
for imageName in image_list:
    imagePath = os.path.join(image_floder, imageName)
    img_gray, img, img_rgb = read_image(imagePath)
    # imagesBW.append(imageResizeTrain(cv2.imread(imagePath, 0)))
    images_gray.append(img_gray)
    images_rgb.append(img_rgb)

for index in range(4):

    left_gray = imageResizeTest(images_gray[index])
    # right_gray = imageResizeTest(images_gray[index])
    
    left_rgb = imageResizeTest(imagesBW[index])
    # right_rgb = imageResizeTest(imagesBW[index])
    
    keypoint1 = fetchKeypointFromFile(index)
    descriptor1 = fetchDescriptorFromFile(index)
    # keypoint2 = fetchKeypointFromFile(index)
    # descriptor2 = fetchDescriptorFromFile(index)
    

    kp_left_img = plot_sift(left_gray, left_rgb, keypoint1)
    # kp_right_img = plot_sift(right_gray, right_rgb, keypoint2)
    # total_kp = np.concatenate((kp_left_img, kp_right_img), axis=1)
    plt.imshow(kp_left_img)
    plt.axis('off')
    if index == 0:
        plt.savefig('results/keypoints_scene.png')
    else:
        plt.savefig(f'results/keypoints_book{index}.png')
    plt.close()


# In[] RANSAC

imagesBW = []
for imageName in image_list:
    imagePath = os.path.join(image_floder, imageName)
    img_gray, img, img_rgb = read_image(imagePath)
    # imagesBW.append(imageResizeTrain(cv2.imread(imagePath, 0)))
    imagesBW.append(img_rgb)

threshold_list = [0.7, 0.5, 0.3]
iteration_list = [1000, 2000, 3000]

spend_time = []

for i_th in threshold_list:
    for i_iters in iteration_list:
        threshold = i_th
        iteration = i_iters

        for index in range(3):
        
            left_rgb = imageResizeTest(imagesBW[3])
            right_rgb = imageResizeTest(imagesBW[index])
            
            keypoint1 = fetchKeypointFromFile(3)
            descriptor1 = fetchDescriptorFromFile(3)
            keypoint2 = fetchKeypointFromFile(index)
            descriptor2 = fetchDescriptorFromFile(index)
            
            # 開始測量
            start = time.process_time()
            # matches = matcher(keypoint1, descriptor1, left_rgb, keypoint2, descriptor2, right_rgb, threshold)
            matches = matcher_2(keypoint1, descriptor1, left_rgb, keypoint2, descriptor2, right_rgb, threshold)
            
            inliers, H = ransac(matches, 0.5, iteration)
            
            # 結束測量
            end = time.process_time()
            # 輸出結果
            # print("RANSAC執行時間：%f 秒" % (end - start))
            spend_time.append([threshold, iteration, image_list[index], (end - start)])
            
            total_img = np.concatenate((left_rgb, right_rgb), axis=1)
            plot_matches(inliers, total_img)
            plt.axis('off')
            plt.savefig(f'results/RANSAC_th{threshold}_iter{iteration}_book{index + 1}.png')
            plt.close()
    
spend_time = pd.DataFrame(spend_time, columns = ['threshold', 'iterations', 'target', 'computing time'])
spend_time.to_csv('spend_time.csv', index = None)
