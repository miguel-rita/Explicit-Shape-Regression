import os, random, glob, tqdm
import numpy as np
import cv2 as cv2
from scipy.linalg import orthogonal_procrustes
from scipy.io import loadmat

# Load two shapes
shape_1_path = '../processed_data/AFW/AFW_134212_1_0_pts.mat'
shape_2_path = '../processed_data/AFW/AFW_815038_2_0_pts.mat'
shape_1 = loadmat(shape_1_path)['pts_2d']
shape_2 = loadmat(shape_2_path)['pts_2d']

shape_1 = np.array([[i,i] for i in range(0,200,10)])
shape_2 = shape_1 + np.random.uniform(-10,10,size=shape_1.shape)

# Rotate one of the shapes theta deg
theta = -np.pi / 4
shape_2 = np.matmul(shape_2, np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]]))

# Create shape 3 from procrustes
normalizing_transform, _ = orthogonal_procrustes(shape_2, shape_1)
shape_3 = np.matmul(shape_2, normalizing_transform)

# Display both shapes on a white background
white_background = np.ones((450,450,3)) * 255

landmarkRadius = 2
landmarkColor1 = (255, 0, 0)
landmarkColor2 = (0, 255, 0)
landmarkColor3 = (0, 0, 255)



visualizerRunning = True
while visualizerRunning:

    for lm in shape_1:
        currentImage = cv2.circle(white_background, (int(lm[0]), int(lm[1])), landmarkRadius, landmarkColor1, -1)
    for lm in shape_2:
        currentImage = cv2.circle(currentImage, (int(lm[0]), int(lm[1])), landmarkRadius, landmarkColor2, -1)
    for lm in shape_3:
        currentImage = cv2.circle(currentImage, (int(lm[0]), int(lm[1])), landmarkRadius, landmarkColor3, -1)

    cv2.imshow('Understanding procrustes', white_background)

    # Process keyboard input
    key = cv2.waitKey(1) & 0xFF
    # Note: 'ord' gets ASCII code
    if key == ord('a'):
        pass
    if key == ord('d'):
        pass
    if key == ord('s'):
        pass
    if key == ord('q'):  # Quit visualization
        print('Quitting visualization ...')
        visualizerRunning = False

