import os, glob
import numpy as np
import scipy.io as sio
import cv2

def visualizer(images, landmarks, landmarks2 = np.array([])):
    '''
    Helper function to visualize precomputed landmarks on a series of images
    Can visualize 2 sets of landmarks per image
    Press 'S' to toggle landmarks on/off
    Press 'A'/'D' for previous/next image
    Press 'Q' to quit visualizer
      
    :param images: (list of np arrays) list of BGR input images to visualize
    :param landmarks: (list of np arrays) list of landmark matrices, where the
    number of rows per matrix equals the number of landmarks and the number of columns is 2, for x and y
    :param landmarks2: optional second list of landmarks per image, same as above
    :return: --
    '''

    # Cosmetic parameters
    landmarkRadius = 2
    landmarkColor1 = (0,255,0)
    landmarkColor2 = (255,0,0)

    # Flow control params
    visualizerRunning = True
    currentImageNum = 0

    isDrawingOn = True
    if len(landmarks2) == 0:
        drawSecondSetLandmarks = False # No second set of landmarks was provided
    else:
        drawSecondSetLandmarks = True

    while visualizerRunning:
        currentSourceImage = images[currentImageNum]
        currentImage = np.copy(currentSourceImage) # Will have edits ie. landmarks drawn on top

        if isDrawingOn:
            for lm in landmarks[currentImageNum]:
                currentImage = cv2.circle(currentImage, (lm[0], lm[1]), landmarkRadius, landmarkColor1, -1)
        if isDrawingOn and drawSecondSetLandmarks:
            for lm in landmarks2[currentImageNum]:
                currentImage = cv2.circle(currentImage, (lm[0], lm[1]), landmarkRadius, landmarkColor2, -1)

        if not isDrawingOn:
            cv2.imshow('Visualizer', currentSourceImage)
        else:
            cv2.imshow('Visualizer', currentImage)

        # Process keyboard input
        key = cv2.waitKey(1) & 0xFF
        # Note: 'ord' gets ASCII code
        if key == ord('a') and currentImageNum>0:
            currentImageNum -= 1
        if key == ord('d') and currentImageNum<len(images)-1:
            currentImageNum += 1
        elif key == ord('s'):
            isDrawingOn = not isDrawingOn
        elif key == ord('q'):  # Quit visualization
            visualizerRunning = False

def main():
    '''
    Visualizer testing function 
    '''

    # Load a couple of images and respective landmarks for testing

    numImages = 1000
    datasetName = 'AFW' # pick numImages first images from this dataset
    imageExtension = 'jpg'

    os.chdir('../data/'+datasetName) # switch to dataset dir
    imageNames = glob.glob('*.'+imageExtension)
    imageNames = [n.split('.')[0] for n in imageNames] # Get filenames without extension

    images = []
    landmarks = []

    for img in imageNames[:numImages]:
        images.append(cv2.imread(img+'.'+imageExtension)) # Load the image
        landmarks.append(sio.loadmat('../landmarks/'+datasetName+'/'+img+'_pts.mat')['pts_2d'])

    # Compute data set mean shape and provide it as second landmark set (constant, in this case)
    meanShape = np.mean(landmarks, axis=0)
    landmarks2 = [meanShape for lm in landmarks]

    visualizer(images, landmarks, landmarks2)

if __name__ == '__main__':
    main()
