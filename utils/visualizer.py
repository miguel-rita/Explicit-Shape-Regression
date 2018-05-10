import os, glob
import numpy as np
import scipy.io as sio
import cv2
import face_detection.detect_face_batch as mtcnn

def visualizer(images, bboxes=None, landmarks=None):
    '''
    Helper function to visualize precomputed landmarks and/or detected bounding boxes on a series of images
    Press 'S' to toggle landmarks/bboxes on/off
    Press 'A'/'D' for previous/next image
    Press 'Q' to quit visualizer
      
    :param images: (list of np arrays) list of BGR input images to visualize
    :param bboxes: (list of np arrays) list of faces found per image (optional)
    :param landmarks: (list of np arrays) list of landmark matrices, where the
    number of rows per matrix equals the number of landmarks and the number of columns is 2, for x and y (optional)
    :return: --
    '''

    # Cosmetic parameters
    landmarkRadius = 2
    landmarkColor1 = (0, 255, 0)
    bbox_color = (0, 0, 255)
    bbox_thickness = 2

    # Flow control params
    visualizerRunning = True
    currentImageNum = 0

    isDrawingOn = True

    while visualizerRunning:
        currentSourceImage = images[currentImageNum]
        currentImage = np.copy(currentSourceImage) # Will have edits ie. landmarks drawn on top

        if isDrawingOn:
            if not landmarks is None:
                for lm in landmarks[currentImageNum]:
                    currentImage = cv2.circle(currentImage, (lm[0], lm[1]), landmarkRadius, landmarkColor1, -1)
            if not bboxes is None:
                for face in bboxes[currentImageNum].astype('int32'):
                    currentImage = cv2.rectangle(currentImage, (face[0], face[1]), (face[2], face[3]), bbox_color,
                                             bbox_thickness)

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



def test1():
    '''
    Visualizer testing function 1
    '''

    # Load a couple of image paths

    numImages = 1000
    datasetName = 'AFW/' # pick numImages first images from this dataset
    imageExtension = '.jpg'

    os.chdir('../data/'+datasetName) # switch to dataset dir
    imageNames = glob.glob('*'+imageExtension)
    imageNames = [n.split('.')[0] for n in imageNames] # Get filenames without extension

    images = []

    for img in imageNames[:numImages]:
        images.append(cv2.imread(img+imageExtension)) # Load the image

    # Detect faces in images
    faces, names = mtcnn.detect_face_batch(image_list=images, image_names=imageNames[:numImages],
                                           are_images_rgb=False)

    landmarks = []
    images_with_faces = []

    # Load images and landmarks for images with detected faces
    os.chdir('..')
    for img_name in names:
        # Load image
        images_with_faces.append(cv2.imread(datasetName+img_name+imageExtension))
        # Load landmarks
        landmarks.append(sio.loadmat('landmarks/'+datasetName+img_name+'_pts.mat')['pts_2d'])

    # Compute data set mean shape
    meanShape = np.mean(landmarks, axis=0)

    visualizer(images_with_faces, faces, landmarks)

def test2():
    '''
    Visualizer testing function 2
    '''

    # Load processed images and landmarks

    datasetName = 'AFW/' # pick numImages first images from this dataset
    imageExtension = '.jpg'

    os.chdir('../processed_data/'+datasetName) # switch to dataset dir
    image_names = glob.glob('*'+imageExtension)
    image_names = [n.split('.')[0] for n in image_names] # Get filenames without extension

    images = []
    landmarks = []

    for img_name in image_names:
        # Load image
        images.append(cv2.imread(img_name+imageExtension))
        # Load landmarks
        landmarks.append(sio.loadmat(img_name+'_pts.mat')['pts_2d'])

    # Compute data set mean shape
    meanShape = np.mean(landmarks, axis=0)

    visualizer(images, None, landmarks)

if __name__ == '__main__':
    test2()
