import os, glob
import numpy as np
import scipy.io as sio
import cv2
import face_detection.detect_face as mtcnn
import tensorflow as tf

def processDataset(old_dataset_image_path, old_dataset_landmark_path, new_dataset_path,
                            scale_factor, image_extension='jpg'):
    '''
    Detects and crops faces in images provided in 'old_dataset_image_path', applying
    'scale_factor' to the bounding box found, and stores cropped image in
    'new_dataset_path'. Also corrects offsets provided in 'old_dataset_landmark_path'
    and stores the new offsets (relative to the new bounding boxes) also in
    'new_dataset_path'
    
    Detects face using a MTCNN detector, see
    'https://arxiv.org/abs/1604.02878'
    Implementation by David Sandberg, see
    'https://github.com/davidsandberg/facenet'
    
    :param old_dataset_image_path: (str) Full path to dataset images
    :param old_dataset_landmark_path: (str) Full path to corresponding landmark files
    :param new_dataset_path: (str) Full path to store processed dataset images
    :param scale_factor: (float) Multiplies the square face bounding boxes found
    by the detector ie. if bounding box is 100 pixels wide and 'scale_factor' is 2 it becomes
    200 pixels wide. Fills cropped box with black pixels where scaled box goes out of image bounds
    :param image_extension: (str) Image extension. Defaults to 'jpg'
    :return: --
    '''

    '''
    Load old images and landmarks
    '''


    imageNames = glob.glob(old_dataset_image_path + '/*.' + image_extension)
    imageNamesNoExtension = [n.split('.')[0] for n in imageNames]  # Get filenames without extension
    landmarkNames =[imgName + '_pts.mat' for imgName in imageNamesNoExtension] # Only grab landmarks for which there is
    # a corresponding image

    images = []
    landmarks = []

    for img, lm in zip(imageNames[:25], landmarkNames[:25]):
        images.append(cv2.imread(img + '.' + image_extension)) # Load the image
        landmarks.append(sio.loadmat(old_dataset_landmark_path + '/' + lm)['pts_2d'])


    '''
    Initialize face detector
    '''

    sess = tf.Session()

    # Create the 3 cascade networks
    pnet, rnet, onet = mtcnn.create_mtcnn(sess, None)

    # Set cascade parameters
    minsize = 40  # Minimum window size to be detected in pixels
    threshold = [.6, .7, .7]  # Quality threshold for all cascade levels
    factor = .709 # Image pyramid scaling factor

    # To store cropped images and adjusted landmarks
    cropped_images = []
    adjusted_landmarks = []

    '''
    Detect faces, crop images, adjust landmarks
    '''

    for img, lm in zip(images, landmarks):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # MTCNN trained on RGB images

        # Run MTCNN detector
        # Output 'faces' is a n-by-m matrix where n is the num of faces detected and m=5 has bounding box info for each
        # face in the format xmin, ymin, xmax, ymax, confidence
        faces, _ = mtcnn.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

        # Get highest confidence face for this image
        bbox = faces[np.argmax([faces[:,-1]]),:-1].astype('int32')

        # Square the found box coordinates
        width = bbox[2]-bbox[0]
        height = bbox[3]-bbox[1]
        side = np.max([width, height])

        if width>height:
            bbox[1] -= side/2
            bbox[3] += side/2
        else:
            bbox[0] -= side/2
            bbox[2] += side/2

        # Scale box
        bbox[0] -= side/2 * (scale_factor - 1)
        bbox[2] += side/2 * (scale_factor - 1)
        bbox[1] -= side/2 * (scale_factor - 1)
        bbox[3] += side/2 * (scale_factor - 1)

        # Check if box goes outside image
        img_height, img_width = img.shape

        x_origin_offset = 0
        y_origin_offset = 0
        x_end_offset = 0
        y_end_offset = 0
        x_min_crop = bbox[0]
        y_min_crop = bbox[1]
        x_max_crop = bbox[2]
        y_max_crop = bbox[3]

        if bbox[0] < 0:
            x_min_crop = 0
            x_origin_offset = bbox[0]
        if bbox[1] < 0:
            y_min_crop = 0
            y_origin_offset = bbox[1]
        if bbox[2] > img_width:
            x_max_crop = img_width
            x_end_offset = bbox[2] - img_width
        if bbox[3] > img_height:
            y_max_crop = img_height
            y_end_offset = bbox[3] - img_height

        # Crop image
        cropped_img = np.zeros((side * scale_factor, side * scale_factor)) # holder for cropped img
        cropped_img[y_origin_offset:-y_end_offset, x_origin_offset:-x_end_offset] = img[y_min_crop:y_max_crop,
                                                                                    x_min_crop:x_max_crop]

        # Adjust landmarks
        lm[:,0] -= bbox[0]
        lm[:,1] -= bbox[1]

        cropped_images.append(cropped_img)
        adjusted_landmarks.append(lm)

    '''
    Save cropped faces and adjusted landmarks
    '''

    # Save in specified directory
    os.chdir(new_dataset_path)
    for cropped_img, adjusted_lm, imgNameNoExtension in zip(cropped_images, adjusted_landmarks, imageNamesNoExtension[:25]):
        cv2.imwrite(imgNameNoExtension + '.' + image_extension, cropped_img)
        sio.savemat(imgNameNoExtension + '_pts.mat', dict([('pts_2d', adjusted_lm)]))

def main():
    '''
    Testing for processDataset
    :return: --
    '''

    old_dataset_image_path = '/Users/miguelrita/Documents/Paper Implementations/esr/data/AFW'
    old_dataset_landmark_path = '/Users/miguelrita/Documents/Paper Implementations/esr/data/landmarks/AFW'
    new_dataset_path = '/Users/miguelrita/Documents/Paper Implementations/esr/processedDataset'
    scale_factor = 1.1

    processDataset(old_dataset_image_path, old_dataset_landmark_path, new_dataset_path, scale_factor)

if __name__ == '__main__':
    main()