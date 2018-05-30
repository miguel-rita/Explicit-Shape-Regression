import os, math, tqdm
import numpy as np
import scipy.io as sio
import cv2
import face_detection.detect_face_batch as mtcnn
import utils.extract_chip as extract_chip

def process_dataset(old_dataset_image_path, old_dataset_landmark_path, new_dataset_path,
                            scale_factor, image_extension='jpg', final_image_size=(450, 450)):
    '''
    Detects and crops faces in images provided in 'old_dataset_image_path', applying
    'scale_factor' to the bounding box found, and stores cropped image in
    'new_dataset_path'. Also corrects offsets provided in 'old_dataset_landmark_path'
    and stores the new offsets (relative to the new bounding boxes) also in
    'new_dataset_path'
    
    :param old_dataset_image_path: (str) Full path to dataset images
    :param old_dataset_landmark_path: (str) Full path to corresponding landmark files
    :param new_dataset_path: (str) Full path to store processed dataset images
    :param scale_factor: (float) Multiplies the square face bounding boxes found
    by the detector ie. if bounding box is 100 pixels wide and 'scale_factor' is 2 it becomes
    200 pixels wide. Fills cropped box with black pixels where scaled box goes out of image bounds
    :param image_extension: (str, optional) Image extension. Defaults to 'jpg'
    :param final_image_size: (tuple of ints) Final desired image size in (x dim in pixels, y dim in pixels).
    All images are resized to 'final_image_size'
    Will process all images if nothing is specified. Useful for debugging
    :return: --
    '''

    '''
    Load old images, get their detected faces (bboxes) and landmarks (pre-processing)
    '''

    # Detect face bounding boxes (bboxes) in preprocessed images
    (image_bboxes, image_names) = mtcnn.detect_face_batch_from_directory(old_dataset_image_path,
                                                                         image_extension=image_extension)

    # Only grab landmarks for which there is a corresponding image
    landmark_names =[img_name.split('.')[-2] + '_pts.mat' for img_name in image_names]

    preprocessed_images = []
    preprocessed_landmarks = []

    # Load images for which faces were found and corresponding landmarks
    for img, lm in zip(image_names, landmark_names):
        preprocessed_images.append(cv2.imread(old_dataset_image_path + img))
        preprocessed_landmarks.append(sio.loadmat(old_dataset_landmark_path + lm)['pts_2d'])

    '''
    Crop images and adjust landmarks
    '''

    for img, bboxes, lm, img_name, lm_name in zip(preprocessed_images, image_bboxes, preprocessed_landmarks,
                                                  image_names, landmark_names):

        #   Get detected face that overlaps the most with provided landmarks. If no face overlaps with more than 2/3
        #   of landmarks, the image is discarded and not considered suitable for training. This is an heuristic for
        #   cleaning the dataset

        num_faces_found, _ = bboxes.shape

        num_landmarks_inside_face = np.zeros(num_faces_found) # Will hold num landmarks inside each face in current img

        for i, face in enumerate(bboxes):
            for lm_point in lm:
                if lm_point[0] <= face[2] and lm_point[0] >= face[0] and lm_point[1] <= face[3] and lm_point[1] >= face[1]:
                    num_landmarks_inside_face[i] += 1

        if max(num_landmarks_inside_face) <= 2/3 * len(lm): # If no detected face covers more than 2/3 of landmarks
            print('\nWARNING: No suitable face bounding box was detected when processing image ' + img_name)
            continue

        chosen_face_number = np.argmax(num_landmarks_inside_face)

        bbox = bboxes[chosen_face_number,:-1]

        # Crop subimage from the given bbox
        cropped_img, final_bbox = extract_chip.extract_chip(img, bbox, scale_factor)

        # Get scaling factor for resizing
        initial_y_dim, initial_x_dim, _channels = cropped_img.shape
        scale_factor_x = final_image_size[0] / initial_x_dim
        scale_factor_y = final_image_size[1] / initial_y_dim

        # Resize image
        final_img = cv2.resize(cropped_img, (0, 0), fx=scale_factor_x, fy=scale_factor_y)

        # Adjust landmarks
        lm[:,0] -= final_bbox[0]
        lm[:,1] -= final_bbox[1]
        lm[:,0] *= scale_factor_x
        lm[:,1] *= scale_factor_y

        '''
        Save cropped face and adjusted landmark
        '''

        cv2.imwrite(new_dataset_path + img_name, final_img)
        sio.savemat(new_dataset_path + lm_name, dict([('pts_2d', lm)]))

def main():
    '''
    Testing for process_dataset
    :return: --
    '''
    os.chdir('..')
    curr_dir = os.getcwd()

    image_directories = ['testAFW/']#['HELEN/', 'IBUG/', 'LFPW/']

    for dir in image_directories:
        old_dataset_image_path = curr_dir + '/data/' + dir
        old_dataset_landmark_path = curr_dir + '/data/landmarks/' + 'AFW/'#dir
        new_dataset_path = curr_dir + '/processed_data/' + dir
        scale_factor = 2

        process_dataset(old_dataset_image_path, old_dataset_landmark_path, new_dataset_path, scale_factor)

if __name__ == '__main__':
    main()