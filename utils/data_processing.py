import os, tqdm, glob
import numpy as np
import scipy.io as sio
import cv2
import face_detection.detect_face_batch as mtcnn
import utils.extract_chip as extract_chip

def process_dataset(old_dataset_image_path, old_dataset_landmark_path, new_dataset_path,
                    scale_factor, image_extension='jpg', num_of_batches = 1,
                    final_image_size=(200, 200), flip_images=True):
    '''
    Detects and crops faces in images provided in 'old_dataset_image_path', applying
    'scale_factor' to the bounding box found, and stores cropped images in
    'new_dataset_path'. Also corrects landmarks provided in 'old_dataset_landmark_path'
    and stores the new landmarks (relative to the new cropped images) in 'new_dataset_path'
    
    :param old_dataset_image_path: (str) Full path to dataset images
    :param old_dataset_landmark_path: (str) Full path to corresponding landmark files
    :param new_dataset_path: (str) Full path to store processed dataset images
    :param scale_factor: (float) Multiplies the square face bounding boxes found
    by the detector ie. if bounding box is 100 pixels wide and 'scale_factor' is 2 it becomes
    200 pixels wide. Fills cropped box with black pixels where scaled box goes out of image bounds
    :param image_extension: (str, optional) Image extension. Defaults to 'jpg'
    :param num_of_batches: (int) Split data into 'num_of_batches' and process one batch at the time
    :param final_image_size: (tuple of ints) Final desired image size in (x dim in pixels, y dim in pixels).
    :param flip_images: (bool) Also create flipped versions of the images and their respective landmarks
    :return: --
    '''

    '''
    Split data to process into batches if needed
    '''

    image_names_full_paths = glob.glob(old_dataset_image_path + '*.' + image_extension)
    images_names_no_path = [name.split('/')[-1] for name in image_names_full_paths]

    # Split all image names into batches (in this case split the indexes of those image names)

    all_img_nums = np.arange(0, len(image_names_full_paths))
    img_nums_per_batch = np.array_split(all_img_nums, num_of_batches)

    for batch_num, batch_img_indexes in enumerate(img_nums_per_batch):

        image_list = []
        batch_image_names_full_paths = [image_names_full_paths[k] for k in batch_img_indexes]
        batch_images_names_no_path = [images_names_no_path[k] for k in batch_img_indexes]

        for img_path in tqdm.tqdm(batch_image_names_full_paths,
                                  total=len(batch_image_names_full_paths),
                                  unit=' images',
                                  desc='Loading image batch ' + str(batch_num + 1) +
                                       ' of ' + str(len(img_nums_per_batch)) + ' for processing . . .'):
            image_list.append(cv2.imread(img_path))

        '''
        Load old images, get their detected faces (bboxes) and landmarks (pre-processing)
        '''

        # Detect face bounding boxes (bboxes) in preprocessed images
        (image_bboxes, image_names, image_indexes) = mtcnn.detect_face_batch(
            image_list,
            batch_images_names_no_path,
            are_images_rgb = False,
            minimum_window_size = 40,
            thresholds = [.6, .7, .7],
            pyramid_scaling_factor = .709
        )

        # Only keep images where faces were found
        preprocessed_images = [image_list[k] for k in image_indexes]

        # Only load landmarks for which there is a corresponding image with faces detected
        landmark_names =[img_name.split('.')[-2] + '_pts.mat' for img_name in image_names]
        preprocessed_landmarks = []

        for img, lm in zip(image_names, landmark_names):
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
            Save cropped face and adjusted landmark, and their flipped version if specified
            '''

            cv2.imwrite(new_dataset_path + img_name, final_img)
            sio.savemat(new_dataset_path + lm_name, dict([('pts_2d', lm)]))

            if flip_images:

                # Get flipped image and landmarks name
                flip_image_name = str.split(img_name, '.')[0] + '_f.' + str.split(img_name, '.')[-1]
                flip_lm_name = str.split(lm_name, 'pts.mat')[0] + 'f_pts.mat'

                # Flip image and landmarks
                flipped_final_img = cv2.flip(final_img, 1)
                flipped_lm = flip_68_landmarks(lm, final_image_size[0])

                cv2.imwrite(new_dataset_path + flip_image_name, flipped_final_img)
                sio.savemat(new_dataset_path + flip_lm_name, dict([('pts_2d', flipped_lm)]))

def flip_68_landmarks(original_landmarks, image_width):
    '''
    Flips a set of 68 landmarks, maintaining correct relative orders ie. mirrors positions but not numbering

    :param original_landmarks: (np array) 68-by-2 array of (x,y) orignal landmarks
    :param image_width: (int) Width of the image that the landmarks refer to
    :return: (np array) 68-by-2 array with the flipped landmarks
    '''

    landmark_mirrored_indexes = np.hstack((
        # Face contour and jawline
        np.array(
            range(16, -1, -1)
        ),
        # Brows
        np.array(
            range(26, 16, -1)
        ),
        # Nose bridge
        np.array(
            range(27, 31, 1)
        ),
        # Nose bottom
        np.array(
            range(35, 30, -1)
        ),
        # Left eye top
        np.array(
            range(45, 41, -1)
        ),
        # Left eye bottom
        np.array(
            range(47, 45, -1)
        ),
        # Right eye top
        np.array(
            range(39, 35, -1)
        ),
        # Right eye botton
        np.array(
            range(41, 39, -1)
        ),
        # Outer mouth (top of top lip and bottom of bottom lip)
        np.array(
            range(54, 47, -1)
        ),
        np.array(
            range(59, 54, -1)
        ),
        # Inner mouth (bottom of top lip and top of bottom lip)
        np.array(
            range(64, 59, -1)
        ),
        np.array(
            range(67, 64, -1)
        ),
    ))

    flipped_landmarks = np.copy(original_landmarks)

    # Flip landmark locations

    flipped_landmarks[:, 0] = image_width - flipped_landmarks[:, 0]

    # Restore landmark relative positioning and return

    return flipped_landmarks[landmark_mirrored_indexes, :]

def main():
    '''
    Debug code for process_dataset

    :return: --
    '''

    # Get source image directories

    os.chdir('..')
    curr_dir = os.getcwd()
    image_directories = ['LFPW/']

    for dir in image_directories:
        old_dataset_image_path = curr_dir + '/data/' + dir
        old_dataset_landmark_path = curr_dir + '/data/landmarks/' + dir
        new_dataset_path = curr_dir + '/processed_data/' + dir
        scale_factor = 1.5

        process_dataset(
            old_dataset_image_path,
            old_dataset_landmark_path,
            new_dataset_path,scale_factor,
            image_extension='jpg',
            num_of_batches = 4,
            final_image_size=(200, 200),
            flip_images=True
        )

if __name__ == '__main__':
    main()