import tqdm, glob
import cv2
import face_detection.detect_face as mtcnn
import tensorflow as tf

def detect_face_batch(image_list, image_names, are_images_rgb = True, minimum_window_size = 40, thresholds = [.6, .7, .7],
                      pyramid_scaling_factor = .709):
    '''
    Detects faces in a batch of images, using a MTCNN detector, see
    'https://arxiv.org/abs/1604.02878'
    Implementation by David Sandberg, see
    'https://github.com/davidsandberg/facenet'

    :param image_list: (list) List of images in numpy array format
    :param image_names: (list) List of image names, without the path
    :param are_images_rgb: (bool) Defaults to 'True'. Specify 'False' if provided images are BGR
    :param minimum_window_size: (int) Minimum face size to be searched in the pictures, in pixels
    :param thresholds: (list) List of 3 floats representing discarding confidence thresholds at each stage in cascade
    :param pyramid_scaling_factor: (float) Float representing scaling factor when building image pyramid
    :return: (tuple(list, list)) Tuple of two lists: 1st list - List of detected faces, in numpy array format.
    Each element of the list is a n-by-m matrix, where n is the number of faces detected
    and m=5 as follows [xmin ymin xmax ymax confidence]. 2nd list - Corresponding image names
    '''

    '''
    Initialize face detector
    '''

    sess = tf.Session()

    # Create the 3 cascade networks
    pnet, rnet, onet = mtcnn.create_mtcnn(sess, None)

    # Set cascade parameters
    minsize = minimum_window_size # Minimum window size to be detected in pixels
    threshold = thresholds  # Quality threshold for all cascade levels
    factor = pyramid_scaling_factor # Image pyramid scaling factor

    '''
    Detect faces
    '''

    faces_list = []
    names_list = []
    number_of_images = len(image_list)

    print('STATUS: Detecting faces ...')
    for img, img_name in tqdm.tqdm(zip(image_list, image_names), total=number_of_images, unit=' images'):

        if not are_images_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # MTCNN trained on RGB images

        # Run MTCNN detector
        # Output 'faces' is a n-by-m matrix where n is the num of faces detected and m=5 has bounding box info for each
        # face in the format xmin, ymin, xmax, ymax, confidence

        faces, _ = mtcnn.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

        num_faces_found, _ = faces.shape

        if num_faces_found < 1: # If no faces found append None placeholder
            print('\nWARNING: No faces found in file ' + img_name)
            continue

        faces_list.append(faces)
        names_list.append(img_name)

    sess.close()

    return (faces_list, names_list)

def detect_face_batch_from_directory(image_directory, image_extension = '.jpg', minimum_window_size = 40, thresholds = [.6, .7, .7],
                      pyramid_scaling_factor = .709):
    '''
    Detects faces in a batch of images in a directory, using a MTCNN detector, see
    'https://arxiv.org/abs/1604.02878'
    Implementation by David Sandberg, see
    'https://github.com/davidsandberg/facenet'

    :param image_directory: (str) Path to image files
    :param image_extension: (str) Image extension. Defaults to .jpg
    :param minimum_window_size: (int) Minimum face size to be searched in the pictures, in pixels
    :param thresholds: (list) List of 3 floats representing discarding confidence thresholds at each stage in cascade
    :param pyramid_scaling_factor: (float) Float representing scaling factor when building image pyramid
    :return: (tuple(list, list)) Tuple of two lists: 1st list - List of detected faces, in numpy array format.
    Each element of the list is a n-by-m matrix, where n is the number of faces detected
    and m=5 as follows [xmin ymin xmax ymax confidence]. 2nd list - Corresponding image names
    '''

    image_names_full_paths = glob.glob(image_directory + '/*.' + image_extension)
    images_names_no_path = [name.split('/')[-1] for name in image_names_full_paths]

    image_list = [cv2.imread(img_path) for img_path in image_names_full_paths]

    return detect_face_batch(image_list, images_names_no_path, are_images_rgb=False,
                             minimum_window_size=minimum_window_size, thresholds=thresholds,
                             pyramid_scaling_factor = pyramid_scaling_factor)
