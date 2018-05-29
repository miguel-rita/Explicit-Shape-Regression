import os, glob
import cv2
import face_detection.detect_face as mtcnn
from scipy.io import loadmat
import tensorflow as tf
import ESRegressor

def main():
    '''
    
    Run landmark detection on webcam feed using a pre-trained ESR regressor
    
    Press 'Q' to quit feed, 'S' to toggle landmarks and bounding box on/off
    
    :return: --
    '''

    '''
    Initialize face detector
    '''

    sess = tf.Session()

    # Create the 3 cascade networks
    pnet, rnet, onet = mtcnn.create_mtcnn(sess, None)

    # Set cascade parameters
    minsize = 40  # Minimum window size to be detected in pixels
    threshold = [.6, .7, .7]  # Quality threshold for all cascade levels
    factor = .709  # Image pyramid scaling factor

    '''
    Initialize ESR regressor
    '''

    pwd = os.getcwd()
    weights_path = pwd + '/processed_data/esr_weights_v3.pickle'
    os.chdir('./processed_data/AFW')

    # Load some landmarks (and images) to provide for initializing shapes
    landmark_names = glob.glob('*.mat')
    image_names = glob.glob('*.jpg')
    ground_truth_landmarks = []
    sample_images = []
    for lm_name in landmark_names:
        ground_truth_landmarks.append(loadmat(img_name + '_pts.mat')['pts_2d'])
    for img in sample_images[:2]:
        sample_images.append(cv2.imread(img))

    R = ESRegressor(sample_images, ground_truth_landmarks, 20, 20)

    R.load_weights(weights_path)

    '''
    Other initializations
    '''

    # Setup webcam
    webcam = cv2.VideoCapture(0)

    # Cosmetic parameters
    landmarkRadius = 2
    landmarkColor = (0, 255, 0)
    bbox_color = (0, 0, 255)
    bbox_thickness = 2

    # Flow control params
    visualizerRunning = True
    isDrawingOn = True

    '''
    Main Loop
    '''

    while visualizerRunning:

        ret, bgr_frame = webcam.read()

        '''
        Detect faces
        '''

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)  # MTCNN trained on RGB images

        # Run MTCNN detector
        # Output 'faces' is a n-by-m matrix where n is the num of faces detected and m=5 has bounding box info for each
        # face in the format xmin, ymin, xmax, ymax, confidence

        faces, _ = mtcnn.detect_face(rgb_frame, minsize, pnet, rnet, onet, threshold, factor)

        num_faces_found, _ = faces.shape

        if num_faces_found < 1:  # If no faces found append None placeholder
            continue

        '''
        Detect landmarks for all detected faces
        '''

        # Grab some images for testing
        test_images = []
        test_landmarks = []

        for img_name in small_pose_image_names[-100:]:
            # Load image
            test_images.append(cv2.imread(img_name + imageExtension))
            # Load landmarks
            test_landmarks.append(loadmat(img_name + '_pts.mat')['pts_2d'])

            # regressed_landmarks = R.test(test_images, 15)

        '''
        Draw landmarks and bboxes
        '''

        if isDrawingOn:
            if not landmarks is None:
                for lm in landmarks[currentImageNum]:
                    currentImage = cv2.circle(currentImage, (int(lm[0]), int(lm[1])), landmarkRadius, landmarkColor1,
                                              -1)
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
        if key == ord('s'):
            isDrawingOn = not isDrawingOn
        if key == ord('q'):  # Quit visualization
            print('Quitting visualization . . .')
            visualizerRunning = False


if __name__ == '__main__':
    main()