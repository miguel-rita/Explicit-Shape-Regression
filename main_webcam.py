import os, glob
import cv2
import face_detection.detect_face as mtcnn
from scipy.io import loadmat
import tensorflow as tf
import ESRegressor
from utils.extract_chip import extract_chip

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
    minsize = 90  # Minimum window size to be detected in pixels
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
        ground_truth_landmarks.append(loadmat(lm_name)['pts_2d'])
    for img in image_names[:50]:
        sample_images.append(cv2.imread(img))

    R = ESRegressor.ESRegressor(sample_images, ground_truth_landmarks, 20, 20)

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

        current_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)  # MTCNN reads RGB images

        # Run MTCNN detector
        # Output 'faces' is a n-by-m matrix where n is the num of faces detected and m=5 has bounding box info for each
        # face in the format xmin, ymin, xmax, ymax, confidence

        faces, _ = mtcnn.detect_face(current_frame, minsize, pnet, rnet, onet, threshold, factor)

        # Put frame back to BGR

        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)

        num_faces_found, _ = faces.shape

        if num_faces_found > 0:

            '''
            Extract face chip
            '''
            scale = 2.0
            face_chip, final_bbox = extract_chip(current_frame, faces[0, :-1], scale)

            '''
            Detect landmarks for all detected faces
            '''

            face_chip = cv2.resize(face_chip, (450, 450))
            regressed_landmarks = R.test([face_chip], 20)

            '''
            Draw landmarks and bboxes
            '''

            if isDrawingOn:
                for lm in regressed_landmarks[0]:
                    face_chip = cv2.circle(face_chip, (int(lm[0]), int(lm[1])), landmarkRadius, landmarkColor, -1)

                face_chip = cv2.flip(face_chip, 1) # For webcam mirroring

                current_frame = face_chip

        else:
            pass

        cv2.imshow("ESR v1", current_frame)

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