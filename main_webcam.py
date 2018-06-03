import os, time, datetime, cProfile, pstats
import numpy as np
import cv2
import face_detection.detect_face as mtcnn
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
    minsize = 60  # Minimum window size to be detected in pixels
    threshold = [.6, .7, .7]  # Quality threshold for all cascade levels
    factor = .709  # Image pyramid scaling factor

    '''
    Initialize ESR regressor
    '''

    # Fac detection settings
    face_chip_size = (200, 200)
    bbox_scale = 1.5

    pwd = os.getcwd()
    savefile_path = pwd + '/processed_data/esr_savefile_2018-06-02_19_00_24.pickle'

    R = ESRegressor.ESRegressor()

    R.load_trained_regressor(savefile_path)

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

        is_ok, bgr_frame = webcam.read()

        '''
        Detect faces
        '''

        current_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)  # MTCNN reads RGB images
        current_frame = cv2.flip(current_frame, 1)  # For webcam mirroring

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
            bbox = faces[0, :-1]
            face_chip, final_bbox = extract_chip(current_frame, np.copy(bbox), bbox_scale)

            '''
            Detect landmarks for all detected faces
            '''

            # Get scaling that will be done to later determine landmark position in original frame
            scaling = face_chip_size[1] / (final_bbox[2] - final_bbox[0])

            # Resize to feed regressor
            face_chip = cv2.resize(face_chip, face_chip_size)

            # Regress landmarks
            regressed_landmarks = R.test([face_chip], 5)

            # Only 1 frame
            regressed_landmarks = regressed_landmarks[0]

            # Restore correct scale
            regressed_landmarks /= scaling

            # Restore correct offsets
            regressed_landmarks[:, 0] += final_bbox[0]
            regressed_landmarks[:, 1] += final_bbox[1]

            '''
            Draw landmarks and bboxes
            '''

            if isDrawingOn:
                for lm in regressed_landmarks:
                    current_frame = cv2.circle(current_frame, (int(lm[0]), int(lm[1])), landmarkRadius, landmarkColor, -1)

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

    # Get timestamp

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

    # Setup logging variables

    #log_name = os.getcwd() + '/esr_webcam_log_' + st + '.txt'

    # Run code with profiling

    #cProfile.run('main()', log_name)

    # Analyse profiling

    log_name = '/Users/miguelrita/Documents/Paper Implementations/esr/esr_webcam_log_2018-06-03_17:31:19.txt'

    p = pstats.Stats(log_name)
    p.strip_dirs().sort_stats('cumulative').print_callees('extract_shape_indexed_pixels')
    p.strip_dirs().sort_stats('cumulative').print_callees('test_stage_regressor')
    p.strip_dirs().sort_stats('cumulative').print_callees('compute_fern_bin')
    p.strip_dirs().sort_stats('cumulative').print_callees('test')