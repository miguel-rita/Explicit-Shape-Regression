import os, random, glob, tqdm, pickle, time, datetime
import numpy as np
import cv2 as cv2
from scipy.linalg import orthogonal_procrustes
from scipy.io import loadmat

import cProfile, pstats
from utils.visualizer import visualizer

class ESRegressor:
    '''
    Explicit Shape Regression implementation, according to the paper by 
    Xudong Cao, Yichen Wei, Fang Wen, Jian Sun:
    'https://pdfs.semanticscholar.org/947a/34cd560a25996804339420483c80de0c3740.pdf'
    
    Class usage is as follows:
    1st - Instantiate class
    2nd - Train using 'train' method
    3rd - Infer using 'test' method
    '''

    def __init__(self):
        '''
        Class constructor. Most initializations are done through the 'train' method
        '''

        self.stage_regressors = []

    def train(self, training_images, ground_truths, num_augmentation, max_num_test_shapes,
              num_features = 400, local_random_displacement = 25, num_stages = 10, num_ferns = 500,
              num_fern_levels = 5, save_weights = None):
        '''
        Train the ESRegressor, augmenting the training set and converting all images to BW
        
        :param training_images: (list) BGR training images as numpy arrays
        :param ground_truths: (list) Landmark (x,y) global coordinates as num-by-2 numpy arrays,
        where 'num' is the number of landmarks
        :param num_augmentation: (int) Number of times to augment training dataset, ie. for each
        ground truth shape and image we'll have 'num_augmentation' different starting positions for
        landmark offset computations
        :param max_num_test_shapes:(int) Max number of predefined starting shapes for bagging results at test time
        :param num_features: (int) Number of features to sample per image
        :param local_random_displacement: (int) Range of uniform random noise when determining feature points
        in local landmark coordinates, in pixels
        :param num_stages: (int) Number of boosted stages
        :param num_ferns: (int) Number of ferns per stage
        :param num_fern_levels: (int) Number of levels per fern
        :param save_weights: (str) If specified, will save training results to the specified path
        :return: --
        '''

        # 1. Initialize and augment training set

        self.training_set = []  # Will contain training lists in the format [image, ground_truth, start_pos]
        self.test_start_shapes = []  # Will contain fixed test start shapes for bagging

        for i, (img, ground_truth) in tqdm.tqdm(enumerate(zip(training_images, ground_truths)),
                                                total=len(training_images),
                                                unit=' images',
                                                desc='Loading train images into regressor. . .'):

            # Convert img to BW
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Get 'num_augmentation' indexes without replacement
            unique_starting_positions = random.sample(range(0, len(ground_truths)), num_augmentation)

            for start_pos in unique_starting_positions:
                self.training_set.append([img, ground_truth, np.copy(ground_truths[start_pos])])

        # 2. Initialize mean shape
        self.mean_shape = np.mean(ground_truths, axis=0)

        # 3. Initialize predefined testing starting shapes
        self.test_start_shapes = self.initialize_test_shape(max_num_test_shapes, random_translation=20,
                              random_rotation=45, random_scaling=0.25)

        # 4. Other initializations

        self.number_of_landmarks = len(ground_truths[0])
        self.num_features = num_features

        for stage in tqdm.tqdm(range(0, num_stages), total=num_stages, unit='stage',
                               desc='Overall training progress', position=0):

            Y_normalized_targets = [] # Will contain normalized targets for this stage
            M_inverse_transforms = [] # Will contain normal transformations for this stage

            # Progress tracking variables
            self.progress_current_stage_num_being_trained = stage + 1
            self.total_num_stages = num_stages

            # I. Compute normalized targets

            for img, ground_truth, start_pos in self.training_set:

                # 1. Get the orthogonal transform to the mean shape (considering scaling was already applied)
                normalizing_transform, _ = orthogonal_procrustes(start_pos, self.mean_shape)

                # 2. Get the regression target and normalize to 'mean shape space'
                Y_normalized_targets.append( np.matmul((ground_truth - start_pos), normalizing_transform) )

                # 3. Store inverse transforms M^-1
                M_inverse_transforms.append(np.linalg.inv(normalizing_transform))

            # Flatten normalized targets - from a list of targets of len = number of training samples, where each
            # target is a num_landmarks-by-2 matrix, to a n-by-m matrix, where n = number of training samples and
            # m = 2 * num_landmarks, ie. [delta_x0, delta_y0, delta_x1, delta_y1, ...]
            Y_normalized_targets = np.vstack([target.flatten() for target in Y_normalized_targets])

            ynt = np.array(Y_normalized_targets)
            ynt_posavg = np.average(ynt[ynt>0])
            ynt_avg = np.average(ynt)
            ynt_var = np.var(ynt)

            print("\n\n***************************************")
            print("Regression target info for stage", str(stage),":")
            print("Average:",ynt_avg)
            print("Positive average:",ynt_posavg)
            print("Std dev:",np.sqrt(ynt_var))
            print("***************************************\n")

            # II. Train and store current stage regressor

            # 1. First generate local coordinates ...
            local_coords = self.generate_local_coordinates(local_random_displacement)

            # 2. ... then get feature locations and pixel intensities in original image, global coordinates ...

            # Reminder : training_set stores a list for each training sample, with list[0] being the image, [1] being
            # the ground truth and [2] the starting position at this stage

            pixel_feats = self.extract_shape_indexed_pixels([t[0] for t in self.training_set],
                                                            [t[2] for t in self.training_set],
                                                            local_coords, M_inverse_transforms)

            # 3. Finally train the stage regressor. Note 'pixel_feats' were computed outside 'train_stage_regressor'
            # function to be reused below (see step III.)
            current_stage_regressor = self.train_stage_regressor(Y_normalized_targets, num_ferns,
                                                                 num_fern_levels, pixel_feats)

            self.stage_regressors.append((current_stage_regressor, local_coords))

            # III. Update starting shapes in 'self.training_set' for next stage
            stage_regressor_output = self.test_stage_regressor(current_stage_regressor, local_coords,
                                                               pixel_feats, [t[0] for t in self.training_set],
                                                               [t[2] for t in self.training_set])

            csr = np.array(stage_regressor_output)

            for i in range(len(self.training_set)):
                self.training_set[i][2] += np.matmul(stage_regressor_output[i], M_inverse_transforms[i])

        if save_weights is not None:

            # Pack dictionary to pickle
            save_dict = {
                'weights' : self.stage_regressors,
                'mean_shape' : self.mean_shape,
                'test_shapes' : self.test_start_shapes,
                'num_features' : self.num_features,
                'num_landmarks' : self.number_of_landmarks,
            }

            with open(save_weights, 'wb') as h:
                pickle.dump(save_dict, h, protocol=pickle.HIGHEST_PROTOCOL)

    def test(self, images, bagging_size):
        '''
        Regressor testing function - will output regressed shapes for all the specified images
        
        :param images: (list) List of images to regress landmarks
        :param bagging_size: (int) Number of times to run the regressor per image, after which
        the median result is picked
        :return: (list) List of regressed landmarks, each n-by-2 numpy array where n = number of landmarks
        '''

        if bagging_size > len(self.test_start_shapes):
            raise ValueError('Not enough starting shapes initialized for selected bagging size')

        # Starting images will be converted to BW and will each have 'bagging_size' starting shapes
        # np.copy to avoid having multiple equal starting shapes (but associated with different images) pointing
        # to the same memory address

        initialized_images = []

        for img in images:

            _h, _w, num_channels = img.shape

            # If color image must convert to grayscale

            if num_channels == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            for i in range(bagging_size):
                initialized_images.append([img, np.copy(self.test_start_shapes[i])])

        # Regress shapes for this stage
        for stage_regressor in self.stage_regressors:

            # Get inverse transforms for this stage

            M_inverse_transforms = []

            for _img, shape in initialized_images:
                scaling_factor = 1.0
                normalizing_transform, _ = orthogonal_procrustes(shape * scaling_factor, self.mean_shape)
                normalizing_transform *= scaling_factor
                M_inverse_transforms.append(np.linalg.inv(normalizing_transform))

            local_coords = stage_regressor[1]

            pixel_feats = self.extract_shape_indexed_pixels([t[0] for t in initialized_images],
                                                            [t[1] for t in initialized_images],
                                                            local_coords, M_inverse_transforms)

            # Get stage regressions in 'mean shape space'
            stage_output = self.test_stage_regressor(stage_regressor[0], stage_regressor[1], pixel_feats,
                                                     [t[0] for t in initialized_images],
                                                     [t[1] for t in initialized_images])

            for i in range(len(initialized_images)):
                # Update shapes to original spaces, for next stage
                initialized_images[i][1] += np.matmul(stage_output[i], M_inverse_transforms[i])

        # Bag results using median landmark values as in the paper and return them
        final_shapes = []
        for i in range(len(images)):
            final_shapes_for_this_image = [t[1] for t in initialized_images[i*bagging_size:(i+1)*bagging_size]]
            median_shape = np.median(final_shapes_for_this_image, axis=0) # Axis 0 to get median across shapes only
            final_shapes.append(median_shape)

        return final_shapes

    def initialize_test_shape(self, max_num_test_shapes, random_translation=20,
                              random_rotation=45, random_scaling=0.25):
        '''
        Initializes in memory 'max_num_test_shapes' for use at test time. Note that
        at test time any number of test shapes can be used, from one up to 'max_num_test_shapes'.
        The initialized shapes are obtaining by applying random translations, rotations and
        scaling to 'self.mean_shape'

        :param max_num_test_shapes: (int) Maximum number of predefined test shapes available in memory. Maximum number
        of test shapes that can be used at test time
        :param random_translation: (int) Random vertical and horizontal translation to apply to mean shape, in pixels
        :param random_rotation: (float) Random rotation to apply to mean shape, in degrees ie. up to 'random_rotation'
        degree rotation counter- or clock-wise around mass center
        :param random_scaling: (float) Random scaling to apply to mean shape
        :return: (list) List of random test shapes
        '''

        test_shapes = []

        for i in range(max_num_test_shapes):

            # Shape to be randomly transformed
            current_test_shape = np.copy(self.mean_shape)

            # Center around mean
            mean_coords = np.mean(current_test_shape, axis=0)
            current_test_shape -= mean_coords

            # Apply random scaling, rotation and translation

            # Scaling
            current_test_shape *= np.random.uniform(1-random_scaling, 1+random_scaling)

            # Rotation
            theta_rot = np.deg2rad(np.random.uniform(-random_rotation, random_rotation))
            cos, sin = np.cos(theta_rot), np.sin(theta_rot)
            rot_matrix = np.array([
                [cos, -sin],
                [sin, cos]
            ])
            current_test_shape = np.matmul(current_test_shape, rot_matrix)

            # Translation
            xy_translation = np.array([
                np.random.uniform(-random_translation, random_translation),
                np.random.uniform(-random_translation, random_translation)
                ])
            current_test_shape -= xy_translation

            # Add back mean values to re-center shape
            current_test_shape += mean_coords

            test_shapes.append(current_test_shape)

        return test_shapes

    def load_trained_regressor(self, savefile_path):
        '''
        Loads and sets up a previously trained regressor from a savefile
        
        :param savefile_path: (str) Path to regressor savefile
        :return: --
        '''

        # Load saved data, including stage regressors. Remember each regressor
        # is a pair (regressor parameters, associated local coords)

        with open(savefile_path, 'rb') as h:
            saved_dict = pickle.load(h)
            self.stage_regressors = saved_dict['weights']
            self.mean_shape = saved_dict['mean_shape']
            self.test_start_shapes = saved_dict['test_shapes']
            self.num_features = saved_dict['num_features']
            #self.number_of_landmarks = saved_dict['num_landmarks']


    def train_stage_regressor(self, Y_normalized_targets, num_ferns, num_fern_levels,
                              pixel_features, threshold_multiplier = 0.2, beta = 5):
        '''
        Train a stage regressor, comprised of 'num_ferns' ferns, each with 'num_fern_levels' decision levels
        
        :param Y_normalized_targets:
        :param num_ferns: 
        :param num_fern_levels: 
        :param pixel_features: 
        :param threshold_multiplier:
        :param beta:
        :return: TODO
        '''

        # I. Initial setup

        # 1. Compute pixel-pixel covariance matrix
        pix2pix_cov_matrix = np.cov(pixel_features, rowvar=False)

        # 2. Initialize current regressed displacements as the original target displacements
        Y_current = Y_normalized_targets

        # II. Start training the ferns

        stage_ferns = []

        for fern_number in tqdm.tqdm(range(num_ferns), total=num_ferns, unit='fern',
                                     desc='Training stage '+
                                             str(self.progress_current_stage_num_being_trained)+
                                             ' of '+str(self.total_num_stages),
                                     position=1): # For each weak regressor ...

            # ... select the best pixel-difference features for this fern ...
            selected_pixel_diff_features, diff_indexes = self.correlation_feature_selection(Y_current,
                                                                                            pixel_features,
                                                                                            pix2pix_cov_matrix,
                                                                                            num_fern_levels)

            # ... compute the random threshold limit based on pixel-difference feature range ...
            min_pixel_diff_value = np.min(selected_pixel_diff_features)
            max_pixel_diff_value = np.max(selected_pixel_diff_features)

            # ... sample threshold for each fern level from uniform distribution ...
            thresholds = np.random.uniform(threshold_multiplier * min_pixel_diff_value,
                                           threshold_multiplier * max_pixel_diff_value,
                                           num_fern_levels)

            # ... partition training samples into bins ...
            bin_index_per_training_sample = self.compute_fern_bin(selected_pixel_diff_features, thresholds)

            bins = [[] for i in range(2**num_fern_levels)] # Empty bins, filled below

            for i, training_sample_target in enumerate(Y_current):
                bins[bin_index_per_training_sample[i]].append(training_sample_target)

            # ... compute bin outputs ...
            bin_outputs = []
            for basket in bins:
                if len(basket) == 0: # If bin contains no training sample, bin output will be zero
                    bin_outputs.append(np.zeros(Y_current[0, :].shape))
                else:
                    bin_outputs.append(np.average(basket, axis=0) * 1/(1 + beta/len(basket)))

            # ... and build the fern
            stage_ferns.append((thresholds, diff_indexes, bin_outputs))

            # Update regression targets for next weak regressor
            for training_sample_number, bin_index in enumerate(bin_index_per_training_sample):
                Y_current[training_sample_number, :] -= bin_outputs[bin_index]

        # 2. Return the complete stage regressor

        return stage_ferns

    def test_stage_regressor(self, stage_ferns, local_coords, pixel_features, images, current_shapes):
        '''
        Apply the supplied regressor stage to all images and corresponding shapes provided in 'images' and
        'current_shapes'
        
        :param stage_ferns: 
        :param local_coords: 
        :param pixel_features:
        :param images: 
        :param current_shapes: 
        :return: TODO
        '''

        # Compute normalized transform for provided images
        normalized_transforms = []

        for img, curr_pos in zip(images, current_shapes):

            scaling_factor = 1.0

            # Get the orthogonal transform to the mean shape (considering scaling was already applied)
            normalized_transform, _ = orthogonal_procrustes(curr_pos * scaling_factor, self.mean_shape)

            # Add scale to orthogonal transform matrix and append to list
            normalized_transform *= scaling_factor
            normalized_transforms.append(normalized_transform)

        delta_shapes = [shape * 0 for shape in current_shapes] # Placeholder for the regressions for each shape

        for (thresholds, feat_differences, bin_outputs) in stage_ferns:

            # Reminder: a fern is a (a,b,c) tuple where a=thresholds, b=feature difference indexes, c=bin outputs

            # 1. Get selected feature differences for this fern

            selected_feats = np.zeros((len(images), len(thresholds))) # Placeholder for a n-by-m matrix, where n is the
            # number of training images and m is the number of levels in a fern (here extracted by number of thresholds)

            for iii, (i,j) in enumerate(feat_differences):
                selected_feats[:, iii] = pixel_features[:, i] - pixel_features[:, j]

            # 2. Get bin numbers for all images

            bin_numbers = self.compute_fern_bin(selected_feats, thresholds)

            # 3. Get and reshape fern output and add to current offsets

            fern_outputs = [bin_outputs[bin_number] for bin_number in bin_numbers]
            final_deltas = [np.reshape(fern_output, (self.number_of_landmarks, 2)) for fern_output in fern_outputs]

            for i, final_delta in enumerate(final_deltas):
                delta_shapes[i] += final_delta

        return delta_shapes



    def correlation_feature_selection(self, Y_regression_target, pixel_features,
                                      pix2pix_cov_matrix, num_fern_levels):
        '''
        Select 'num_fern_levels' pixel difference features with highest correlation to displacement projections
        as detailed in the paper
        
        :param Y_regression_target: TODO
        :param pixel_features:
        :param pix2pix_cov_matrix: 
        :param num_fern_levels: 
        :return: (tuple of 2 lists) First list contains selected pixel difference feature values, second list
        contains corresponding indexes
        '''

        pixel_difference_features = np.zeros((len(Y_regression_target), num_fern_levels)) # Placeholder, filled below
        pixel_difference_indexes = []

        for fern_level in range(num_fern_levels):

            # Random projection from unit gaussian, normalized
            gaussian_proj = np.random.normal(0, 1, size=(self.number_of_landmarks * 2, 1))
            gaussian_proj /= np.linalg.norm(gaussian_proj)

            # Projected target for this fern level
            Y_proj = np.squeeze(np.matmul(Y_regression_target, gaussian_proj))

            target_pix_cov = np.zeros(self.num_features) # Placeholder, filled below

            # Compute target-pixel covariance
            for i in range(self.num_features):
                target_pix_cov[i] = np.cov(Y_proj, pixel_features[:, i])[0, 1] # [0, 1] (or [1, 0]) to fetch the
                # covariance term from the covariance matrix

            # Compute project target variance
            Y_var = np.var(Y_proj)

            # To be filled below
            corr_matrix = np.ones((self.num_features, self.num_features)) * -1

            for i in range(self.num_features):
                for j in range(self.num_features):

                    denominator = np.sqrt(
                        Y_var * (pix2pix_cov_matrix[i,i] + pix2pix_cov_matrix[j,j] - 2 * pix2pix_cov_matrix[i,j]) )

                    if denominator == 0:
                        corr_matrix[i, j] = -1 # Not a pixel difference, trying to corr pixel with itself, which is
                        # always the case for i==j but can happen for different combinations due to random shape indexed
                        # feature sampling ie. randomly sample the same pixel for the same landmark twice
                    else:
                        corr = (target_pix_cov[i] - target_pix_cov[j]) / denominator
                        corr_matrix[i, j] = corr

            # Get max correlation pixel pair
            i_max, j_max = np.unravel_index(np.argmax(corr_matrix), corr_matrix.shape)

            pixel_difference_features[:, fern_level] =  pixel_features[:,i_max] - pixel_features[:, j_max]
            pixel_difference_indexes.append((i_max, j_max))

        return (pixel_difference_features, pixel_difference_indexes)

    def compute_fern_bin(self, selected_feature_differences, thresholds):
        '''
        Computes a bin number for each training sample in 'selected_feature_differences'
        
        :param selected_feature_differences: 
        :param thresholds: 
        :return: (int) Bin number, from 0 to 2^number of fern levels
        '''

        num_levels = len(thresholds)

        # Get fern level results in binary
        binary_checkpoint_results = selected_feature_differences\
                                    >= np.tile(thresholds, (len(selected_feature_differences), 1))

        # Convert to base ten
        final_bin_numbers = selected_feature_differences[:,0] * 0 # Initialize vector that'll contain final bin numbers
        for level in range(0, num_levels):
            final_bin_numbers += binary_checkpoint_results[:,level] * 2**(num_levels - level - 1) # Base 2 to 10

        return final_bin_numbers.astype('int16')

    def extract_shape_indexed_pixels(self, images, current_shapes, local_coords, inverse_transforms):
        '''
        Extract self.num_features pixel intensities for each image in the original image-shape pairs provided,
        in global coordinates
        
        :param images: (list of np arrays) Images from which to extract pixel intensities (features)
        :param current_shapes: (list of np arrays) Current shapes associated with 'images' 
        :param local_coords: (np array) N-by-m array where N = number of features to extract per image, and m = 3,
        with format [num_associated_landmark, x_offset, y_offset]
        :param inverse_transforms: (list of np arrays) Inverse transforms back to original 'current_shapes' space
        the local coordinates indexed by the numbers in 'selected_feature_numbers'. See 'return' below
        :return: A n-by-m numpy array, where n = number of images (len(images)), and m = number of features
        self.num_features, with contents being pixel intensities.
        '''

        pixel_feats = np.zeros((len(images), self.num_features))  # Placeholder for pixel intensities
        pixel_feats_y_dim, pixel_feats_x_dim = pixel_feats.shape
        global_coordinates = np.zeros(
            (2 * pixel_feats_x_dim, 2 * pixel_feats_y_dim))  # Place holder for global coordinates

        for i in range(pixel_feats_y_dim):
            for j in range(pixel_feats_x_dim):

                # Get landmark coords in start_pos for the sample being requested (sample i) for the landmark being
                # requested (must check which landmark is associated with the feature being requested, feature j).

                landmark_coords_in_start_pos = current_shapes[i][local_coords[j, 0], :]

                # Transform the local coords back to 'start_pos space'

                local_coords_transformed_back_to_start_pos = np.matmul(inverse_transforms[i],
                                                                       local_coords[j, 1:])
                global_coords = local_coords_transformed_back_to_start_pos + landmark_coords_in_start_pos
                global_coordinates[j, 2 * i:2 * i + 2] = global_coords

                # Ensure coordinates do not go outside image

                y = int(global_coords[1])
                x = int(global_coords[0])

                img_height, img_width = images[i].shape

                if y >= img_height - 1:
                    y = np.clip(y, 0, img_height - 1)
                if x >= img_width - 1:
                    x = np.clip(x, 0, img_width - 1)

                # Grab pixels from the start_pos image
                pixel_feats[i, j] = images[i][y, x]

        return pixel_feats

    def generate_local_coordinates(self, local_random_displacement):
        '''
        Generate random local coordinate offsets relative to random landmarks
        
        :param local_random_displacement: (int) Limit to the range of random offset, in pixels, that will be used
        to produce random local coordinates
        :return: (np array) A n-by-3 array where n is the number of features and each line is as follows:
        [index of landmark selected as basis for local coord, random x offset to apply, random y offset to apply]
        '''
        displacements = np.random.randint(-local_random_displacement, local_random_displacement,
                                                 size=(self.num_features, 2)) # Offsets to add to each landmark location

        landmark_indexes = np.random.randint(self.number_of_landmarks, size=(self.num_features, 1)) # Random landmark
        # locations

        return np.hstack((landmark_indexes, displacements)) # Combine both

def main():
    '''
    Debugging function for class ESRegressor
    
    :return: --
    '''

    # Load processed images and landmarks

    dataset_names = ['LFPW/']#['AFW/', 'HELEN/', 'IBUG/', 'LFPW/']
    imageExtension = '.jpg'

    image_names = []

    for dataset_name in dataset_names:

        # Switch to dataset dir

        temp_image_names = glob.glob('processed_data/' + dataset_name +'*' + imageExtension)

        # Get filenames without extension

        image_names.extend([n.split('.')[0] for n in temp_image_names])

    images = []
    landmarks = []

    for img_name in tqdm.tqdm(image_names[:5000], total=len(image_names), unit=' images', desc='Loading image files. . .'):

        # Load image

        images.append(cv2.imread(img_name + imageExtension))

        # Load landmarks

        landmarks.append(loadmat(img_name + '_pts.mat')['pts_2d'])

    # Get timestamp

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

    # Setup savefile settings

    pwd = os.getcwd()
    weights_path = pwd + '/processed_data/esr_savefile_'+st+'.pickle'

    # Instantiate regressor

    R = ESRegressor()

    # Train regressor

    print("Save path:", weights_path, "\nTrain set num. of images:", len(images), '\n')
    R.train(
        training_images = images,
        ground_truths = landmarks,
        num_augmentation = 20,
        max_num_test_shapes = 20,
        num_features = 200,
        local_random_displacement = 20,
        num_stages = 7,
        num_ferns = 250,
        num_fern_levels = 5,
        save_weights = weights_path,
    )

    #R.load_trained_regressor('/home/lanfear/Explicit-Shape-Regression/processed_data/esr_savefile_2018-05-31_17:39:03.pickle')

    # Grab some images for testing

    test_images = []
    test_landmarks = []

    for img_name in image_names[40:60]:

        # Load image

        test_images.append(cv2.imread(img_name + imageExtension))

        # Load landmarks

        #test_landmarks.append(loadmat(img_name+'_pts.mat')['pts_2d'])

    regressed_landmarks = R.test(test_images, 5)

    visualizer(images=test_images, landmarks=regressed_landmarks)

if __name__ == '__main__':

    # Get timestamp

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

    # Setup logging variables

    log_name = os.getcwd() + '/esr_log_'+ st +'.txt'

    # Run code with profiling

    cProfile.run('main()', log_name)

    # Analyse profiling

    p = pstats.Stats(log_name)
    p.strip_dirs().sort_stats('cumulative').print_stats()
