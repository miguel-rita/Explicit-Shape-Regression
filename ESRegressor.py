import os, random, glob, tqdm, pickle
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

    def __init__(self, training_images, ground_truths, num_augmentation, num_test_shapes):
        '''
        Constructor, will augment training set by default and convert all images to BW
        
        :param training_images: (list) BGR training images as numpy arrays
        :param ground_truths: (list) Landmark (x,y) global coordinates as num-by-2 numpy arrays,
        where 'num' is the number of landmarks
        :param num_augmentation: (int) Number of times to augment training dataset, ie. for each
        ground truth shape and image we'll have 'num_augmentation' different starting positions for
        landmark offset computations
        :param num_test_shapes:(int) Max number of predefined starting shapes for bagging results at test time
        '''

        # 1. Initialize the training set and testing predefined shapes

        self.training_set = [] # Will contain training lists in the format [image, ground_truth, start_pos]
        self.test_start_shapes = [] # Will contain fixed test start shapes for bagging

        for i, (img, ground_truth) in enumerate(zip(training_images, ground_truths)):

            # Convert img to BW
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Get 'num_augmentation' indexes without replacement
            unique_starting_positions = random.sample(range(0,len(ground_truths)), num_augmentation)

            for start_pos in unique_starting_positions:
                self.training_set.append([img, ground_truth, np.copy(ground_truths[start_pos])])

        testing_start_positions = random.sample(range(0, len(ground_truths)), num_test_shapes)

        for pos in testing_start_positions:
            self.test_start_shapes.append(np.copy(ground_truths[pos]))

        # 2. Initialize mean shape

        self.mean_shape = np.mean(ground_truths, axis=0)

        # 3. Other initializations

        self.number_of_landmarks = len(ground_truths[0])
        self.stage_regressors = []

    def train(self, num_features = 400, local_random_displacement = 25, num_stages = 10,
              num_ferns = 500, num_fern_levels = 5, save_weights = None):
        '''
        Train the ESRegressor
        
        :param num_features: (int) Number of features to sample per image
        :param local_random_displacement: (int) Range of uniform random noise when determining feature points
        in local landmark coordinates, in pixels
        :param num_stages: (int) Number of boosted stages
        :param num_ferns: (int) Number of ferns per stage
        :param num_fern_levels: (int) Number of levels per fern
        :param save_weights: (str) If specified, will save training results to the specified path
        :return: --
        '''

        # Property assignments
        self.num_features = num_features

        for stage in tqdm.tqdm(range(0, num_stages), total=num_stages, unit='stage',
                               desc='Overall training progress', position=0): # For each stage in the regressor, we must ...

            Y_normalized_targets = [] # Will contain normalized targets for this stage
            M_inverse_transforms = [] # Will contain normal transformations for this stage

            # Progress tracking variables
            self.progress_current_stage_num_being_trained = stage + 1
            self.total_num_stages = num_stages

            tset_shapes = np.array([t[2] for t in self.training_set])
            shapes_armax = np.unravel_index(np.argmax(tset_shapes), tset_shapes.shape)
            shapes_armin = np.unravel_index(np.argmin(tset_shapes), tset_shapes.shape)

            # I. Compute normalized targets

            for img, ground_truth, start_pos in self.training_set: # ... compute the regression targets, for
                # each element in training set

                # Unused code below \/

                # 1. Normalize start_pos scale to match mean_shape scale
                # Check https://en.wikipedia.org/wiki/Procrustes_analysis for details
                # We're assuming start_pos and mean_shape have same average position already. Simplifying assumption
                # to remove translation adjustment also done in the paper.

                # Calculate scaling factor (to go from start_pos scale to mean_shape scale)
                # scaling_factor = np.sum(self.mean_shape**2) / np.sum(start_pos**2)

                # End of unused code /\

                # NOTE: Although scaling is used in the paper, for training datasets with large pose variation
                # the assumption of mean landmark position at the origin is not true, which would greatly distort
                # procrustes scaling. Hence we chose to complicate the model by sacrificing scale invariance

                scaling_factor = 1.0

                # 2. Get the orthogonal transform to the mean shape (considering scaling was already applied)
                normalizing_transform, _ = orthogonal_procrustes(start_pos * scaling_factor, self.mean_shape)

                # 2.5. Add scale to orthogonal transform matrix. These are the 'M' matrices in the paper
                normalizing_transform *= scaling_factor

                # 3. Get the regression target and normalize to 'mean shape space'
                Y_normalized_targets.append( np.matmul((ground_truth - start_pos), normalizing_transform) )

                # Also store inverse transforms M^-1
                M_inverse_transforms.append(np.linalg.inv(normalizing_transform))

            # Flatten normalized targets - from a list of targets of len = number of training samples, where each
            # target is a num_landmarks-by-2 matrix, to a n-by-m matrix, where n = number of training samples and
            # m = 2 * num_landmarks, ie. [delta_x0, delta_y0, delta_x1, delta_y1, ...]
            Y_normalized_targets = np.vstack([target.flatten() for target in Y_normalized_targets])

            ynt = np.array(Y_normalized_targets)
            ynt_posavg = np.average(ynt[ynt>0])
            ynt_avg = np.average(ynt)
            ynt_var = np.var(ynt)
            ynt_argmax = np.unravel_index(np.argmax(ynt), ynt.shape)
            ynt_argmin = np.unravel_index(np.argmin(ynt), ynt.shape)
            print("\nRegression target info for stage", str(stage),":")
            print("Average:",ynt_avg)
            print("Positive average:",ynt_posavg)
            print("Std dev:",np.sqrt(ynt_var))

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
            current_stage_regressor = self.train_stage_regressor(Y_normalized_targets, num_features, num_ferns,
                                                                 num_fern_levels, pixel_feats)

            self.stage_regressors.append((current_stage_regressor, local_coords))

            # III. Update starting shapes in 'self.training_set' for next stage
            stage_regressor_output = self.test_stage_regressor(current_stage_regressor, local_coords,
                                                               pixel_feats, [t[0] for t in self.training_set],
                                                               [t[2] for t in self.training_set])

            csr = np.array(stage_regressor_output)
            crs_argmax = np.unravel_index(np.argmax(csr), csr.shape)
            crs_argmin = np.unravel_index(np.argmin(csr), csr.shape)
            for i in range(len(self.training_set)):
                self.training_set[i][2] += np.matmul(stage_regressor_output[i], M_inverse_transforms[i])

        if save_weights is not None:
            with open(save_weights, 'wb') as h:
                pickle.dump(self.stage_regressors, h, protocol=pickle.HIGHEST_PROTOCOL)

    def test(self, images, bagging_size):
        '''
        TODO - BW
        
        :param images: 
        :param bagging_size: 
        :return: 
        '''

        if bagging_size > len(self.test_start_shapes):
            raise ValueError('Not enough starting shapes initialized for selected bagging size')

        # Starting images will be converted to BW and will each have 'bagging_size' starting shapes
        # np.copy to avoid having multiple equal starting shapes (but associated with different images) pointing
        # to the same memory address

        initialized_images = []

        for img in images:
            for i in range(bagging_size):
                initialized_images.append([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), np.copy(self.test_start_shapes[i])])

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

    def load_weights(self, saved_weights_path):
        '''
        TODO
        
        :param saved_weights_path: 
        :return: 
        '''

        # Load stage regressors. Each regressor is a pair (regressor parameters, associated local coords)
        with open(saved_weights_path, 'rb') as h:
            self.stage_regressors = pickle.load(h)

        # Update class properties
        self.num_features = len(self.stage_regressors[0][1]) # Infer number of features used by regressors

    def train_stage_regressor(self, Y_normalized_targets, num_features, num_ferns, num_fern_levels,
                              pixel_features, threshold_multiplier = 0.2, beta = 5):
        '''
        Train a stage regressor, comprised of 'num_ferns' ferns, each with 'num_fern_levels' decision levels
        
        :param Y_normalized_targets: 
        :param num_features: 
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

                # TODO : Ensure global coords do not go outside image

                # Grab pixels from the start_pos image
                a = pixel_feats[i, j]
                b = images[i][int(global_coords[1]), int(global_coords[0])]
                pixel_feats[i, j] = images[i][int(global_coords[1]), int(global_coords[0])]

        return pixel_feats

    def generate_local_coordinates(self, local_random_displacement):
        '''
        Generate random local coordinate offsets relative to random landmarks
        
        :param local_random_displacement: 
        :return: TODO
        '''
        displacements = np.random.randint(-local_random_displacement, local_random_displacement,
                                                 size=(self.num_features, 2)) # Offsets to add to each landmark location

        landmark_indexes = np.random.randint(self.number_of_landmarks, size=(self.num_features, 1)) # Random landmark
        # locations

        return np.hstack((landmark_indexes, displacements)) # Combine both

def main():
    '''
    Testing function for class ESRegressor
    
    :return: --
    '''

    # Load processed images and landmarks

    datasetName = 'AFW/'
    imageExtension = '.jpg'

    os.chdir('./processed_data/' + datasetName)  # switch to dataset dir
    image_names = glob.glob('*' + imageExtension)
    image_names = [n.split('.')[0] for n in image_names]  # Get filenames without extension

    image_names_indexes = [int(n.split('_')[-1]) for n in image_names]

    pose_max = 4
    small_pose_image_names = []
    for i, index in enumerate(image_names_indexes):
        if index < pose_max:
            small_pose_image_names.append(image_names[i])

    images = []
    landmarks = []

    for img_name in small_pose_image_names[:-100]:
        # Load image
        images.append(cv2.imread(img_name + imageExtension))
        # Load landmarks
        landmarks.append(loadmat(img_name + '_pts.mat')['pts_2d'])

    # 1st. Instantiate Regressor

    os.chdir('../..')
    pwd = os.getcwd()
    weights_path = pwd + '/processed_data/esr_weights_v1beta.pickle'
    os.chdir('./processed_data/' + datasetName)

    R = ESRegressor(images, landmarks, 20, 20)

    print("Save path:",weights_path,"Train set len:",len(images))
    R.train(num_features=100, num_stages=1, num_ferns=100, local_random_displacement=15, save_weights=weights_path)
    R.load_weights(weights_path)

    # Grab some images for testing
    test_images = []
    test_landmarks = []

    for img_name in small_pose_image_names[-100:]:
        # Load image
        test_images.append(cv2.imread(img_name + imageExtension))
        # Load landmarks
        test_landmarks.append(loadmat(img_name + '_pts.mat')['pts_2d'])

    # regressed_landmarks = R.test(test_images, 15)
    # visualizer(test_images, landmarks=regressed_landmarks)

if __name__ == '__main__':
    log_name = os.getcwd() + '/profiling_log2.txt'
    cProfile.run('main()', log_name)
    p = pstats.Stats(log_name)
    p.strip_dirs().sort_stats('line').print_stats()
    p.sort_stats('cumulative').print_callees('correlation_feature_selection')
