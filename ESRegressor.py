import os, random, glob
import numpy as np
import cv2 as cv2
from scipy.linalg import orthogonal_procrustes
from scipy.io import loadmat

from utils.visualizer import visualizer

class ESRegressor:
    '''
    Explicit Shape Regression implementation, by 
    Xudong Cao, Yichen Wei, Fang Wen, Jian Sun as described in:
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

        self.training_set = [] # Will contain training tuples in the format (image, ground_truth, start_pos)
        self.test_start_shapes = [] # Will contain fixed test start shapes for bagging

        for i, (img, ground_truth) in enumerate(zip(training_images, ground_truths)):

            # Convert img to BW
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Get 'num_augmentation' indexes without replacement
            unique_starting_positions = random.sample(range(0,len(ground_truths)), num_augmentation)

            for start_pos in unique_starting_positions:
                self.training_set.append((img, ground_truth, ground_truths[start_pos]))

        testing_start_positions = random.sample(range(0, len(ground_truths)), num_test_shapes)

        for pos in testing_start_positions:
            self.test_start_shapes.append(ground_truths[pos])

        # 2. Initialize mean shape

        self.mean_shape = np.mean(ground_truths, axis=0)

        # 3. Other initializations

        self.number_of_landmarks = len(ground_truths[0])
        self.stage_regressors = []

    def train(self, num_features = 400, local_random_displacement = 25, num_stages = 10,
              num_ferns = 500, num_fern_levels = 5):
        '''
        Train the ESRegressor
        
        :param num_features: (int) Number of features to sample per image
        :param local_random_displacement: (int) Range of uniform random noise when determining feature points
        in local landmark coordinates, in pixels
        :param num_stages: (int) Number of boosted stages
        :param num_ferns: (int) Number of ferns per stage
        :param num_fern_levels: (int) Number of levels per fern
        :return: --
        '''

        for stage in range(0, num_stages): # For each stage in the regressor, we must ...

            Y_normalized_targets = [] # Will contain normalized targets for this stage
            M_inverse_transforms = [] # Will contain normal transformations for this stage

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

            # II. Train and store current stage regressor
            current_stage_regressor = self.train_stage_regressor(Y_normalized_targets, num_features, num_ferns,
                                                                 num_fern_levels, local_random_displacement,
                                                                 M_inverse_transforms)

            self.stage_regressors.append(current_stage_regressor)

            # III. Update starting shapes in 'self.training_set' for next stage
            stage_regressor_output = self.test_stage_regressor(current_stage_regressor[0], current_stage_regressor[1],
                                                               [t[0] for t in self.training_set],
                                                               [t[2] for t in self.training_set])

            for i in range(len(self.training_set)):
                self.training_set[i][2] += np.matmul(stage_regressor_output[i], M_inverse_transforms[i])

    def test(self, images, bagging_size):
        '''
        TODO
        
        :param images: 
        :param bagging_size: 
        :return: 
        '''

        if bagging_size > len(self.test_start_shapes):
            raise ValueError('Not enough starting shapes initialized for selected bagging size')



    def train_stage_regressor(self, Y_normalized_targets, num_features, num_ferns, num_fern_levels,
                              local_random_displacement, inverse_transforms, threshold_multiplier = 0.2, beta = 5):
        '''
        Train a stage regressor, comprised of 'num_ferns' ferns, each with 'num_fern_levels' decision levels
        
        :param Y_normalized_targets: 
        :param num_features: 
        :param num_ferns: 
        :param num_fern_levels: 
        :param local_random_displacement: 
        :param inverse_transforms: 
        :param threshold_multiplier:
        :param beta:
        :return: TODO
        '''

        # I. Initial setup

        # 1. Generate pixel features
        pixel_feats, local_coords = self.generate_pixel_features(num_features, local_random_displacement, inverse_transforms)

        # 2. Compute pixel-pixel covariance matrix
        pix2pix_cov_matrix = np.cov(pixel_feats)

        # 3. Initialize current regressed displacements as the original target displacements
        Y_current = Y_normalized_targets

        # II. Start training the ferns

        stage_ferns = []

        for fern_number in num_ferns: # For each weak regressor ...

            # ... select the best pixel-difference features for this fern ...
            selected_pixel_diff_features, diff_indexes = self.correlation_feature_selection(Y_current, num_features,
                                                                                            pixel_feats,
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
            stage_ferns.append([thresholds, diff_indexes, bin_outputs])

            # Update regression targets for next weak regressor
            for training_sample_number, bin_index in enumerate(bin_index_per_training_sample):
                Y_current[training_sample_number, :] -= bin_outputs[bin_index]

        # 2. Return the complete stage regressor

        return (stage_ferns, local_coords)

    def test_stage_regressor(self, stage_ferns, local_coords, images, current_shapes):
        '''
        Apply the supplied regressor stage to all images and corresponding shapes provided in 'images' and
        'current_shapes'
        
        :param stage_ferns: 
        :param local_coords: 
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

        pixel_feats = self.extract_shape_indexed_pixels(images, current_shapes, len(local_coords),
                                                        local_coords, normalized_transforms)

        delta_shapes = [shape * 0 for shape in current_shapes] # Placeholder for the regressions for each shape

        for k, fern in enumerate(stage_ferns):

            # Reminder: a fern is a (a,b,c) tuple where a=thresholds, b=feature difference indexes, c=bin outputs

            # 1. Get selected feature differences for this fern

            selected_feats = np.zeros(len(images), len(fern[0])) # Placeholder for a n-by-m matrix, where n is the
            # number of training images and m is the number of levels in a fern (here extracted by number of thresholds)

            for k, (i,j) in enumerate(fern[1]):
                selected_feats[:, k] = pixel_feats[:, i] - pixel_feats[:, j]

            # 2. Get bin numbers for all images

            bin_numbers = self.compute_fern_bin(selected_feats, fern[0])

            # 3. Get and reshape fern output and add to current offsets

            fern_outputs = [fern[2][bin_number] for bin_number in bin_numbers]
            final_deltas = [np.reshape(fern_output, (self.number_of_landmarks, 2)) for fern_output in fern_outputs]

            for i, final_delta in enumerate(final_deltas):
                delta_shapes[i] += final_delta

        return delta_shapes



    def correlation_feature_selection(self, Y_regression_target, num_features, pixel_features,
                                      pix2pix_cov_matrix, num_fern_levels):
        '''
        Select 'num_fern_levels' pixel difference features with highest correlation to displacement projections
        as detailed in the paper
        
        :param Y_regression_target:
        :param num_features:
        :param pixel_features:
        :param pix2pix_cov_matrix: 
        :param num_fern_levels: 
        :return: (tuple of 2 lists) First list contains selected pixel difference feature values, second list
        contains corresponding indexes
        '''

        pixel_difference_features = []
        pixel_difference_indexes = []

        for fern_level in range(num_fern_levels):

            # Random projection from unit gaussian, normalized
            gaussian_proj = np.random.normal(0, 1, size=(self.number_of_landmarks, 1))
            gaussian_proj /= np.linalg.norm(gaussian_proj)

            # Projected target for this fern level
            Y_proj = np.matmul(Y_regression_target, gaussian_proj)

            target_pix_cov = np.zeros(num_features) # Placeholder, filled below

            # Compute target-pixel covariance
            for i in range(num_features):
                target_pix_cov[i] = np.cov(Y_proj, pixel_features[:, i])[0, 1] # [0, 1] (or [1, 0]) to fetch the
                # covariance term from the covariance matrix

            # Compute project target variance
            Y_var = np.var(Y_proj)

            # To be filled below
            corr_matrix = np.ones((num_features, num_features)) * -1

            for i in range(num_features):
                for j in range(num_features):
                    corr = (target_pix_cov[i] - target_pix_cov[j]) / np.sqrt(
                        Y_var *(pix2pix_cov_matrix[i,i] + pix2pix_cov_matrix[j,j] - 2 * pix2pix_cov_matrix[i,j]) )

                    corr_matrix[i, j] = corr

            # Get max correlation pixel pair
            i_max, j_max = np.unravel_index(np.argmax(corr_matrix), corr_matrix.shape)

            pixel_difference_features.append(pixel_features[:,i_max] - pixel_features[:, j_max])
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
        binary_checkpoint_results = selected_feature_differences >= thresholds

        # Convert to base ten
        final_bin_numbers = selected_feature_differences[:,0] * 0 # Initialize vector that'll contain final bin numbers
        for level in range(0, num_levels):
            final_bin_numbers += selected_feature_differences[:,level] * 2**(num_levels - level - 1) # Base 2 to 10

        return final_bin_numbers

    def generate_pixel_features(self, num_features, local_random_displacement, inverse_transforms):
        '''
        (1) Generate a N-by-'num_features' array, where N is the training set size, and feature values are
        pixel intensities extracted according to the shape indexed feautre scheme described in the paper
        
        :param num_features: (int) Number of features to extract per image
        :param local_random_displacement: (int) Range of uniform random noise when determining feature points
        in local landmark coordinates, in pixels
        :param inverse_transforms: (list) Normalized transforms for the training set at this stage
        :return: (numpy array, numpy array) Pixel features array as described above ( see (1) ), and local
        coordinates (an 3-by-'num_features' array, with each line being [lm number, x coord, y coord])
        '''

        # 1. Generate local coordinates
        local_coords = self.generate_local_coordinates(num_features, local_random_displacement)

        # 2. Get feature locations and pixel intensities in original image, global coordinates

        # Reminder : training_set stores a tuple for each training sample, with tuple[0] being the image, [1] being
        # the ground truth and [2] the starting position at this stage
        pixel_feats = self.extract_shape_indexed_pixels([t[0] for t in self.training_set],
                                                        [t[2] for t in self.training_set],
                                                        num_features, local_coords, inverse_transforms)

        return pixel_feats, local_coords

    def extract_shape_indexed_pixels(self, images, current_shapes, num_features, local_coords, inverse_transforms):
        '''
        Extract pixel intensities in the original image-shape pairs provided, in global coordinates
        
        :param images: 
        :param current_shapes: 
        :param num_features: 
        :param local_coords: 
        :param inverse_transforms: 
        :return: TODO
        '''

        pixel_feats = np.zeros((len(images), num_features))  # Placeholder for pixel intensities
        pixel_feats_y_dim, pixel_feats_x_dim = pixel_feats.shape
        global_coordinates = np.zeros(
            (2 * pixel_feats_x_dim, 2 * pixel_feats_y_dim))  # Place holder for global coordinates

        for i in range(pixel_feats_y_dim):
            for j in range(pixel_feats_x_dim):

                # Get landmark coords in start_pos for the sample being requested (sample i) for the landmark being
                # requested (must check which landmark is associated with the feature being requested, feature j)
                landmark_coords_in_start_pos = current_shapes[i][local_coords[j, 0], :]

                # Transform the local coords back to 'start_pos space'
                local_coords_transformed_back_to_start_pos = np.matmul(inverse_transforms[i],
                                                                       local_coords[j, 1:])
                global_coords = local_coords_transformed_back_to_start_pos + landmark_coords_in_start_pos
                global_coordinates[j, 2 * i:2 * i + 2] = global_coords

                # Grab pixels from the start_pos image
                pixel_feats[i, j] = images[i][int(global_coords[1]), int(global_coords[0])]

        return pixel_feats

    def generate_local_coordinates(self, num_features, local_random_displacement):
        '''
        Generate random local coordinate offsets relative to random landmarks
        
        :param num_features: 
        :param local_random_displacement: 
        :return: TODO
        '''
        displacements = np.random.randint(-local_random_displacement, local_random_displacement,
                                                 size=(num_features, 2)) # Offsets to add to each landmark location

        landmark_indexes = np.random.randint(self.number_of_landmarks, size=(num_features, 1)) # Random landmark
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

    images = []
    landmarks = []

    for img_name in image_names[:100]:
        # Load image
        images.append(cv2.imread(img_name + imageExtension))
        # Load landmarks
        landmarks.append(loadmat(img_name + '_pts.mat')['pts_2d'])

    # 1st. Instantiate Regressor

    R = ESRegressor(images, landmarks, 20, 5)
    R.train()

if __name__ == '__main__':
    main()
