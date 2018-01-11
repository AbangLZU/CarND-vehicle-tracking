from __future__ import print_function
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from feature_helper import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split


class DataReader(object):
    def __init__(self, car_root_path, no_car_root_path):
        self.car_path_list = self.read_file_list(car_root_path)
        self.noncar_path_list = self.read_file_list(no_car_root_path)

        car_features, notcar_features = self.get_features()

        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        self.X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        self.scaled_X = self.X_scaler.transform(X)

        # Define the labels vector
        self.y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.scaled_X, self.y, test_size=0.2,
                                                                                random_state=rand_state)

        print('the shape of the X_train are', self.X_train.shape)
        print('number of test samples is', len(self.y_test), np.sum(self.y_test), 'car samples and ',
              len(self.y_test) - np.sum(self.y_test), 'noncar samples')

    def read_file_list(self, path):
        file_list = glob.glob(path+'/*/*.jpeg')
        return file_list

    def get_features(self):
        color_space = 'HLS'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        orient = 9  # HOG orientations
        pix_per_cell = 8  # HOG pixels per cell
        cell_per_block = 2  # HOG cells per block
        hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
        spatial_size = (16, 16)  # Spatial binning dimensions
        hist_bins = 64  # Number of histogram bins
        spatial_feat = True  # Spatial features on or off
        hist_feat = True  # Histogram features on or off
        hog_feat = True  # HOG features on or off

        print('Using:', orient, 'orientations', pix_per_cell,
              'pixels per cell and', cell_per_block, 'cells per block')

        car_features = extract_features(self.car_path_list, color_space=color_space,
                                        spatial_size=spatial_size, hist_bins=hist_bins,
                                        orient=orient, pix_per_cell=pix_per_cell,
                                        cell_per_block=cell_per_block,
                                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                                        hist_feat=hist_feat, hog_feat=hog_feat)
        notcar_features = extract_features(self.noncar_path_list, color_space=color_space,
                                           spatial_size=spatial_size, hist_bins=hist_bins,
                                           orient=orient, pix_per_cell=pix_per_cell,
                                           cell_per_block=cell_per_block,
                                           hog_channel=hog_channel, spatial_feat=spatial_feat,
                                           hist_feat=hist_feat, hog_feat=hog_feat)

        return car_features, notcar_features


def model(X_train, y_train, X_test, y_test):
    # Check the training time for the SVC
    t = time.time()
    clf = svm.LinearSVC()
    # parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    # svc = svm.SVC()
    # clf = GridSearchCV(svc, parameters)
    clf.fit(X_train, y_train)

    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
    return clf


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


def test_on_new_img():
    # load the Model
    with open('svc.pickle', 'rb') as f:
        disk_pickle = pickle.load(f)
        clf = disk_pickle['clf']
        X_scaler = disk_pickle['X_scaler']

    color_space = 'HLS'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16)  # Spatial binning dimensions
    hist_bins = 64  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off
    y_start_stop = [400, 700]  # Min and max in y to search in slide_window()
    test_img_paths = glob.glob('test_images/*.jpg')
    plt.figure(figsize=(8, 12))
    for i, item in enumerate(test_img_paths):
        # notice that the mpimg.imread() to read jpg is np array ~(0, 255) ,not (0, 1)
        img = mpimg.imread(item)
        draw_image = np.copy(img)
        windows = slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop,
                               xy_window=(128, 128), xy_overlap=(0.5, 0.5))

        hot_windows = search_windows(img, windows, clf, X_scaler, color_space=color_space,
                                     spatial_size=spatial_size, hist_bins=hist_bins,
                                     orient=orient, pix_per_cell=pix_per_cell,
                                     cell_per_block=cell_per_block,
                                     hog_channel=hog_channel, spatial_feat=spatial_feat,
                                     hist_feat=hist_feat, hog_feat=hog_feat)

        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
        plt.subplot(3, 2, i + 1)
        plt.imshow(window_img)

    plt.show()

if __name__ == '__main__':
    # data = DataReader('vehicles_smallset', 'non-vehicles_smallset')
    #
    # clf = model(data.X_train, data.y_train, data.X_test, data.y_test)
    #
    # # save the svc Model and Scaler
    # with open('svc_smallset.pickle', 'wb') as f:
    #     disk_pickle = {}
    #     disk_pickle['clf'] = clf
    #     disk_pickle['X_scaler'] = data.X_scaler
    #     pickle.dump(disk_pickle, f)

    test_on_new_img()






