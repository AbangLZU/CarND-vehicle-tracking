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
from sklearn.model_selection import train_test_split


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
        file_list = glob.glob(path+'/*/*.png')
        return file_list

    def get_features(self):
        color_space = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        orient = 12  # HOG orientations
        pix_per_cell = 16  # HOG pixels per cell
        cell_per_block = 2  # HOG cells per block
        hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
        spatial_size = (32, 32)  # Spatial binning dimensions
        hist_bins = 64  # Number of histogram bins
        spatial_feat = False  # Spatial features on or off
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
    clf = svm.SVC()
    # parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    # svc = svm.SVC()
    # clf = GridSearchCV(svc, parameters)
    clf.fit(X_train, y_train)

    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
    return clf

if __name__ == '__main__':
    data = DataReader('vehicles', 'non-vehicles')

    clf = model(data.X_train, data.y_train, data.X_test, data.y_test)

    # save the svc Model and Scaler
    with open('svc.pickle', 'wb') as f:
        disk_pickle = {}
        disk_pickle['clf'] = clf
        disk_pickle['X_scaler'] = data.X_scaler
        pickle.dump(disk_pickle, f)

    # test_on_new_img()



