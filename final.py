import skin_color_segmentation as scs
import numpy as np
import cv2
import os
import glob
import skimage.io as io
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import random

# get the image paths
dataset_path = [f'.\\Dataset_0-5\\men\\{i}\\' for i in range(0,6)] + [f'.\\Dataset_0-5\\women\\{i}\\' for i in range(0,6)]
image_paths = []
for path in dataset_path:
    image_paths += glob.glob(os.path.join(path, '*.JPG'))


# Define the HOG parameters
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (3, 3)
block_norm = 'L2-Hys'


# Extract the HOG features from each image
hog_features = []
labels = []
i = 0
for image_path in image_paths:
    i += 1
    image = io.imread(image_path)
    if image is not None:
        # resizd the image 
        mask = scs.process(image, thresh=0.3)
        mask[mask > 0] = 1
        mask = cv2.resize(mask, (512,256))
        image = cv2.resize(image, (512,256))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        print(i)

        image[mask == 0] = 0
        # get the segmented image 
        hog_feature = hog(image, orientations=orientations,
                        pixels_per_cell=pixels_per_cell,
                        cells_per_block=cells_per_block,
                        block_norm=block_norm)
        hog_features.append(hog_feature)
        label = os.path.basename(image_path).split('.')[0][0]
        labels.append(int(label))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(hog_features, labels, test_size=0.2, random_state=42)


# Apply PCA to the data 
pca = PCA(n_components=80)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Train the SVM classifier on the PCA-transformed data
svm = LinearSVC()
svm.fit(X_train_pca, y_train)

# Train the KNN classifier on the PCA-transformed data
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)

# Train the Naive Bayes classifier on the PCA-transformed data
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)

# # Train the Random Forest classifier on the PCA-transformed data
# rfc = RandomForestClassifier(n_estimators=100,random_state=42)
# rfc.fit(X_train, y_train)
# Evaluate the classifier on the PCA-transformed testing data
y_pred = svm.predict(X_test_pca)
print('y_test',y_test)
print('y_pred',y_pred)
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy*100) 