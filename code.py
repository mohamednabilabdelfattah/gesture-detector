import skin_color_segmentation as scs
import numpy as np
import os
import glob
import skimage.io as io
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

# get the image paths
dataset_path = [f'.\\Dataset_0-5\\men\\{i}\\' for i in range(0,6)] + [f'.\\Dataset_0-5\\Women\\{i}\\' for i in range(0,6)]
image_paths = []
for path in dataset_path:
    image_paths += glob.glob(os.path.join(path, '*.JPG'))


orientations = 9
pixels_per_cell = (50, 50)
cells_per_block = (3, 3)
visualize = False
transform_sqrt = False
normalize = True

# Extract the HOG features from each image
hog_features = []
labels = []
for image_path in image_paths:
    image = io.imread(image_path)
    if image is not None:
        # resizd the image
        mask = scs.process(image, thresh=0.3)
        # io.imsave(f'.\\preprocessed\\{i}.jpg',mask)
        hog_feature = hog(mask, orientations=orientations,
                        pixels_per_cell=pixels_per_cell,
                        cells_per_block=cells_per_block,transform_sqrt=transform_sqrt,visualize=visualize)
        hog_features.append(hog_feature)
        label = os.path.basename(image_path).split('.')[0][0]
        labels.append(int(label))


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(hog_features, labels, test_size=0.2, random_state=42)


# Train the SVM classifier on the PCA-transformed data
svm = SVC(kernel='poly', degree=4, C=1.0)
svm.fit(X_train, y_train)


# Evaluate the classifier on the PCA-transformed testing data
y_pred = svm.predict(X_test)    


# Accuracy score
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Precision, recall, and F1 score
precision = metrics.precision_score(y_test, y_pred,average='weighted')
recall = metrics.recall_score(y_test, y_pred,average='weighted')
f1_score = metrics.f1_score(y_test, y_pred,average='weighted')
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1_score)

# Confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
print('Confusion matrix:\n', conf_matrix)

# ROC curve and AUC score
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
print('ROC AUC score:', roc_auc)