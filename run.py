# this script to read tht pickle file and predict the images, save test results to a file and time to another file
import pickle
import os
import skimage.io as io
import time
import skin_color_segmentation as scs
from skimage.feature import hog

model = pickle.load(open('svm.pkl', 'rb'))

# read the test images
data_set_path = '.\\data'
resultsFile = open('results.txt', 'w')
timeFile = open('time.txt', 'w')
# get the list of files in the folder
image_paths = os.listdir(data_set_path)

# the used hog parameters
orientations = 9
pixels_per_cell = (50, 50)
cells_per_block = (3, 3)
visualize = False
transform_sqrt = False
normalize = True

# sort the files in numeric order
image_paths = sorted(image_paths, key=lambda x: int(x.split('.')[0]))

# get images and predict
for image_path in image_paths:
    print(image_path)
    image = io.imread(os.path.join(data_set_path, image_path))
    if image is not None:
        time1 = time.time()
        image = scs.process(image, thresh=0.3)
        hog_feature = hog(image, orientations=orientations,
                                pixels_per_cell=pixels_per_cell,
                                cells_per_block=cells_per_block,transform_sqrt=transform_sqrt,visualize=visualize)
        result = model.predict([hog_feature])
        time2 = time.time()
        finalTime = time2 - time1
        timeFile.write('{:.3f}\n'.format(finalTime))
        resultsFile.write(f'{result[0]}\n')

resultsFile.close()
timeFile.close()