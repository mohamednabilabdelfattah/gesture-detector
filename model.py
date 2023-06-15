import glob
import os
from sklearn.svm import SVC
from skimage import io
import skin_color_segmentation as scs
from skimage.feature import hog
import pickle
import time

class main:
    # the used classifier
    svm = SVC(kernel='poly', degree=4, C=1.0)
    # the used hog parameters
    orientations = 9
    pixels_per_cell = (50, 50)
    cells_per_block = (3, 3)
    visualize = False
    transform_sqrt = False
    normalize = True
    

    def train(self):
        hog_features = []
        labels = []
        # get the image paths
        dataset_path = [f'.\\Dataset_0-5\\men\\{i}\\' for i in range(0,6)] + [f'.\\Dataset_0-5\\Women\\{i}\\' for i in range(0,6)]
        image_paths = []
        for path in dataset_path:
            image_paths += glob.glob(os.path.join(path, '*.JPG'))
        
        for image_path in image_paths:
            image = io.imread(image_path)
            if image is not None:
                # resizd the image
                mask = scs.process(image, thresh=0.3)
                # io.imsave(f'.\\preprocessed\\{i}.jpg',mask)
                hog_feature = hog(mask, orientations=self.orientations,
                                pixels_per_cell=self.pixels_per_cell,
                                cells_per_block=self.cells_per_block,transform_sqrt=self.transform_sqrt,visualize=self.visualize)
                hog_features.append(hog_feature)
                label = os.path.basename(image_path).split('.')[0][0]
                labels.append(int(label))
        self.svm.fit(hog_features, labels)
        pickle.dump(self.svm, open('svm.pkl', 'wb'))
    
    def predictOneImage(self, image):
        mask = scs.process(image, thresh=0.3)
        hog_feature = hog(mask, orientations=self.orientations,
                                pixels_per_cell=self.pixels_per_cell,
                                cells_per_block=self.cells_per_block,transform_sqrt=self.transform_sqrt,visualize=self.visualize)
        return self.svm.predict(hog_feature.reshape(1, -1))
    
    def predict(self):
        data_set_path = '.\\data'
        resultsFile = open('results.txt', 'w')
        timeFile = open('time.txt', 'w')
        image_paths = glob.glob(os.path.join(data_set_path, '*.JPG'))
        for image_path in image_paths:
            image = io.imread(image_path)
            if image is not None:
                time1 = time.time()
                result = self.predictOneImage(image)
                time2 = time.time()
                finalTime = time2 - time1
                timeFile.write('{:.3f}\n'.format(finalTime))
                resultsFile.write(f'{result}\n')
        resultsFile.close()
        timeFile.close()

            
main = main()
main.train()
# main.predict()
