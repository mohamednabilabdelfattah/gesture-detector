import os
import skimage.io as io

data_set_path = '.\\data'
new_data_set_path = '.\\new_data'
image_paths = os.listdir(data_set_path)
new_image_paths = [image_paths[i].lstrip(' (')[0:-5].rstrip(')')+'.jpg' for i in range(len(image_paths))]

for i in range(len(image_paths)):
    image = io.imread(os.path.join(data_set_path, image_paths[i]))
    io.imsave(os.path.join(new_data_set_path, new_image_paths[i]), image)