
# Import the necessary libraries
import tflearn
from tflearn.data_utils import build_hdf5_image_dataset


# Generate dataset.h5 using training samples
dataset_file = 'my_dataset.txt'
build_hdf5_image_dataset(dataset_file, image_shape=(32, 32), mode='file',
        output_path='dataset.h5', categorical_labels=True, normalize=True)
print("Done generating training dataset.")


# Generate testset.h5 using validation samples
dataset_file = 'my_testset.txt'
build_hdf5_image_dataset(dataset_file, image_shape=(32, 32), mode='file',
        output_path='testset.h5', categorical_labels=True, normalize=True)
print("Done generating testing dataset.")
