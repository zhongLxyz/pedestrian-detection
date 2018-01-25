
import os


def get_file_names(path):
    """ Get full path to all images in parameter path """
    namelist = os.listdir(path)
    filtered_list = []
    for i in range(len(namelist)):
        if namelist[i].endswith(""):
            filtered_list.append(path + namelist[i])
    return filtered_list


# Store names of all dataset images in array
# X_name/X_test_name are positive datasets
# Y_name/Y_test_name are negative datasets
X_name = get_file_names("./1/ped_examples/")
X_name2 = get_file_names("./2/ped_examples/")
X_name3 = get_file_names("./3/ped_examples/")
Y_name = get_file_names("./1/non-ped_examples/")
Y_name2 = get_file_names("./2/non-ped_examples/")
Y_name3 = get_file_names("./3/non-ped_examples/")
X_test_name = get_file_names("./T1/ped_examples/")
Y_test_name = get_file_names("./T1/non-ped_examples/")


# write path to dataset images into my_dataset.txt for h5py_dumper.py to generate dataset
data = open("my_dataset.txt", 'w')
for i in range(len(X_name)):
    data.write(X_name[i] + " 1\n")
    data.write(X_name2[i] + " 1 \n")
    data.write(X_name3[i] + " 1 \n")
for i in range(len(Y_name)):
    data.write(Y_name[i] + " 0\n")
    data.write(Y_name2[i] + " 0\n")
    data.write(Y_name3[i] + " 0\n")
    # data.write(Y_test_name[i] + " 3 \n")

data.close()

# write path to dataset images into my_testset.txt for h5py_dumper.py to generate dataset
data = open("my_testset.txt", 'w')
for i in range(len(X_test_name)):
    data.write(X_test_name[i] + " 1\n")

for i in range(len(Y_test_name)):
    data.write(Y_test_name[i] + " 0\n")

data.close()
