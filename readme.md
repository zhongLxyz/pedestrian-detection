# Pedestrian Detection with Neural networks

A convolutional neural network that can be trained to detect pedestrians from the other objects on the street.

This project was completed in December 2016 in a team of 4.

## How to train
1. run "python3 generate_dataset_list.py" to generate list of dataset files & their class
2. run "python3 h5py_dumper.py" to generate dataset.h5 and testset.h5
3. Finally, run "python train.py" to begin training

OR

Simply run “python3 generate_dataset_list.py && python3 h5py_dumper.py && python train.py”.

NOTE: Images for training & validation can be found here: [Daimler Pedestrian Classification Benchmark Dataset] (http://www.gavrila.net/Datasets/Daimler_Pedestrian_Benchmark_D/Daimler_Mono_Ped__Class__Bench/daimler_mono_ped__class__bench.html)

## Flow Diagram

! [flow diagram] (others/flow_diagram.PNG)

## How to test
Simply run “python3 model_tester.py” in same directory as testset.h5 (also generated using h5py)

Output result:

Confusion Matrix:
Non pedestrians images: [4935 classified as non pedestrian] / [65 classified as pedestrian]
Pedestrians images: [4215 classified as non pedestrian] / [585 classified as pedestrian]
