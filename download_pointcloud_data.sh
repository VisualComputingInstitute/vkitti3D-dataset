#!/bin/bash

# Download pointcloud data for unaltered vkitti3d dataset
wget https://www.vision.rwth-aachen.de/media/resource_files/vkitti3d_dataset_v1.0.zip
unzip vkitti3d_dataset_v1.0.zip
mv vkitti3d_dataset_v1.0 vkitti3d_dataset_original
rm vkitti3d_dataset_v1.0.zip
