#!/bin/bash

cd $1

# Download extrinsic parameters
wget http://download.europe.naverlabs.com/virtual-kitti-1.3.1/vkitti_1.3.1_extrinsicsgt.tar.gz
tar -xvf vkitti_1.3.1_extrinsicsgt.tar.gz
rm vkitti_1.3.1_extrinsicsgt.tar.gz

# Download RGB images
wget http://download.europe.naverlabs.com/virtual-kitti-1.3.1/vkitti_1.3.1_rgb.tar
tar -xvf vkitti_1.3.1_rgb.tar
rm vkitti_1.3.1_motgt.tar.gz

# Download semantic ground truth
wget http://download.europe.naverlabs.com/virtual-kitti-1.3.1/vkitti_1.3.1_scenegt.tar
tar -xvf vkitti_1.3.1_scenegt.tar
rm vkitti_1.3.1_scenegt.tar

# Download semantic ground truth
wget http://download.europe.naverlabs.com/virtual-kitti-1.3.1/vkitti_1.3.1_depthgt.tar
tar -xvf vkitti_1.3.1_depthgt.tar
rm vkitti_1.3.1_depthgt.tar

# Download multi object tracking (MOT) data for vehicle boundaries
wget http://download.europe.naverlabs.com/virtual-kitti-1.3.1/vkitti_1.3.1_motgt.tar.gz
tar -xvf vkitti_1.3.1_motgt.tar.gz
rm vkitti_1.3.1_motgt.tar.gz

echo Finished