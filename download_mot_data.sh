#!/bin/bash

# Download multi object tracking (MOT) data for vehicle boundaries
wget http://download.europe.naverlabs.com/virtual-kitti-1.3.1/vkitti_1.3.1_motgt.tar.gz
tar -xvf vkitti_1.3.1_motgt.tar.gz
mv vkitti_1.3.1_motgt vkitti3d_mot
rm vkitti_1.3.1_motgt.tar.gz