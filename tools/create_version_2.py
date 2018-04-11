import numpy as np
import pandas as pd
import argparse
from argparse import RawTextHelpFormatter
import os
from tqdm import *
import glob


def distance_cutoff(pointcloud, cutoff):
    cut = np.sqrt(np.square(pointcloud[:,:3]).sum(axis=1)) < cutoff
    return pointcloud[cut]


def main(mot_path, pc_paths, eps, out_path, cutoff):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    progressbar = tqdm(total=len(glob.glob(pc_paths + '**/*.npy')))

    for dirpath, dirnames, filenames in os.walk(pc_paths):
        for filename in [f for f in filenames if f.endswith(".npy")]:
            progressbar.set_description("Processing %s" % filename)

            pointcloud = np.load(os.path.join(dirpath, filename))

            new_seq_number = dirpath.split('/')[-1]

            if not os.path.exists(os.path.join(out_path, new_seq_number)):
                os.makedirs(os.path.join(out_path, new_seq_number))

            original_seq_number, frame_number = filename.split('_')
            frame_number = int(frame_number.replace('.npy', ''))

            pointcloud = distance_cutoff(pointcloud, cutoff)

            mot_data = pd.read_csv(os.path.join(mot_path, original_seq_number + '_clone.txt'), sep=" ", index_col=False)
            mot_data = mot_data[mot_data['frame'] == frame_number]

            # save indices of voxels belonging to vehicles such that we do not delete their class labels
            vehicle_idx = []

            for index, row in mot_data.iterrows():
                if row['orig_label'] in ['Car', 'Van']:
                    cc_y = -row['x3d']
                    cc_z = -row['y3d']
                    cc_x = row['z3d']

                    rot_x = row['rz']
                    rot_y = -row['rx']
                    rot_z = row['ry']

                    width = row['w3d']
                    length = row['l3d']
                    height = row['h3d']

                    transformed = pointcloud.copy()

                    translation_vector = np.zeros(transformed.shape[1])
                    translation_vector[:3] = [cc_x, cc_y, cc_z]

                    # set the coordinate system origin to the center of the vehicle
                    transformed -= translation_vector

                    rotation_matrix = get_rotation_matrix(rot_x, rot_y, rot_z)

                    # rotate all points such that the vehicle is axis-aligned
                    transformed[:, :3] = np.einsum('ij, kj -> ki', rotation_matrix, transformed[:, :3])

                    width_cond = (transformed[:, 0] <= width / 2 + eps) & (transformed[:, 0] >= -width / 2 - eps)
                    length_cond = (transformed[:, 1] <= length / 2 + eps) & (transformed[:, 1] >= -length / 2 - eps)
                    height_cond = transformed[:, 2] <= height + eps

                    # for cond == true, the corresponding voxel is a vehicle voxel
                    cond = width_cond & length_cond & height_cond

                    vehicle_idx.extend(np.where(cond)[0])

            # set to don't care class
            check_idx = list(set(range(pointcloud.shape[0])) - set(vehicle_idx))
            pointcloud[np.array(check_idx)[np.where(pointcloud[check_idx, -1] >= 11)[0]], -1] = 13

            # convert to float32
            pointcloud = pointcloud.astype(np.float32)

            # save fixed point cloud
            np.save(os.path.join(out_path, new_seq_number, filename), pointcloud)

            # update progress bar
            progressbar.update(1)

    progressbar.close()


def get_rotation_matrix(rot_x, rot_y, rot_z):
    c, s = np.cos(rot_z), np.sin(rot_z)

    RZ = np.matrix(
        '{} {} 0;'
        '{} {} 0;'
        ' 0  0 1'.format(c, -s, s, c)
    )

    c, s = np.cos(rot_x), np.sin(rot_x)

    RX = np.matrix(
        ' 1  0  0;'
        ' 0 {} {};'
        ' 0 {} {}'.format(c, -s, s, c)
    )

    c, s = np.cos(rot_y), np.sin(rot_y)

    RY = np.matrix(
        '{} {}  0;'
        '{} {}  0;'
        ' 0  0  1'.format(c, -s, s, c)
    )

    return RX.dot(RY).dot(RZ)


def arg_check_positive(value):
    val = float(value)
    if val <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive float value" % value)
    return val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                     description="Fix the projection error occuring in the original VKITTI "
                                                 "numpy pointclouds introduced by the wrong depth of car windows and "
                                                 "cut off points which are farer away than a specified distance from "
                                                 "the camera. Please note that vehicles farer away than 100m are not "
                                                 "labeled anymore.\n"
                                                 "The problem with the original point clouds is that they are created "
                                                 "from RGBD images. Pixels behind a car glass will get the car class "
                                                 "label which is erroneous and thus, decreasing the quality of the "
                                                 "test and training set.")

    parser.add_argument('--mot_path', nargs='?', const='../vkitti3d_mot/',
                        type=str, default='../vkitti3d_mot/',
                        help='path to the .csv file containing the multi object tracking data '
                             'for bounding boxes of vehicles')

    parser.add_argument('--pc_path', nargs='?', const='../vkitti3d_dataset_original/',
                        help='path to the root directory of sequence folders containing the original vkitti '
                             'pointclouds in numpy format',
                        default='../vkitti3d_dataset_original/')

    parser.add_argument('--out_path', nargs='?', const='../vkitti3d_dataset_fixed_0_3/',
                        help='path to the directory where the fixed pointclouds should be saved',
                        default='../vkitti3d_dataset_fixed_0_3/')

    parser.add_argument('--eps', const=0.3, default=0.3, type=arg_check_positive, nargs='?',
                        help='since the bounding boxes are not 100% accurate introduce an epsilon >= 0 to enlarge the '
                             'bounding boxes a bit')

    parser.add_argument('--cutoff', const=100, default=100, type=arg_check_positive, nargs='?',
                        help='distance cutoff: points farer away than specified cutoff distance will be discarded')

    args = parser.parse_args()

    main(
        mot_path=args.mot_path,
        pc_paths=args.pc_path,
        eps=args.eps,
        out_path=args.out_path,
        cutoff=args.cutoff
)