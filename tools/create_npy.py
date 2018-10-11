import numpy as np
import pandas as pd
import os
import glob
import cv2
import pandas
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
from argparse import RawTextHelpFormatter
from typing import Tuple
import time


center_x = 620.5
center_y = 187.0
focal_x = 725.0
focal_y = 725.0

yv, xv = np.meshgrid(range(375), range(1242), indexing='ij')

rgb_root = None
depth_root = None
labels_root = None
mot_root = None
flow_root = None
extrinsic_root = None


g_is_v1 = None
g_bb_eps = 0.3
g_cutoff = 100.

extgt = None
encoding_df = None
mot_data = None
rgb2label = {}

sem2label = {
    'Terrain': 0,
    'Tree': 1,
    'Vegetation': 2,
    'Building': 3,
    'Road': 4,
    'GuardRail': 5,
    'TrafficSign': 6,
    'TrafficLight': 7,
    'Pole': 8,
    'Misc': 9,
    'Truck': 10,
    'Car': 11,
    'Van': 12,
    'Sky': 99
}


def parallel_process(array, function, n_jobs: int = 8, use_kwargs: bool = False, front_num: int = 0):
    """
        A parallel version of the map function with a progress bar.
        copyright: http://danshiebler.com/2016-09-14-parallel-progress-bar/

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=8): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of
                keyword arguments to function
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job.
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    front = []
    # We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    # If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs == 1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    # Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        # Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    # Get the results from the futures.
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out


def get_rotation_matrix(rot_x: np.ndarray, rot_y: np.ndarray, rot_z: np.ndarray) -> np.ndarray:
    """
    create 3D rotation matrix given rotations around axes in rad
    :param rot_x: rotation around x in rad
    :param rot_y: rotation around y in rad
    :param rot_z: rotation around z in rad
    :return: 3D rotation matrix
    """
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


def knn_interpolation(pointcloud: np.ndarray, k: int = 1, eps: float = 0.2, njobs: int = 4) -> np.ndarray:
    """
    points without label will get an interpolated label given the k closest neighbors with a maximal distance of eps
    :param pointcloud: points (labeled and unlabeled)
    :param k: maximal k neighbors are considered
    :param eps: considered neighbors have to be in range of eps
    :return: interpolated point cloud
    """
    labeled = pointcloud[pointcloud[:, -1] != 13]

    unlabeled_idx = (pointcloud[:, -1] == 13)
    to_be_predicted = pointcloud[unlabeled_idx]

    neigh = NearestNeighbors(n_neighbors=k, radius=eps, algorithm='ball_tree', metric='euclidean', n_jobs=njobs)
    neigh.fit(labeled[:, :3])

    if to_be_predicted.shape[0] != 0:
        dist, ind = neigh.kneighbors(to_be_predicted[:, :3])

        knn_classes = labeled[ind][:, :, -1].astype(int)
        knn_classes[dist > eps] = 13

        pointcloud[unlabeled_idx, -1] = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 1, knn_classes)

    return pointcloud


def distance_cutoff(pointcloud: np.ndarray, cutoff: float = 100.) -> np.ndarray:
    """
    points which are too far away are clipped away
    :param pointcloud: point cloud
    :param cutoff: maximal distance for considering a point
    :return: clipped point cloud
    """
    cut = np.sqrt(np.square(pointcloud[:, :3]).sum(axis=1)) < cutoff
    return pointcloud[cut]


def remove_car_shadows(pointcloud: np.ndarray, img_no: str, eps: float = 0.3) -> np.ndarray:
    """
    original vkitti dataset produces shadows of cars with wrong labels. Remove wrong class labels
    :param pointcloud: point cloud
    :param img_no: number of image within the sequence
    :param eps: bounding boxes are a bit wrong. Enlarge boundig boxes by eps
    :return: point cloud where wrong points have don't care class label
    """
    mot_data_img = mot_data[mot_data['frame'] == int(img_no)]

    # save indices of voxels belonging to vehicles such that we do not delete their class labels
    vehicle_idx = []

    for index, row in mot_data_img.iterrows():
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

    return pointcloud


def transform2worldspace(pointcloud: np.ndarray, img_no: str) -> np.ndarray:
    """
    transform from camera space into world space
    :param pointcloud: point cloud
    :param img_no: image number within sequence
    :return: point cloud in world space
    """
    worldspace = pointcloud.copy()
    worldspace = worldspace[:, (1,2,0,3,4,5,6)]
    worldspace[:, 0] = -worldspace[:,0]
    worldspace[:, 1] = -worldspace[:,1]
    worldspace[:, 2] = worldspace[:,2]

    mf = extgt[extgt['frame'] == int(img_no)]

    translation = np.array([float(mf['t1']), float(mf['t2']), float(mf['t3'])])

    matrix = np.array([[float(mf['r1,1']), float(mf['r1,2']), float(mf['r1,3'])],
                       [float(mf['r2,1']), float(mf['r2,2']), float(mf['r2,3'])],
                       [float(mf['r3,1']), float(mf['r3,2']), float(mf['r3,3'])]])

    # set the coordinate system origin to the center of the vehicle
    worldspace[:, :3] = worldspace[:, :3] - translation

    # rotate all points such that the vehicle is axis-aligned
    worldspace[:, :3] = np.einsum('ij, kj -> ki', np.transpose(matrix), worldspace[:, :3])

    return worldspace


def process_frame(image_path: str) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """
    fix given frame
    :param image_path: path to frame which should be fixed
    :return: fixed frame
    """
    seq_no = image_path.split('/')[-3]
    img_no = image_path.split('/')[-1].split('.')[0]

    depth_path = f"{depth_root}/{seq_no}/clone/{img_no}.png"
    semantic_path = f"{labels_root}/{seq_no}/clone/{img_no}.png"

    # BGR -> RGB
    rgb_map = cv2.imread(image_path)[:, :, (2, 1, 0)]

    # convert centimeters to meters
    depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 100.

    # semantic image
    semantic_map = cv2.imread(semantic_path)[:, :, (2, 1, 0)]
    label_map = np.apply_along_axis(lambda r: rgb2label[tuple(r)], 2, semantic_map)

    # backprojection to camera space
    x3 = (xv - center_x) / focal_x * depth_map
    y3 = (yv - center_y) / focal_y * depth_map

    erg = np.stack((depth_map, -x3, -y3), axis=-1).reshape((-1, 3))
    erg = np.hstack((erg, rgb_map.reshape(-1, 3), label_map.reshape(-1, 1)))

    # delete sky points
    erg = distance_cutoff(erg, g_cutoff)

    if g_is_v1:
        return None, erg, seq_no, img_no
    else:
        erg = remove_car_shadows(erg, img_no, g_bb_eps)
        worldspace = transform2worldspace(erg, img_no)
        return worldspace, erg, seq_no, img_no


def main(root_path: str, out_path: str, bb_eps: float, cutoff: float, knn_eps: float,
         knn_number: int, sequence: str, njobs: int, is_v1: bool):
    global rgb_root
    global depth_root
    global labels_root
    global mot_root
    global flow_root
    global extrinsic_root

    global g_bb_eps
    global g_cutoff

    global g_is_v1

    g_bb_eps = bb_eps
    g_cutoff = cutoff

    g_is_v1 = is_v1

    rgb_root = root_path + '/vkitti_1.3.1_rgb'
    depth_root = root_path + '/vkitti_1.3.1_depthgt'
    labels_root = root_path + '/vkitti_1.3.1_scenegt'
    mot_root = root_path + '/vkitti_1.3.1_motgt'
    flow_root = root_path + '/vkitti_1.3.1_flowgt'
    extrinsic_root = root_path + '/vkitti_1.3.1_extrinsicsgt'

    seq_no = sequence

    print(f"\nProcessing sequence {seq_no} ...\n")

    image_paths = sorted(glob.glob(f"{rgb_root}/{seq_no}/clone/*.png"))
    rgb2sem_path = f"{labels_root}/{seq_no}_clone_scenegt_rgb_encoding.txt"
    mot_path = f"{mot_root}/{seq_no}_clone.txt"
    extrinsic_path = f"{extrinsic_root}/{seq_no}_clone.txt"

    global extgt
    global encoding_df
    global mot_data

    extgt = pd.read_csv(extrinsic_path, sep=" ", index_col=False)
    encoding_df = pandas.read_csv(rgb2sem_path, sep=' ')
    mot_data = pd.read_csv(os.path.join(mot_path), sep=" ", index_col=False)

    for index, row in encoding_df.iterrows():
        rgb2label[(row['r'], row['g'], row['b'])] = sem2label[row['Category(:id)'].split(':')[0]]

    output = parallel_process(image_paths, process_frame, front_num=0, n_jobs=njobs)

    if g_is_v1:
        for i in range(len(output)):
            out_folder_str = f"{out_path}/{output[i][2]}"
            output_folder = Path(out_folder_str)
            output_folder.mkdir(parents=True, exist_ok=True)

            np.save(f"{out_folder_str}/{output[i][3]}", output[i][1])
    else:
        # find potential labels for don't care points
        overall_scene = np.vstack([output[i][0] for i in range(len(output))])
        overall_scene = knn_interpolation(overall_scene, knn_number, knn_eps)

        lower_bound = 0
        # transmit labels from worldspace pixels to cameraspace pixels
        for i in range(len(output)):
            upper_bound = lower_bound + output[i][0].shape[0]
            output[i][1][:, -1] = overall_scene[lower_bound:upper_bound, -1]
            lower_bound = upper_bound

            out_folder_str = f"{out_path}/{output[i][2]}"
            output_folder = Path(out_folder_str)
            output_folder.mkdir(parents=True, exist_ok=True)

            np.save(f"{out_folder_str}/{output[i][3]}", output[i][1])


def arg_check_positive_float(value):
    val = float(value)
    if val <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive float value" % value)
    return val


def arg_check_positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


def pretty_print_arguments(args):
    """
    return a nicely formatted list of passed arguments
    :param args: arguments passed to the program via terminal
    :return: None
    """
    longest_key = max([len(key) for key in vars(args)])

    print('Program was launched with the following arguments:')

    for key, item in vars(args).items():
        print("~ {0:{s}} \t {1}".format(key, item, s=longest_key))

    print('')
    # Wait a bit until program execution continues
    time.sleep(0.1)


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
                                                 "test and training set.\n"
                                                 "In order to further increase the ground truth quality, unkown point "
                                                 "labels are interpolated using label information "
                                                 "from different frames")

    parser.add_argument('--root_path', type=str, required=True,
                        help='path to the root directory of folders containing the original vkitti data')

    parser.add_argument('--out_path', type=str, required=True,
                        help='path to the directory where the fixed pointclouds should be saved')

    parser.add_argument('--bb_eps', const=0.3, default=0.3, type=arg_check_positive_float, nargs='?',
                        help='since the bounding boxes are not 100% accurate introduce an epsilon >= 0 to enlarge the '
                             'bounding boxes a bit')

    parser.add_argument('--cutoff', const=100., default=100., type=arg_check_positive_float, nargs='?',
                        help='distance cutoff: points farer away than specified cutoff distance will be discarded')

    parser.add_argument('--knn_eps', const=0.2, default=0.2, type=arg_check_positive_float, nargs='?',
                        help='just neighboring points with a distance of up to knn_eps '
                             'are considered for interpolation')

    parser.add_argument('--knn_number', const=1, default=1, type=arg_check_positive_int, nargs='?',
                        help='maximal number of neighbors to be considered')

    parser.add_argument('--sequence', required=True, type=str,
                        help='which sequence to process')

    parser.add_argument('--njobs', const=4, default=4, type=arg_check_positive_int, nargs='?',
                        help='number of cores used for processing')

    repair_parser = parser.add_mutually_exclusive_group(required=False)
    repair_parser.add_argument('--v1', dest='is_v1', action='store_true', help='create vkitti version 1')
    repair_parser.add_argument('--v2', dest='is_v1', action='store_false', help='create vkitti version 2')
    parser.set_defaults(is_v1=True)

    args = parser.parse_args()
    pretty_print_arguments(args)

    main(args.root_path, args.out_path, args.bb_eps, args.cutoff, args.knn_eps,
         args.knn_number, args.sequence, args.njobs, args.is_v1)
