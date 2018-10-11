import glob
from pathlib import Path
import argparse
from argparse import RawTextHelpFormatter
import time
from shutil import copy
from tqdm import tqdm


def main(root_path: str, out_path: str):
    seq2area = {
        '0001': [
            ('Area_1', [0, 12, 24, 36, 48, 60, 72, 85, 97, 109, 121, 133, 145, 157, 170]),
            ('Area_2', [230, 243, 257, 270, 284, 297, 311, 325, 338, 352, 365, 379, 392, 406, 420])
            ],
        '0002': [
            ('Area_3', [0, 15, 31, 47, 63, 79, 95, 111, 127, 143, 159, 175, 191, 207, 223])
            ],
        '0006': [],  # Not used
        '0018': [
            ('Area_4', [30, 52, 74, 96, 118, 140, 162, 184, 206, 228, 250, 272, 294, 316, 338])
            ],
        '0020': [
            ('Area_5', [80, 106, 132, 158, 184, 210, 236, 262, 288, 314, 340, 366, 392, 418, 444]),
            ('Area_6', [500, 521, 542, 564, 585, 607, 628, 650, 671, 692, 714, 735, 757, 778, 800])
            ],
    }

    image_paths = sorted(glob.glob(f"{root_path}/*/*.npy"))

    for image_path in tqdm(image_paths, desc='Creating sets'):
        seq_no = image_path.split('/')[-2]
        img_no = int(image_path.split('/')[-1].split('.')[-2])

        for area in seq2area[seq_no]:
            if img_no in area[1]:
                out_folder_str = f"{out_path}/{area[0]}"
                output_folder = Path(out_folder_str)
                output_folder.mkdir(parents=True, exist_ok=True)
                copy(image_path, out_folder_str)
                break

    # Wait a bit until program execution continues
    time.sleep(0.1)
    print('Successfully terminated')


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
                                     description="Create set splits for evaluating as proposed in our workshop paper")

    parser.add_argument('--root_path', type=str, required=True,
                        help='path to the root directory of folders containing the vkitti npy data')

    parser.add_argument('--out_path', type=str, required=True,
                        help='path to the directory where the sets should be saved')

    args = parser.parse_args()
    pretty_print_arguments(args)

    main(args.root_path, args.out_path)
