import os
import re
from sys import exit
from time import localtime
from typing import List
from warnings import warn

import numpy as np

# per OpenCV docs
SUPPORTED_IMGS = [
    '.bmp', '.dib', '.jpeg', '.jpg', '.jpe', '.jp2', '.png', '.webp', '.pbm',
    '.pgm', '.ppm', '.pxm', '.pnm', '.pfm', '.ras', '.tiff', '.tif', '.exr',
    '.hdr', '.pic'
    ]


def transpose_list_of_lists(list_of_lists: List[list]) -> list:
    """ Finds the biggest list within within a list.
        Coverts sub-lists to np.array.
        Pads shorter lists with `0.0` to make same shape as biggest list.
        Appends padded list to empty list.
        Flattens the list of arrays, uses ('F') in flatten to transpose.
        Removes the `0.0` padding that is left over
        RETURNS 1D list of the transposed list of lists
    """
    # Find the biggest list within all lists of paths
    biggest_list = list(filter(lambda lst: len(lst) == max(
        [len(sub_lst) for sub_lst in list_of_lists]), list_of_lists))
    biggest_len = len(biggest_list[0])

    # Convert to np array - pad shorter lists than biggest_list with zeros - append to empty list
    all_np = []
    for contents_list in list_of_lists:
        np_lst = np.array(contents_list)
        padded = np.append(np_lst, np.zeros(biggest_len - np_lst.shape[0]))
        all_np.append(padded)

    # Flatten list of np arrays to single list while mixing up the contents ('F')
    # NOTE Will still have '0.0' for padded values
    dirty_list = np.array(all_np).flatten('F').tolist()

    # Filter out the '0.0'
    new_all_paths = list(filter(lambda x: not x.replace('.', '1').isdigit(), dirty_list))
    return new_all_paths


def save_date_str():
    save_time = localtime()
    save_date = ':'.join([str(save_time.tm_mon), str(save_time.tm_mday)])
    save_time = ':'.join([str(save_time.tm_hour), str(save_time.tm_min)])
    save_str = '::'.join([save_date, save_time])
    return save_str


def does_it_float(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def replace_goofy_slashes(path: str):
    return re.sub(r'(\\+|\/+)', r'/', path)


def ask_for_input(path_checked: str):  # NOT USED
    response = input("Begin checking next path?\tY/N\t").lower()

    if response in ["n", "no"]:
        return False

    if response in ['y', 'yes']:
        return True

    print("Bad input...")
    return ask_for_input(path_checked)


def dir_length_warning_input():  # NOT USED
    warn('\n\nWARNING:\tAmount of files to be compared is more than 100!\n', stacklevel=3)

    resp = input("Continue?\tY/N\t").lower()
    if resp in ['n', 'no']:
        return exit()
    if resp in ['y', 'yes']:
        return None

    print("Bad input...")
    dir_length_warning_input()


def get_supported_images(checking_dir):  # NOT USED
    r_ptrn = re.compile('\.\w{3,4}$')
    dir_contents = next(os.walk(checking_dir))

    supported_files = [os.path.join(dir_contents[0], i) for i in dir_contents[2]
                       if re.search(r_ptrn, i) and re.search(r_ptrn, i).group() in SUPPORTED_IMGS]

    return supported_files
