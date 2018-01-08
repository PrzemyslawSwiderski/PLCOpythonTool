import glob
import os

from os.path import isfile, join


def ensure_dir(dir_name):
    """
    Function checks if there is a directory with specified dir_name
    if not, it is not existing, dir structure is created

    :param dir_name:
    """
    directory = os.path.dirname(dir_name)
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_file_names_by_ext(path, extension='pkl'):
    """
    Method returns list of files with specified extension
    in directory

    :param path: directory
    :param extension: file_extension
    """
    files_grabbed = []
    files_grabbed.extend(glob.glob(path + os.sep + "*." + extension))
    return files_grabbed
