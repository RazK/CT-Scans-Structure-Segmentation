import os


def get_filename_no_suffix(path):
    """
    Extract <filename> from '/path/to/<filename>.ex.ten.si.on'
    :param path: path leading to the filename
    :return: just the filename, no path and no extensions.
    """
    nifti_filename = os.path.basename(path).split('.')[0]
    return nifti_filename
