import logging

import nibabel as nib
from nibabel import Nifti1Image

from utils import get_filename_no_suffix


def save_segmentation_nifti(segmentation_data, nifti_file, output_path):
    """
    Save the segmentation mask to a nifti file at output_path.
    :param segmentation_data: an np.array with 1s where True, 0s where False.
    :param nifti_file: the original nifti file for which this segmentation belongs
    :param output_path: path to write segmentation nifti file.
    """
    new_nifti = nib.Nifti1Image(segmentation_data, nifti_file.affine)
    logging.info(f"writing a segmentation mask to '{output_path}'")
    nib.save(new_nifti, output_path)


def load_nifti_file(nifti_path: str):
    """
    Read the data from the given nifti_path.
    :param nifti_path: path to nifti file.
    :return: np.array with nifti file data, or FAILURE on error.
    """
    logging.info(f"loading '{nifti_path}'...")
    try:
        nifti_file: Nifti1Image = nib.load(f'{nifti_path}')
    except Exception as e:
        logging.error(f"failed loading file '{nifti_path}'")
        raise e
    return nifti_file


def get_segmentation_output_path(nifti_path, i_max, i_min, post_process=False):
    """
    Return the output path where the nifti segmentation will be saved.
    :param post_process:
    :param i_max: upper threshold.
    :param i_min: lower threshold.
    :param nifti_path: path to input nifti file.
    :return: path to save the segmentation nifty file.
    """
    nifti_filename = get_filename_no_suffix(nifti_path)
    output_path = f'out/{nifti_filename}_seg_{i_min}_{i_max}{f"_post_processed" if post_process else ""}.nii.gz'
    return output_path


def load_nifti_data(nifti_path):
    nifti_file = load_nifti_file(nifti_path)
    # getting a pointer to the data
    logging.info(f"getting the data...")
    img_data = nifti_file.get_fdata()
    return img_data, nifti_file
