import logging
import os

import nibabel as nib
from nibabel import Nifti1Image


def input_ct_path(i):
    return f"data/Case{i}_CT.nii.gz"


def input_l1_path(i):
    return f"data/Case{i}_L1.nii.gz"


def input_aorta_path(i):
    return f"data/Case{i}_Aorta.nii.gz"


def get_filename_no_suffix(path):
    """
    Extract <filename> from '/path/to/<filename>.ex.ten.si.on'
    :param path: path leading to the filename
    :return: just the filename, no path and no extensions.
    """
    nifti_filename = os.path.basename(path).split('.')[0]
    return nifti_filename


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


def load_nifti_data(nifti_path: str, return_file=False):
    nifti_file = load_nifti_file(nifti_path)
    logging.info(f"getting the data...")
    img_data = nifti_file.get_fdata()
    if return_file:
        return img_data, nifti_file
    return img_data


def save_aorta_segmentation(segmentation_mask, input_ct_path, original_ct_nifti):
    output_path = build_aorta_segmentation_out_path(input_ct_path)
    save_segmentation_nifti(segmentation_mask, output_path, original_ct_nifti)


def save_bones_segmentation(input_nifti_file, input_nifti_path, segmentation_mask, i_min, i_max, post_processed=False):
    """
    Save the segmentation mask to a nifti file at output_path.
    :param segmentation_mask: np.array with 1s where True, 0s where False.
    :param input_nifti_file: the original nifti file for which this segmentation belongs
    :param output_path: path to write segmentation nifti file.
    """
    output_path = build_bones_segmentation_out_path(input_nifti_path=input_nifti_path,
                                                    i_min=i_min,
                                                    i_max=i_max,
                                                    post_processed=post_processed)
    save_segmentation_nifti(segmentation_mask, output_path, input_nifti_file)


def build_bones_segmentation_out_path(input_nifti_path, i_min, i_max, post_processed=False):
    """
    Return the output path where the nifti segmentation will be saved.
    :param input_nifti_path:
    :param i_max: upper threshold.
    :param i_min: lower threshold.
    :param post_processed: True/False.
    :return: path to save the segmentation nifti file.
    """
    nifti_filename = get_filename_no_suffix(input_nifti_path)
    output_path = f'out/{nifti_filename}_Bones_Segmentation_{i_min}_{i_max}{f"_post_processed" if post_processed else ""}.nii.gz'
    return output_path


def build_aorta_segmentation_out_path(input_ct_path):
    """
    Return the output path where the nifti segmentation will be saved.
    :param input_nifti_path:
    :param i_max: upper threshold.
    :param i_min: lower threshold.
    :param post_process: True/False.
    :return: path to save the segmentation nifti file.
    """
    ct_name = get_filename_no_suffix(input_ct_path)
    output_path = f'out/{ct_name}_Aorta_Segmentation.nii.gz'
    return output_path


def save_segmentation_nifti(segmentation, out_path, origianl_nifti_file):
    new_nifti = nib.Nifti1Image(segmentation, origianl_nifti_file.affine)
    nib.save(new_nifti, out_path)
    logging.info(f"writing a segmentation mask to '{out_path}'")
