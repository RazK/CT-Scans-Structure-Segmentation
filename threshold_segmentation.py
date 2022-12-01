# nibabel is the required library for nifti files.
import logging
import os.path
import sys

logging.basicConfig(format="[%(funcName)s] %(message)s", stream=sys.stdout, level=logging.DEBUG)

import nibabel as nib
import numpy as np
from nibabel import Nifti1Image
from numpy import uint16

SUCCESS = 1
FAILURE = 0

DEMO_INPUT_FILE = "Case1_CT.nii.gz"
DEMO_OUTPUT_FILE = "TH.nii.gz"

BONES_MAX_HU_VALUE = 1300


def segmentation_by_threshold(nifti_path: str, i_min: uint16, i_max: uint16):
    """
    Generates a segmentation NIFTI file of the same dimensions as the input file, with a binary segmentation
    where 1 stands for voxels within [i_min, i_max] and 0 otherwise.
    This segmentation NIFTI file is saved under the name <nifti_file>_seg_<Imin>_<Imax>.nii.gz.
    :param nifti_path: path of input nifti file.
    :param i_min: minimal threshold.
    :param i_max: maximal threshold.
    :return: 1 if successful, 0 otherwise.
    """
    # loading the nifti file
    nifti_file = load_nifti_file(nifti_path)
    if nifti_file == FAILURE:
        return FAILURE

    # getting a pointer to the data
    logging.info(f"getting the data...")
    img_data = nifti_file.get_fdata()

    # performing segmentation
    segmentation_data = get_threshold_segmentation(img_data, i_max, i_min)

    # saving the result
    output_path = get_segmentation_output_path(i_max, i_min, nifti_path)
    save_segmentation_nifti(segmentation_data, nifti_file, output_path)
    logging.info("done.")


def get_segmentation_output_path(i_max, i_min, nifti_path):
    """
    Return the output path where the nifti segmentation will be saved.
    :param i_max:
    :param i_min:
    :param nifti_path:
    :return:
    """
    nifti_filename = os.path.basename(nifti_path)
    output_path = f'out/{nifti_filename}_seg_{i_min}_{i_max}.nii.gz'
    return output_path


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
        return FAILURE
    return nifti_file


def get_threshold_segmentation(img_data, i_max, i_min):
    """
    Return a threshold segmentation mask with 1s within, 0s out.
    :param img_data: image to perform threshold segementation
    :param i_max: upper inclusion threshold
    :param i_min: lower inclusion threshold
    :return: threshold segmentation mask with 1s within, 0s out.
    """
    logging.info(f"performing segmentation based on threshold [{i_min},{i_max}]...")
    segmentation_data = np.zeros_like(img_data)
    within_threshold = np.logical_and(i_min < img_data, img_data < i_max)
    segmentation_data[within_threshold] = 1
    return segmentation_data


def skeleton_threshold_finder(nifti_filename: str):
    """
    Iterates over 25 candidate Imin thresholds in the range of [150,500] (with intervals of 14). In each run,
    uses segmentation_th to count the number of connectivity components in the resulting segmentation with the
    current Imin. Plots the results – number of connectivity components per Imin. Chooses Imin which is the first or
    second minima in the plot. Also, make sure to include that graph in your report. Performs post-processing (
    morphological operations – clean out single pixels, close holes, etc.) until you are left with a single
    connectivity component. Finally, this function should save a segmentation NIFTI file called
    “<nifti_file>_SkeletonSegmentation.nii.gz” and return the Imin used for that.
    :param nifti_file:
    :return:
    """
    pass


def demo():
    # loads the nifti files
    nifti_file = nib.load(f'data/{DEMO_INPUT_FILE}')
    # getting a pointer to the data
    img_data = nifti_file.get_fdata()
    # returning a boolean matrix with True in place where the intensity level were below 500
    low_values_flags = img_data < 500
    # returning a boolean matrix with True in place where the intensity level were above or equal to 500
    high_values_flags = img_data >= 500

    # Turning boolean to {0,1}
    img_data[low_values_flags] = 0
    img_data[high_values_flags] = 1

    # creating new NiftiImage
    new_nifti = nib.Nifti1Image(img_data, nifti_file.affine)
    # saving the nifti file.
    nib.save(new_nifti, f'out/{DEMO_OUTPUT_FILE}')


if __name__ == "__main__":
    # demo()
    segmentation_by_threshold(f'data/{DEMO_INPUT_FILE}', 500, 800)
