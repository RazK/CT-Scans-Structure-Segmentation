# nibabel is the required library for nifti files.
import logging
import sys

import numpy as np
import skimage
from numpy import uint16
from skimage.measure import label

from nifti_wrapper import save_segmentation_nifti, get_segmentation_output_path, load_nifti_data
from plotting import save_threshold_finder_fig, plot_thresholds

logging.basicConfig(format="[%(funcName)s] %(message)s", stream=sys.stdout, level=logging.DEBUG)

SUCCESS = 1
FAILURE = 0

DEMO_INPUT_FILE_1 = "Case1_CT.nii.gz"
DEMO_INPUT_FILE_2 = "Case2_CT.nii.gz"
DEMO_INPUT_FILE_3 = "Case3_CT.nii.gz"
DEMO_OUTPUT_FILE = "TH.nii.gz"

BONES_HU_MAX_VALUE = 1300
BONES_HU_MIN_RANGE_START = 150
BONES_HU_MIN_RANGE_END = 500
BONES_HU_MIN_RANGE_STEP = 14
BONES_HU_MIN_RANGE = range(BONES_HU_MIN_RANGE_START,
                           BONES_HU_MIN_RANGE_END + 1,  # include end
                           BONES_HU_MIN_RANGE_STEP)
CONNECTIVITY = 1


def run(nifti_path: str, i_min: uint16 = None, i_max: uint16 = BONES_HU_MAX_VALUE, post_process: bool = False):
    """
    Generates a segmentation NIFTI file of the same dimensions as the input file, with a binary segmentation
    where 1 stands for voxels within [i_min, i_max] and 0 otherwise.
    This segmentation NIFTI file is saved under the name <nifti_file>_seg_<Imin>_<Imax>.nii.gz.
    :param nifti_path: path of input nifti file.
    :param i_min: minimal threshold. If None - find the best i_min automatically.
    :param i_max: maximal threshold.
    :param post_process: True if post-processing is wanted, False otherwise.
    """
    logging.info(f"running on nifti_path={nifti_path}...")
    # load the nifti data
    img_data, nifti_file = load_nifti_data(nifti_path)

    # do the segmentation
    if i_min:
        logging.info(f"performing segmentation using i_min={i_min}...")
        segmentation_data = get_threshold_segmentation(img_data, i_min, i_max)
    else:  # find best i_min
        logging.info(f"finding best i_min and performing segmentation...")
        segmentation_data, i_min, threshold_finder_fig = find_best_segmentation(img_data, BONES_HU_MIN_RANGE)
        save_threshold_finder_fig(nifti_path, threshold_finder_fig, BONES_HU_MIN_RANGE)

    if post_process:
        segmentation_data = post_process_segmentation(segmentation_data)

    # saving the result
    output_path = get_segmentation_output_path(nifti_path, i_max, i_min, post_process)
    save_segmentation_nifti(segmentation_data.astype(int), nifti_file, output_path)
    logging.info("done.")


def get_threshold_segmentation(img_data, i_min, i_max):
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
    return segmentation_data.astype(bool)


def find_best_segmentation(img_data, min_threshold_range):
    """
    Iterates over all i_min thresholds in the range of BONES_HU_MIN_RANGE to find an segmentation threshold.
    In each run, counts the number of connectivity components in the resulting segmentation with the current i_min.
    :param min_threshold_range: range(start, stop, step) to iterate when finding best i_min.
    :param img_data: nifty image data to do segmentation on.
    :return: (best_segmentation_data, best_i_min, threshold_finding_figure)
    """
    i_min_list, num_components_list = list(), list()
    min_components_num = np.infty
    best_i_min = -1
    best_segmentation_data = None
    last_fig = None
    for i_min in min_threshold_range:
        segmentation_data = get_threshold_segmentation(img_data, i_min, BONES_HU_MAX_VALUE)
        labels, num = label(segmentation_data, connectivity=CONNECTIVITY, return_num=True)
        logging.info(f"segmentation threshold [{i_min},{BONES_HU_MAX_VALUE}] has {num} 1-connectivity components")
        i_min_list.append(i_min)
        num_components_list.append(num)
        if num < min_components_num:
            min_components_num = num
            best_i_min = i_min
            best_segmentation_data = segmentation_data
        last_fig = plot_thresholds(i_min_list, num_components_list)
    logging.info(f"minimum number of 1-connectivity components: f({best_i_min})={min_components_num}")
    return best_segmentation_data, best_i_min, last_fig


def post_process_segmentation(segmentation_data):
    """
    Performs post-processing (morphological operations – clean out single pixels, close holes, etc.) until left with a
    single connectivity component.
    :param segmentation_data: data to post-process
    :return: post_processed_data - data after cleaning single pixels, holes, etc. - a single connectivity componenet.
    """
    logging.info("post-processing segmentation data...")
    logging.info("removing all objects except the largest...")
    segmentation_data = remove_all_small_objects(segmentation_data)
    logging.info("filling all holes except the background...")
    segmentation_data = remove_all_small_holes(segmentation_data)
    labels, num = label(segmentation_data, connectivity=CONNECTIVITY, return_num=True)
    logging.info(f"{num} connected-components left...")
    logging.info("cleanup completed.")
    return segmentation_data


def remove_all_small_objects(segmentation_data):
    labels, num = label(segmentation_data, connectivity=CONNECTIVITY, return_num=True)
    logging.info(f"found {num} connected-components in the segmentation...")
    components_sizes = sorted([component["area"] for component in skimage.measure.regionprops(labels)], reverse=True)
    logging.info(f"connected-components sizes (descending): {components_sizes}")
    cleanup_size = components_sizes[1]+1
    logging.info(f"cleaning up everything smaller than {cleanup_size}...")
    segmentation_data = skimage.morphology.remove_small_objects(segmentation_data, cleanup_size)
    return segmentation_data

def remove_all_small_holes(segmentation_data):
    return np.invert(remove_all_small_objects(np.invert(segmentation_data)))


if __name__ == "__main__":
    run(f'data/{DEMO_INPUT_FILE_3}', post_process=True)
