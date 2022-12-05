import logging

import numpy as np

BONES_HU_MAX_VALUE = 1300
BONES_HU_MIN_RANGE_START = 150
BONES_HU_MIN_RANGE_END = 500
BONES_HU_MIN_RANGE_STEP = 14
BONES_HU_MIN_RANGE = range(BONES_HU_MIN_RANGE_START,
                           BONES_HU_MIN_RANGE_END + 1,  # include end
                           BONES_HU_MIN_RANGE_STEP)


def threshold_segmentation(img_data,
                           i_min,
                           i_max):
    """
    Return a threshold segmentation mask with 1s within, 0s out.
    :param img_data: image to perform threshold segmentation
    :param i_max: upper inclusion threshold
    :param i_min: lower inclusion threshold
    :return: threshold segmentation mask with 1s within, 0s out.
    """
    logging.info(f"performing segmentation based on threshold [{i_min},{i_max}]...")
    segmentation_data = np.zeros_like(img_data)
    within_threshold = np.logical_and(i_min < img_data, img_data < i_max)
    segmentation_data[within_threshold] = 1
    return segmentation_data.astype(bool)
