import logging

import numpy as np
from skimage.measure import label

from plotting import ThresholdsPlotter
from threshold_segmentation import threshold_segmentation, BONES_HU_MAX_VALUE


def find_best_segmentation(img_data,
                           min_threshold_range,
                           connectivity,
                           thresholds_plotter: ThresholdsPlotter):
    """
    Iterates over all i_min thresholds in the range of BONES_HU_MIN_RANGE to find an segmentation threshold.
    In each run, counts the number of connectivity components in the resulting segmentation with the current i_min.
    :param connectivity: TBD
    :param thresholds_plotter: TBD
    :param min_threshold_range: range(start, stop, step) to iterate when finding best i_min.
    :param img_data: nifty image data to do segmentation on.
    :return: (best_segmentation_data, best_i_min, threshold_finding_figure)
    """
    min_components_num = np.infty
    best_i_min = -1
    best_segmentation_data = None
    for i_min in min_threshold_range:
        segmentation_data = threshold_segmentation(img_data, i_min, BONES_HU_MAX_VALUE)
        labels, num = label(segmentation_data, connectivity=connectivity, return_num=True)
        logging.info(f"segmentation threshold [{i_min},{BONES_HU_MAX_VALUE}] has {num} 1-connectivity components")
        if num < min_components_num:
            min_components_num = num
            best_i_min = i_min
            best_segmentation_data = segmentation_data
        if thresholds_plotter:
            thresholds_plotter.append(i_min, num)
            thresholds_plotter.plot()
    logging.info(f"minimum number of 1-connectivity components: f({best_i_min})={min_components_num}")
    return best_segmentation_data, best_i_min
