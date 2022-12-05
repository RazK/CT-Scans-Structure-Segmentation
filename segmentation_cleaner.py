import logging

import numpy as np
import skimage
from skimage.measure import label


def post_process_segmentation(segmentation_data, connectivity):
    """
    Performs post-processing (morphological operations – clean out single pixels, close holes, etc.) until left with a
    single connectivity component.
    :param connectivity: TBD
    :param segmentation_data: data to post-process
    :return: post_processed_data - data after cleaning single pixels, holes, etc. - a single connectivity componenet.
    """
    logging.info("post-processing segmentation data...")
    logging.info("removing all objects except the largest...")
    segmentation_data = remove_all_small_objects(segmentation_data, connectivity)
    logging.info("filling all holes except the background...")
    segmentation_data = remove_all_small_holes(segmentation_data, connectivity)
    labels, num = label(segmentation_data, connectivity=connectivity, return_num=True)
    logging.info(f"{num} connected-components left...")
    logging.info("cleanup completed.")
    return segmentation_data


def remove_all_small_objects(segmentation_data, connectivity):
    labels, num = label(segmentation_data, connectivity=connectivity, return_num=True)
    logging.info(f"found {num} connected-components in the segmentation...")
    components_sizes = sorted([component["area"] for component in skimage.measure.regionprops(labels)],
                              reverse=True)
    logging.info(f"connected-components sizes (descending): {components_sizes}")
    cleanup_size = components_sizes[1] + 1
    logging.info(f"cleaning up everything smaller than {cleanup_size}...")
    segmentation_data = skimage.morphology.remove_small_objects(segmentation_data, cleanup_size)
    return segmentation_data


def remove_all_small_holes(segmentation_data, connectivity):
    return np.invert(remove_all_small_objects(np.invert(segmentation_data), connectivity))
