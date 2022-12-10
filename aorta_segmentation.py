import logging
import sys

import numpy as np

from common.nifti import save_aorta_segmentation, load_nifti_data, input_ct_path, input_l1_path, input_aorta_path, \
    build_aorta_segmentation_out_path
from common.segmentation import get_first_and_last_nonzero_layers, get_key_layer_and_aorta_point_from_ct_and_l1, \
    create_aorta_mask_return_center, clip_to_ROI

logging.basicConfig(format="[%(module)s:%(funcName)s] %(message)s", stream=sys.stdout, level=logging.INFO)


def AortaSegmentation(ct_nifti_path, l1_seg_nifti_path):
    """
    Perform a segmentation of the aorta based on the given L1 segmentation.
    Make some plots along the way and save the result.
    :param ct_nifti_path: str
    :param l1_seg_nifti_path: str
    :return: aorta_segmentation (after saving it)
    """
    ct, ct_file = load_nifti_data(ct_nifti_path, return_file=True)
    l1, l1_file = load_nifti_data(l1_seg_nifti_path, return_file=True)
    aorta_segmentation = np.zeros_like(ct)

    # Find the key L1 layer and starting aorta point in that layer
    bottom, top = get_first_and_last_nonzero_layers(l1)
    key_layer, aorta_center = get_key_layer_and_aorta_point_from_ct_and_l1(ct, l1)
    logging.info(f"L1 layers = [{bottom}, {top}]. Key layer = {key_layer}")

    # Iterate up from the key layer and segment the aorta
    for layer in range(key_layer, top, 1):
        logging.info(f"-------------------- Layer {layer} -------------------")
        aorta_mask, aorta_center = create_aorta_mask_return_center(ct, layer, aorta_center, show_level=1)
        aorta_segmentation[:, :, layer] = aorta_mask

    # Iterate up from the key layer and segment the aorta
    for layer in range(key_layer, bottom, -1):
        logging.info(f"-------------------- Layer {layer} -------------------")
        aorta_mask, aorta_center = create_aorta_mask_return_center(ct, layer, aorta_center, show_level=1)
        aorta_segmentation[:, :, layer] = aorta_mask

    # Finally, save the result segmentation
    save_aorta_segmentation(aorta_segmentation, ct_nifti_path, ct_file)

    return aorta_segmentation


def evaluate_segmentation(gt_seg, est_seg):
    """
    Return VOD, DICE scores for the given ground-truth (GT) and estimation.
    :param gt_seg: nparray
    :param est_seg: nparray
    :return: (vod, dice)
    """
    gt_seg, est_seg = clip_to_ROI(gt_seg, est_seg)
    size_GT = np.count_nonzero(gt_seg)
    size_EST = np.count_nonzero(est_seg)
    intersect = np.count_nonzero(np.logical_and(est_seg, gt_seg))
    dice = (2 * intersect) / (size_EST + size_GT)
    union = np.count_nonzero(np.logical_or(est_seg, gt_seg))
    vod = 1 - (intersect / union)
    return vod, dice


if __name__ == "__main__":
    logging.info("aorta_segmentation.py running...")
    for i in range(1, 4 + 1):
        # Perform Aorta segmentation
        aorta_segmentation = AortaSegmentation(input_ct_path(i), input_l1_path(i))

        # Evaluate the segmentation with the ground truth (GT)
        GT_path = input_aorta_path(i)
        est_path = build_aorta_segmentation_out_path(input_ct_path(i))
        GT_seg = load_nifti_data(GT_path)
        est_seg = load_nifti_data(est_path)
        logging.info(f"evaluating '{est_path}' vs '{GT_path}'...")
        VOD_result, DICE_result = evaluate_segmentation(GT_seg, est_seg)

        # Print the results
        print("VOD result:", VOD_result)
        print("DICE result:", DICE_result)
