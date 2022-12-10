import logging
import sys

import numpy as np

from common.nifti import save_aorta_segmentation, load_nifti_data, input_ct_path, input_l1_path, input_aorta_path, \
    build_aorta_segmentation_out_path
from common.plotting import show
from common.segmentation import get_first_and_last_nonzero_layers, get_key_layer_and_aorta_point_from_ct_and_l1, \
    create_aorta_mask_return_center, clip_to_ROI

logging.basicConfig(format="[%(module)s:%(funcName)s] %(message)s", stream=sys.stdout, level=logging.INFO)


def AortaSegmentation(ct_nifti_path, l1_seg_nifti_path):
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


# def evaluate_segmentation(clipped_gt_seg, clipped_est_seg):
#     """Calculate the DICE and VOD scores of two 3D numpy arrays.
#
#     Args:
#         clipped_gt_seg: A 3D numpy array containing the ground truth segmentation.
#         clipped_est_seg: A 3D numpy array containing the estimated segmentation.
#
#     Returns:
#         A tuple with the DICE score and the VOD score.
#     """
#
#     # Calculate the DICE score
#     intersection = np.sum(clipped_gt_seg * clipped_est_seg)
#     GT_count = np.sum(clipped_gt_seg)
#     est_count = np.sum(clipped_est_seg)
#     dice = 2 * intersection / (GT_count + est_count)
#
#     # Calculate the VOD score
#     GT_unique, GT_counts = np.unique(clipped_gt_seg, return_counts=True)
#     vod = 0
#     for val, count in zip(GT_unique, GT_counts):
#         # Ignore background pixels (assumed to have value 0)
#         if val == 0:
#             continue
#         GT_seg_val = (clipped_gt_seg == val)
#         est_seg_val = (clipped_est_seg == val)
#         intersection = np.sum(GT_seg_val * est_seg_val)
#         vod += intersection / count
#
#     logging.info("DICE score: %.4f", dice)
#     logging.info("VOD score: %.4f", vod)
#     return dice, vod
#

def evaluate_segmentation(gt_seg, est_seg):
    gt_seg, est_seg = clip_to_ROI(gt_seg, est_seg)
    size_GT = np.count_nonzero(gt_seg)
    size_EST = np.count_nonzero(est_seg)
    intersect = np.count_nonzero(np.logical_and(est_seg, gt_seg))
    dice = (2 * intersect) / (size_EST + size_GT)
    union = np.count_nonzero(np.logical_or(est_seg, gt_seg))
    vod = 1 - (intersect / union)
    return vod, dice


# def clip_gt_est_seg(gt_seg, est_seg):
#
#     # Convert the arrays to int type and ensure they have the same shape
#     GT_seg = GT_seg.astype(int)
#     est_seg = est_seg.astype(int)
#     print(f"GT_seg shape={GT_seg.shape}")
#     print(f"est_seg shape={est_seg.shape}")
#     assert GT_seg.shape == est_seg.shape, "Arrays must have the same shape"
#
#     # Clip GT_seg to the same ROI as est_seg
#     show(GT_seg[:, :, 270])
#     show(est_seg[:, :, 270])
#     GT_seg, est_seg = clip_to_ROI(GT_seg, est_seg)
#     show(GT_seg[:, :, 270])
#     show(est_seg[:, :, 270])
#

if __name__ == "__main__":
    logging.info("aorta_segmentation.py running...")
    for i in range(1, 4 + 1):
        # Perform Aorta segmentation
        # aorta_segmentation = AortaSegmentation(input_ct_path(i), input_l1_path(i))

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
