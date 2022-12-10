import logging
import sys

from common.nifti import load_nifti_data, save_bones_segmentation, input_ct_path
from common.plotting import ThresholdsPlotter
from common.segmentation import remove_all_but_largest_object_and_background, segment_by_threshold, \
    find_optimal_imin_and_segmentation

logging.basicConfig(format="[%(module)s:%(funcName)s] %(message)s", stream=sys.stdout, level=logging.INFO)

BONES_HU_MAX_VALUE = 1300
BONES_HU_MIN_RANGE_START = 150
BONES_HU_MIN_RANGE_END = 500
BONES_HU_MIN_RANGE_STEP = 14
BONES_HU_MIN_RANGE = range(BONES_HU_MIN_RANGE_START,
                           BONES_HU_MIN_RANGE_END + 1,  # include end
                           BONES_HU_MIN_RANGE_STEP)


def SegmentationByTH(nifti_path, Imin, Imax):
    logging.info(f"SegmentationByTH({nifti_path},{Imin},{Imax})...")
    nifti_data, nifti_file = load_nifti_data(nifti_path, return_file=True)
    bones_segmentation = segment_by_threshold(nifti_data, Imin, Imax)
    save_bones_segmentation(nifti_file,
                            nifti_path,
                            bones_segmentation.astype(int),
                            Imin,
                            Imax)


def SkeletonTHFinder(nifti_path, connectivity=1):
    nifti_data, nifti_file = load_nifti_data(nifti_path,
                                             return_file=True)
    thresholds_plotter = ThresholdsPlotter()
    i_min, segmentation_mask = find_optimal_imin_and_segmentation(nifti_data,
                                                                  BONES_HU_MIN_RANGE,
                                                                  BONES_HU_MAX_VALUE,
                                                                  connectivity,
                                                                  thresholds_plotter)
    thresholds_plotter.save_last_figure(nifti_path)
    segmentation_mask = remove_all_but_largest_object_and_background(segmentation_mask,
                                                                     connectivity)
    save_bones_segmentation(nifti_file,
                            nifti_path,
                            segmentation_mask.astype(int),
                            i_min,
                            BONES_HU_MIN_RANGE.stop,
                            post_processed=True)


if __name__ == "__main__":
    logging.info("bones_segmentation.py running...")
    for i in range(1,5+1):
        SegmentationByTH(input_ct_path(i), 240, 1300)
        SkeletonTHFinder(input_ct_path(i))
