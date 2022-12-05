import logging

from numpy import uint16

from nifti_wrapper import load_nifti_data, save_segmentation_nifti
from plotting import ThresholdsPlotter
from segmentation_cleaner import post_process_segmentation
from segmentation_finder import find_best_segmentation
from threshold_segmentation import BONES_HU_MAX_VALUE, threshold_segmentation, BONES_HU_MIN_RANGE


class SegmentationManager:
    def __init__(self,
                 nifti_path,
                 connectivity):

        logging.info(f"running on nifti_path='{nifti_path}'...")
        self.nifti_path = nifti_path
        self.connectivity = connectivity
        self.img_data, self.nifti_file = load_nifti_data(nifti_path)

    def run(self,
            i_min: uint16 = None,
            i_max: uint16 = BONES_HU_MAX_VALUE,
            post_process: bool = False):
        """
        Generates a segmentation NIFTI file of the same dimensions as the input file, with a binary segmentation
        where 1 stands for voxels within [i_min, i_max] and 0 otherwise.
        This segmentation NIFTI file is saved under the name <nifti_file>_seg_<Imin>_<Imax>.nii.gz.
        :param i_min: minimal threshold. If None - find the best i_min automatically.
        :param i_max: maximal threshold.
        :param post_process: True if post-processing is wanted, False otherwise.
        """
        if i_min:
            logging.info(f"performing segmentation using i_min={i_min}...")
            segmentation_data = threshold_segmentation(self.img_data, i_min, i_max)
        else:  # find best i_min
            logging.info(f"finding best i_min in {BONES_HU_MIN_RANGE} and performing segmentation...")
            thresholds_plotter = ThresholdsPlotter()
            segmentation_data, i_min = find_best_segmentation(self.img_data,
                                                              BONES_HU_MIN_RANGE,
                                                              self.connectivity,
                                                              thresholds_plotter)
            thresholds_plotter.save_last_figure(self.nifti_path)

        if post_process:
            segmentation_data = post_process_segmentation(segmentation_data, self.connectivity)

        # saving the result
        save_segmentation_nifti(segmentation_data.astype(int),
                                self.nifti_file,
                                self.nifti_path,
                                i_min,
                                i_max,
                                post_process)
        logging.info("done.")
