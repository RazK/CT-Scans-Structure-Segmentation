import logging

import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure
from skimage.transform import resize

from common.nifti import get_filename_no_suffix


class ThresholdsPlotter:
    def __init__(self):
        self.i_min_list = list()
        self.num_components_list = list()
        self.last_figure = None

    def append(self, i_min, num_components):
        self.i_min_list.append(i_min)
        self.num_components_list.append(num_components)

    def save_last_figure(self, nifti_path):
        nifti_filename = get_filename_no_suffix(nifti_path)
        start = self.i_min_list[0]
        stop = self.i_min_list[-1]
        step = self.i_min_list[1] - self.i_min_list[0]
        output_path = f"out/{nifti_filename}_threshold_finder_{start}_{stop}_{step}.png"
        self.last_figure.savefig(output_path, dpi=200, format="png")
        logging.info(f"saved figure to '{output_path}'")

    def plot(self):
        logging.info(f"drawing figure x={self.i_min_list}, y={self.num_components_list}")
        plt.title("Skeleton segmentation threshold finder")
        plt.xlabel("Segmentation threshold i_min")
        plt.ylabel("# of 1-connectivity components")
        plt.scatter(self.i_min_list, self.num_components_list)
        plt.plot(self.i_min_list, self.num_components_list)
        self.last_figure = plt.gcf()
        plt.show()


def plot_3d(slices):
    smaller_slices = [resize(slice, (100, 100)) for slice in slices]

    # Get the shape of the image
    height, width = smaller_slices[0].shape

    # Create two 2D arrays containing the x-coordinates and y-coordinates of each cell
    X, Y = np.meshgrid(np.arange(width), np.arange(height))

    # create the figure
    fig = plt.figure()

    # show the 3D rotated projection
    ax = fig.add_subplot(111, projection='3d')

    for i, slice in enumerate(smaller_slices):
        Z = np.zeros_like(slice) + 10 * i
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=plt.cm.gray_r(slice / slice.max()), shade=False)


SHOW_LEVEL = 0


def show(img, level=1):
    if (level >= SHOW_LEVEL):
        plt.imshow(img, cmap="gray_r")
        plt.show()


def show_equalize(slice):
    slice_equalized = exposure.equalize_hist(slice)
    plt.subplot(1, 2, 1)
    plt.imshow(slice, cmap="gray_r")
    plt.title('Original')
    plt.subplot(1, 2, 2)
    plt.imshow(slice_equalized, cmap="gray_r")
    plt.title('Equalized')
    plt.show()
