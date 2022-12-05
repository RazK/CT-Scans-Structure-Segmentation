import logging

from matplotlib import pyplot as plt

from utils import get_filename_no_suffix


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
        step = self.i_min_list[1]-self.i_min_list[0]
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
