import logging

from matplotlib import pyplot as plt

from utils import get_filename_no_suffix


def save_threshold_finder_fig(nifti_path, fig, imin_range):
    nifti_filename = get_filename_no_suffix(nifti_path)
    output_path = f"out/{nifti_filename}_threshold_finder_{imin_range.start}_{imin_range.stop}_{imin_range.step}.png"
    fig.savefig(output_path, dpi=200, format="png")
    logging.info(f"saved figure to '{output_path}'")


def plot_thresholds(x, y):
    logging.info(f"updating figure x={x}, y={y}")
    plt.title("Skeleton segmentation threshold finder")
    plt.xlabel("Segmentation threshold i_min")
    plt.ylabel("# of 1-connectivity components")
    plt.scatter(x, y)
    plt.plot(x, y)
    fig = plt.gcf()
    plt.show()
    return fig
