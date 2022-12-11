import logging

import numpy as np
import skimage
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
from skimage.feature import canny
from skimage.filters.thresholding import threshold_otsu
from skimage.measure import label
from skimage.transform import hough_circle, hough_circle_peaks

from common.plotting import ThresholdsPlotter, show

SHOW_LEVEL = 3


def segment_by_threshold(img_data,
                         i_min,
                         i_max):
    """
    keep only values within i_min, i_max
    :param img_data: nparray
    :param i_min: int
    :param i_max: int
    :return: thershed image
    """
    logging.info(f"performing segmentation based on threshold [{i_min},{i_max}]...")
    segmentation_mask = np.zeros_like(img_data)
    within_threshold = np.logical_and(i_min < img_data, img_data < i_max)
    segmentation_mask[within_threshold] = 1
    return segmentation_mask.astype(bool)


def find_optimal_imin_and_segmentation(img_data,
                                       i_min_range,
                                       i_max,
                                       connectivity,
                                       thresholds_plotter: ThresholdsPlotter = None):
    """
    find the best i_min to segment this image and return both
    :param img_data: nparray
    :param i_min_range: range
    :param i_max: int
    :param connectivity: int
    :param thresholds_plotter: ThresholdPlotter
    :return: best_i_min, best_segmentation_mask
    """
    min_components_num = np.infty
    best_i_min = -1
    best_segmentation_mask = None
    for i_min in i_min_range:
        segmentation_mask = segment_by_threshold(img_data, i_min, i_max)
        labels, num = label(segmentation_mask, connectivity=connectivity, return_num=True)
        logging.info(f"segmentation threshold [{i_min},{i_max}] has {num} 1-connectivity components")
        if num < min_components_num:
            min_components_num = num
            best_i_min = i_min
            best_segmentation_mask = segmentation_mask
        if thresholds_plotter:
            thresholds_plotter.append(i_min, num)
            thresholds_plotter.plot()
    logging.info(f"minimum number of 1-connectivity components: f({best_i_min})={min_components_num}")
    return best_i_min, best_segmentation_mask


def remove_all_but_largest_object_and_background(segmentation_mask, connectivity):
    """
    function name is self explenatory...
    :param segmentation_mask: nparray
    :param connectivity: int
    :return: segmentation mask with only 1 connected component and no holes
    """
    logging.info("post-processing segmentation data...")
    logging.info("removing all objects except the largest...")
    segmentation_mask = remove_all_objects_but_largest(segmentation_mask, connectivity)
    logging.info("filling all holes except the background...")
    segmentation_mask = remove_all_holes_but_background(segmentation_mask, connectivity)
    labels, num = label(segmentation_mask, connectivity=connectivity, return_num=True)
    logging.info(f"{num} connected-components left...")
    logging.info("cleanup completed.")
    return segmentation_mask


def remove_all_objects_but_largest(segmentation_mask, connectivity):
    """
    function name is self explenatory...
    :param segmentation_mask: nparray
    :param connectivity: int
    :return: segmentation mask with only 1 connected
    """
    labels, num = label(segmentation_mask, connectivity=connectivity, return_num=True)
    logging.info(f"found {num} connected-components in the segmentation...")
    components_sizes = sorted([component["area"] for component in skimage.measure.regionprops(labels)],
                              reverse=True)
    logging.info(f"connected-components sizes (descending): {components_sizes}")
    if len(components_sizes) > 1:
        cleanup_size = components_sizes[1] + 1
        logging.info(f"cleaning up everything smaller than {cleanup_size}...")
        segmentation_mask = skimage.morphology.remove_small_objects(segmentation_mask, cleanup_size)
    return segmentation_mask


def remove_all_holes_but_background(segmentation_mask, connectivity):
    """
    function name is self explenatory...
    :param segmentation_mask: nparray
    :param connectivity: int
    :return: segmentation mask with no holes
    """
    return np.invert(remove_all_objects_but_largest(np.invert(segmentation_mask), connectivity))


def connected_components(gray_image, sigma=3.0, t=0.5, connectivity=2):
    """
    return a list of labeled connected components after thresholding the image
    :param gray_image: nparray
    :param sigma: used to threshold
    :param t: no idea what that is
    :param connectivity: int
    :return: labeled_image, count
    """
    # denoise the image with a Gaussian filter
    blurred_image = skimage.filters.gaussian(gray_image, sigma=sigma)
    # mask the image according to threshold
    binary_mask = blurred_image > t
    # perform connected component analysis
    labeled_image, count = skimage.measure.label(binary_mask, connectivity=connectivity, return_num=True)
    return labeled_image, count


def get_first_and_last_nonzero_layers(img_data):
    """
    tell me which layers contain any data (assuming all in between is valid)
    :param img_data: nparray
    :return: (first, last)
    """
    logging.info(f"finding ROI for image of shape {img_data.shape}...")

    # Find the first non-zero frame in the z-axis
    first_nonzero = np.where(img_data.any(axis=(0, 1)))[0][0]
    logging.info(f'first non-zero frame: {first_nonzero}')

    # Find the last non-zero frame in the z-axis
    last_nonzero = np.where(img_data.any(axis=(0, 1)))[0][-1]
    logging.info(f'last non-zero frame:{last_nonzero}')

    # return (first_nonzero, last_nonzero)
    return (first_nonzero, last_nonzero)


def get_key_layer_and_aorta_point_from_ct_and_l1(ct, l1):
    """
    that one is tricky
    :param ct: nparray
    :param l1: nparray
    :return: sleep depraved
    """
    # Find the largest slice of the L1 segmentation
    key_layer = find_largest_slice_layer(l1)

    # Extract key layers of CT and Aorta (and normalize)
    key_ct = ct[:, :, key_layer]
    key_l1 = l1[:, :, key_layer]

    # Extract key points of L1
    key_l1_smooth = gaussian_filter(key_l1, sigma=10)
    threshold = threshold_otsu(key_l1_smooth)
    key_l1_thresh = key_l1_smooth > threshold * 2
    bounding_points = get_bounding_points(key_l1_thresh)
    best_point = bounding_points[1]
    plot_largest_bounding_box(bounding_points, key_ct, key_l1_thresh)  ########
    print("bounding_points")

    # Correct the point with a fixed offset
    off_x = -10
    off_y = -5
    key_point = (best_point[0] + off_x, best_point[1] + off_y)
    cx = key_point[0]
    cy = key_point[1]
    key_point_preview = np.zeros_like(key_ct)
    key_point_preview[cy - 3:cy + 3, cx - 3:cx + 3] = 1
    show(key_point_preview)
    print(f"key_point: {key_point}")

    return key_layer, key_point


def find_largest_slice_layer(segmentation):
    """
    find index of layer where the slice area is largest
    :param segmentation: nparray
    :return: index of layer where the slice area is largest
    """
    # Calculate the area of each layer in the mask
    slice_areas = np.sum(segmentation, axis=(0, 1))

    # Find the layer with the largest area
    largest_layer = np.argmax(slice_areas)
    print(f"largest slice found at layer: {largest_layer}")

    # Extract the layer with the largest area from the mask
    return largest_layer


def get_bounding_points(key_l1):
    """
    get 4 points around l1
    :param key_l1: nparray
    :return: 4 points
    """
    points = np.transpose(np.where(key_l1))
    bounding_points = maximum_bounding_rectangle(points)
    bounding_points = [(int(p[1]), int(p[0])) for p in bounding_points]
    return bounding_points


def maximum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    pi2 = np.pi / 2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points) - 1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles - pi2),
        np.cos(angles + pi2),
        np.cos(angles)]).T
    #     rotations = np.vstack([
    #         np.cos(angles),
    #         -np.sin(angles),
    #         np.sin(angles),
    #         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    # best_idx = np.argmin(areas)
    best_idx = np.argmax(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


def plot_largest_bounding_box(bounding_points, key_ct, key_l1):
    """
    just for funz
    :param bounding_points: lst of tpls
    :param key_ct: nparray
    :param key_l1: nparray
    :return: None, just prints
    """
    polygon = Polygon(bounding_points, closed=True, facecolor="none", edgecolor="r", linewidth=1)
    # Create a figure and axes object
    fig, ax = plt.subplots()

    # Plot the image on the axes
    ax.imshow(key_ct + key_l1, cmap="gray_r")

    # Use the fill function to plot the polygon on the image
    ax.add_patch(polygon)

    for i, point in enumerate(bounding_points):
        # circle = Circle(point, 4*i, color=f"{0.2 * (i + 1)}")
        # ax.add_patch(circle)
        plt.annotate(i, point, size=10)

    # Show the plot
    plt.show()

    return bounding_points


def create_aorta_mask_return_center(ct, layer, estimated_aorta_center, show_level=1):
    """

    :param ct:
    :param layer:
    :param estimated_aorta_center:
    :param show_level:
    :return:
    """
    print(f"create_aorta_mask_return_center(ct, layer={layer}, estimated_aorta_center={estimated_aorta_center})")
    # Extract ct layer
    ct_layer = ct[:, :, layer]
    ct_layer_for_preview = ct_layer / ct_layer.max()

    # Create an array with the shape of the image
    circle_mask = create_circular_mask(ct_layer.shape, estimated_aorta_center, 30).astype(bool)
    show(ct_layer_for_preview + circle_mask)
    print("circle_mask")

    # Crop a circle around the estimate position of the aorta
    key_circle = np.ma.masked_array(ct_layer, mask=~circle_mask, fill_value=ct_layer.min()).filled()
    show(ct_layer_for_preview + key_circle)
    print(f"key_circle")

    # Find the aorta_circle in the key_circle
    cx, cy, radius = get_circle_around_aorta(key_circle, estimated_aorta_center, show_level=show_level)
    print(f"aorta_circle: ({cx},{cy}), r={radius}")

    # Extract the center of the aorta_circle
    aorta_center = (cx, cy)
    aorta_center_preview = np.zeros_like(key_circle)
    aorta_center_preview[cy - 3:cy + 3, cx - 3:cx + 3] = 1
    show(ct_layer_for_preview + aorta_center_preview)
    print(f"aorta_center: {aorta_center}")

    # Create a segmentation of the circle around the aorta
    aorta_mask = create_circular_mask(key_circle.shape, aorta_center, radius)
    show(ct_layer_for_preview + aorta_mask)
    print("aorta_mask")

    return aorta_mask, aorta_center


def create_circular_mask(shape, center, radius):
    """
    enough is enough
    :param shape: I
    :param center: Want
    :param radius: To
    :return: Sleep
    """
    h = shape[0]
    w = shape[1]
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def get_circle_around_aorta(slice_mask, estimated_center, rad_min=10, rad_max=20, rad_step=1, show_level=1):
    """
    :param slice_mask: nparray
    :param estimated_center: (x,y)
    :param rad_min: int
    :param rad_max: ont
    :param rad_step: ant
    :param show_level: unt
    :return: x,y,r
    """
    # Use the Canny edge detector to find edges in the image
    edges = canny(slice_mask, sigma=2)

    # Use the Hough Circle Transform to find circles in the image
    hough_radii = np.arange(rad_min, rad_max, rad_step)
    hough_res = hough_circle(edges, hough_radii)

    # Use the hough_circle_peaks() function to find the centers and radii of the circles
    x, y, radius = find_circle_closest_to_point(estimated_center, hough_res, hough_radii)

    plot_circle_around_aorta(slice_mask, x, y, radius, show_level=show_level)
    print(f"Estimated center: {estimated_center}")
    print(f"Closest circle: ({x},{y}), radius={radius}")

    return x, y, radius


def find_circle_closest_to_point(point, hough_res, hough_radii):
    """
    Name os self explanatory
    :param point: (x,y)
    :param hough_res: look up the docs
    :param hough_radii: likewise
    :return: x,y,r
    """
    # Select the most prominent circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=3)

    distances = [(cx[i] - point[0]) ** 2 + (cy[i] - point[1]) ** 2 for i in range(len(cx))]
    closest_index = np.argmin(distances)

    # Draw the closest circle on the original image
    return cx[closest_index], cy[closest_index], radii[closest_index]


def plot_circle_around_aorta(slice_mask, x, y, radius, show_level=1):
    """
    Plot a circle with given params over the given slice of the image
    :param slice_mask: nparray
    :param x: int
    :param y: int
    :param radius: int
    :param show_level: secret
    :return: None, just plots
    """
    if (show_level >= SHOW_LEVEL):
        # Use imshow() to display the image with the detected circle
        plt.imshow(slice_mask, cmap="gray_r")

        # Use the plot() function to draw a circle on the image using the detected center and radius
        circle = plt.Circle((x, y), radius, color="r", fill=False)
        ax = plt.gca()
        ax.add_artist(circle)

        # Show the plot
        plt.show()


def clip_to_ROI(GT_seg, est_seg):
    """
    Zero out any layer of GT where est has no values
    :param GT_seg: nparray
    :param est_seg: nparray
    :return: GT and est after clipping
    """
    # Find the indices of the slices with data in est_seg
    slice_indices = np.where(np.any(est_seg, axis=(0, 1)))[0]
    first_slice = slice_indices[0]
    last_slice = slice_indices[-1]
    logging.info(f"slice_indices={slice_indices}")
    GT_seg[:, :, :first_slice] = 0
    est_seg[:,:,:first_slice] = 0
    GT_seg[:, :, last_slice:] = 0
    est_seg[:,:,last_slice:] = 0
    return GT_seg, est_seg
