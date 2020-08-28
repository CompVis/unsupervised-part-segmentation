import cv2
import numpy as np
from edflow.iterators import batches

from supermariopy import imageutils


def make_legend_image(legend_labels, colors, text_colors, legend_size, n_cols):
    """
    legend_labels : list of str
        list of semantic labels (str) for each label
    colors : ndarray or list
        list of same length as legend_labels with colors for each label OR ndarray of shape [n_labels, 3]. Colors in range [0, 1]
    text_colors : ndarray or list
        list of same length as legend_labels with colors for each text OR ndarray of shape [n_labels, 3]. Colors in range [0, 1]
    legend_size : tuple of ints
        the size of the legend image. the width should match the label_image because it is stacked vertically
    n_cols : int
        into how many colums the legend labels should be splitted for visualization

    # TODO: example usage
    """
    N_labels = len(legend_labels)
    box_h = 512 // len(legend_labels)
    box_w = 512 // n_cols
    legend = [
        colors[np.ones((box_h, box_w), dtype=np.int16) * i] for i in range(N_labels)
    ]
    legend = [
        imageutils.put_text(
            patch, t, loc="center", font_scale=1.0, thickness=1, color=tc
        )
        for patch, t, tc in zip(legend, legend_labels, text_colors)
    ]
    legend = np.stack(legend, axis=0)
    legend = batches.batch_to_canvas(legend, cols=n_cols)
    legend = cv2.resize(legend, (legend_size[1], legend_size[0]), cv2.INTER_LANCZOS4)
    return legend


def blend_labels(labels, image, alpha):
    """
    blend segmentation labels onto image with some alpha.
    labels and image have to be in range [0, 1]
    label_image : ndarray
        [x, y, 3]  image or image grid of color labels in range [0, 1]
    image : ndarray
        [x, y, 3] RGB image in range [0, 1]
    """
    blended = image * (1 - alpha) + alpha * labels
    return blended


def make_figure(
    label_image,
    image,
    legend_labels,
    label_colors,
    text_colors,
    legend_size,
    n_legend_cols,
    alpha,
):
    """
    make figure by blending label_image and image and then adding a legend image on top.
    The legend will be stacked on top if label_image (vertically)

    label_image : ndarray
        [x, y, 3]  image or image grid of color labels in range [0, 1]
    image : ndarray
        [x, y, 3] RGB image in range [0, 1]
    legend_labels : list of str
        list of semantic labels (str) for each label
    label_colors : ndarray or list
        list of same length as legend_labels with colors for each label OR ndarray of shape [n_labels, 3]. Colors in range [0, 1]
    text_colors : ndarray or list
        list of same length as legend_labels with colors for each text OR ndarray of shape [n_labels, 3]. Colors in range [0, 1]
    legend_size : tuple of ints
        the size of the legend image. the width should match the label_image because it is stacked vertically
    n_legend_cols : int
        into how many colums the legend labels should be splitted for visualization
    alpha : float
        alpha for merging label_image and image
    """
    legend_image = make_legend_image(
        legend_labels, label_colors, text_colors, legend_size, n_legend_cols
    )
    blended = blend_labels(label_image, image, alpha)
    figure = np.vstack([legend_image, blended])
    return figure
