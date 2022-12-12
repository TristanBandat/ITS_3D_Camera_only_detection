import numpy as np
import torch


def calculate_label_image(sequence, xy):
    """

    :param sequence: sequence where the labels are stored in
    :param xy: size of the image
    :return:
    """
    label = np.zeros(xy, dtype=np.int32)
    boxes = sequence['boxes']
    for box_cluster in boxes:
        for s_label in box_cluster.labels:
            if s_label.type != 1:
                continue
            for y in range(int(s_label.box.center_y - 0.5 * s_label.box.width),
                           int(s_label.box.center_y + 0.5 * s_label.box.width)):
                for x in range(int(s_label.box.center_x - 0.5 * s_label.box.length),
                               int(s_label.box.center_x + 0.5 * s_label.box.length)):
                    label[y, x] = 1
    return label