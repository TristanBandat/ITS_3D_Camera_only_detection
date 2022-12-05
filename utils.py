import numpy as np
import torch
import math
import os
from matplotlib import pyplot as plt


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


def collate_fn(batch_as_list: list):
    #
    # Handle sequences
    #
    # Get the maximum sequence length in the current minibatch
    max_X = np.max([seq['image'].shape[0] for seq in batch_as_list])
    # Allocate a tensor that can fit all padded sequences
    max_Y = np.max([seq['image'].shape[1] for seq in batch_as_list])
    stacked_sequences = torch.zeros(size=(len(batch_as_list), max_X, max_Y,
                                          batch_as_list[0]['image'].shape[2]), dtype=torch.int32)
    stacked_labels = torch.zeros(size=(len(batch_as_list), max_X, max_Y), dtype=torch.int32)
    # Write the sequences into the tensor stacked_sequences
    for i, sequence in enumerate(batch_as_list):
        stacked_sequences[i] = torch.from_numpy(sequence['image'])
        label_image = calculate_label_image(sequence, (max_X, max_Y))
        stacked_labels[i] = torch.from_numpy(label_image)

    return stacked_sequences, stacked_labels
