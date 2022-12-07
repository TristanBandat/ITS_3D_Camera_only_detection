import pickle
import numpy as np
import os
from ImageDataset import ImageDataset
from os.path import join
import torch
from skimage.transform import resize


def calculate_label_image(boxes, xy):
    """

    :param boxes: boxes where the labels are stored in
    :param xy: size of the image
    :return:
    """
    label = np.zeros(xy, dtype=np.int32)
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


def compress(compression_factor=4):
    dataset = ImageDataset(frame_path=join(os.getcwd(), 'data2/data_part1_4.pkl'))
    comp_X = int(dataset.get_X()/compression_factor)
    comp_Y = int(dataset.get_Y()/compression_factor)
    comp_data = list()
    for element in dataset.image_files:
        image_tensor_resized = torch.from_numpy(resize(element['image'], (comp_X, comp_Y)))
        label_image = calculate_label_image(element['boxes'], (dataset.get_X(), dataset.get_Y()))
        label_tensor_resized = torch.from_numpy(resize(label_image, (comp_X, comp_Y)))
        comp_data.append((image_tensor_resized, label_tensor_resized))
    return comp_data


def main():
    filename = 'final_data/waymo-data_part1_comp.pkl'
    compressed_data = compress()
    with open((join(os.curdir, filename)), 'wb') as f:
        pickle.dump(compressed_data, f)
    pass


if __name__ == '__main__':
    main()
