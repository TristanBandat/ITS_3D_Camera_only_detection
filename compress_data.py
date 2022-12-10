import pickle
import numpy as np
import os
from ImageDataset import ImageDataset
from os.path import join
import torch
from skimage.transform import resize
from skimage.color import rgb2gray
import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt


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


def compress(dataset: ImageDataset, compression_factor=4):
    comp_X = int(dataset.get_X() / compression_factor)
    comp_Y = int(dataset.get_Y() / compression_factor)
    comp_data = list()
    update_progress_bar = tqdm.tqdm(total=len(dataset.image_files), desc="Images: ", position=0)
    for element in dataset.image_files:
        image_tensor_resized = torch.from_numpy(resize(rgb2gray(element['image']), (comp_X, comp_Y)))
        label_image = calculate_label_image(element['boxes'], (dataset.get_X(), dataset.get_Y()))
        label_tensor_resized = torch.from_numpy(resize(label_image, (comp_X, comp_Y)))
        comp_data.append({'image': image_tensor_resized, 'boxes': label_tensor_resized})
        update_progress_bar.update(1)
    return comp_data


def compress_frame_list(frames: list, compression_factor=4):
    full_X = 1280
    full_Y = 1920
    comp_X = full_X / compression_factor
    comp_Y = full_Y / compression_factor
    comp_data = list()
    update_progress_bar = tqdm.tqdm(total=len(frames), desc="Frames: ", position=0)
    for frame in frames:
        tf_image = tf.image.decode_jpeg(frame.images[0].image)
        np_image = tf_image.numpy()
        boxes = frame.camera_labels
        image_tensor_resized = torch.from_numpy(resize(rgb2gray(np_image), (comp_X, comp_Y)))
        label_image = calculate_label_image(boxes, (full_X, full_Y))
        label_tensor_resized = torch.from_numpy(resize(label_image, (comp_X, comp_Y)))
        comp_data.append({'image': image_tensor_resized, 'boxes': label_tensor_resized})
        update_progress_bar.update(1)
    return comp_data


def main():
    filename = 'final_data/waymo-data_part2_comp.pkl'
    dataset = ImageDataset(frame_path=join(os.getcwd(), 'data/data_part5_8.pkl'))
    compressed_data = compress(dataset, compression_factor=6)
    with open((join(os.curdir, filename)), 'wb') as f:
        pickle.dump(compressed_data, f)


if __name__ == '__main__':
    main()
