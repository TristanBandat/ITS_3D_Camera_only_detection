import os
from os.path import join
import numpy as np
import torch
import torch.utils.data
from matplotlib import pyplot as plt
import tensorflow as tf
from ImageDataset import ImageDataset
from utils import collate_fn


def main():
    # TODO: Maybe change this to 3 separate files, but this is easier for now
    image_dataset = ImageDataset(frame_path=join(os.getcwd(), 'data/data.pkl'))
    train_set = torch.utils.data.Subset(image_dataset, indices=np.arange(int(len(image_dataset) * (3 / 5))))
    valid_set = torch.utils.data.Subset(image_dataset, indices=np.arange(int(len(image_dataset) * (3 / 5)),
                                                                         int(len(image_dataset) * (4 / 5))))
    test_set = torch.utils.data.Subset(image_dataset, indices=np.arange(int(len(image_dataset) * (4 / 5)),
                                                                        int(len(image_dataset))))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=2, collate_fn=collate_fn,
                                               shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(valid_set, batch_size=1, collate_fn=collate_fn,
                                             shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, collate_fn=collate_fn,
                                              shuffle=True, num_workers=1)

    # Demo Code
    for data_batch in train_loader:
        image_data, boxes = data_batch[0], data_batch[1]
        # Demo Code
        plt.imshow(image_data[0, :, :])
        plt.grid(False)
        plt.axis('off')
        plt.show()
        plt.imshow(boxes[0, :, :])
        plt.grid(False)
        plt.axis('off')
        plt.show()
        # print(boxes[0])


if __name__ == '__main__':
    main()
