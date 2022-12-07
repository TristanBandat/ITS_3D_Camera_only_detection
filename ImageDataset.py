import pickle
import tensorflow as tf
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, frame_path):
        self.x = 1280
        self.y = 1920
        self.image_files = list()
        # load frames from pickle
        with open(frame_path, "rb") as f:
            pkl = pickle.load(f)
        # iterate over frames
        for f in pkl:
            # get front image data
            tf_image = tf.image.decode_jpeg(f.images[0].image)
            # convert to numpy array
            np_image = tf_image.numpy()
            boxes = f.camera_labels
            # convert np_image to torch.tensor
            self.image_files.append({'image': np_image, 'boxes': boxes})

    def __getitem__(self, index):
        return self.image_files[index]

    def __len__(self):
        return len(self.image_files)

    def get_X(self):
        return self.x

    def get_Y(self):
        return self.y


# TODO: Change to fit the compressed data
class ImageDataset_comp(Dataset):
    def __init__(self, frame_path):
        self.image_files = list()
        # load frames from pickle
        with open(frame_path, "rb") as f:
            pkl = pickle.load(f)
        # iterate over frames
        for f in pkl:
            # get front image data
            tf_image = tf.image.decode_jpeg(f.images[0].image)
            # convert to numpy array
            np_image = tf_image.numpy()
            boxes = f.camera_labels
            # convert np_image to torch.tensor
            self.image_files.append({'image': np_image, 'boxes': boxes})

    def __getitem__(self, index):
        return self.image_files[index]

    def __len__(self):
        return len(self.image_files)
