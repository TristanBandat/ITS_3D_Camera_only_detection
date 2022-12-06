import pickle
import tensorflow as tf
from torch.utils.data import Dataset


# TODO: Change resolution of images
class ImageDataset(Dataset):
    def __init__(self, frame_path):
        self.image_files = list()
        # load frames from pickle
        with open(frame_path, "rb") as f:
            self.pkl = pickle.load(f)
        # iterate over frames
        for f in self.pkl:
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
