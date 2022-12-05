import pickle
import tensorflow as tf
from torch.utils.data import Dataset


#Todo: Change resolution of images
class ImageDataset(Dataset):
    def __init__(self, frame_path):
        self.image_files = []
        # load frames from pickle
        frames = open(frame_path, "rb")
        pkl = pickle.load(frames)
        frames.close()
        # iterate over frames
        for f in pkl:
            for index, image in enumerate(f.images):
                # get image data
                tf_image = tf.image.decode_jpeg(image.image)

                # convert to numpy array
                np_image = tf_image.numpy()

                # convert np_image to torch.tensor
                self.image_files.append(np_image)

    def __getitem__(self, index):
        return self.image_files[index]

    def __len__(self):
        return len(self.image_files)
