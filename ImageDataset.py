import pickle
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, frame_path):
        self.pkl = None
        self.image_files = list()
        # load frames from pickle
        with open(frame_path, "rb") as f:
            self.pkl = pickle.load(f)

        height = self.pkl[0]['image'].shape[0]
        width = self.pkl[0]['image'].shape[1]

        # iterate over frames
        for f in self.pkl:
            # convert to numpy array
            np_image = f['image'].numpy().reshape(1, height, width)
            boxes = f['boxes'].numpy().reshape(1, height, width)

            self.image_files.append({'image': np_image, 'boxes': boxes})

    def __getitem__(self, index):
        return self.image_files[index]

    def __len__(self):
        return len(self.image_files)
