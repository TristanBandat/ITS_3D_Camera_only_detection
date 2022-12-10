import pickle
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, frame_path):
        self.image_files = list()
        # load frames from pickle
        with open(frame_path, "rb") as f:
            pkl = pickle.load(f)

        self.height = pkl[0]['image'].shape[0]
        self.width = pkl[0]['image'].shape[1]

        # iterate over frames
        for f in pkl:
            # convert to numpy array
            np_image = f['image'][None, :]
            boxes = f['boxes'][None, :]
            self.image_files.append({'image': np_image, 'boxes': boxes})

    def __getitem__(self, index):
        return self.image_files[index]

    def __len__(self):
        return len(self.image_files)
