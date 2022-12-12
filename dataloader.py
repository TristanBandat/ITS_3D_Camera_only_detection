import numpy as np
from torch.utils.data import Subset
from torch.utils.data import DataLoader


def get_dataloaders(dataset, valid_ratio, test_ratio, batchsize, num_workers):
    assert valid_ratio + test_ratio < 1, "Testset size plus Validationset size are not smaller 1"

    train_ratio = 1 - test_ratio - valid_ratio

    train_start_index = 0
    train_end_index = int(len(dataset) * train_ratio)
    valid_start_index = train_end_index
    valid_end_index = int(len(dataset) * (1 - test_ratio))
    test_start_index = valid_end_index
    test_end_index = len(dataset)

    train_set = Subset(dataset, indices=np.arange(train_start_index, train_end_index))
    valid_set = Subset(dataset, indices=np.arange(valid_start_index, valid_end_index))
    test_set = Subset(dataset, indices=np.arange(test_start_index, test_end_index))

    train_loader = DataLoader(train_set, batch_size=batchsize, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_set, batch_size=batchsize, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batchsize, shuffle=True, num_workers=num_workers)

    return train_loader, valid_loader, test_loader
