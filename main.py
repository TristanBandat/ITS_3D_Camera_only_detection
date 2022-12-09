from CNN import CNN
from UNet import UNet
from train import train
from utils import collate_fn

import torch
import torch.utils.data
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    ############
    # Parameters
    ############

    cnn_net = CNN(n_hidden_layers=5, n_input_channels=1, n_hidden_kernels=64, kernel_size=3)
    unet = UNet(n_channels=1, n_classes=2, bilinear=False)  # classes are 1 and 0 (car, no car)

    # select a model
    net = cnn_net


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    lr = 1e-3

    weight_decay = 1e-5

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    batchsize = 16

    loss_fn = torch.nn.L1Loss()

    nupdates = 50000

    testset_ratio = 1 / 5

    validset_ratio = 1 / 5

    num_workers = 0

    seed = 1234

    resultpath = 'results/cnn'

    datapath = os.path.join(os.getcwd(), 'data/waymo-data_part1_comp.pkl')

    collate_function = collate_fn  # TODO: Maybe not needed

    ############
    # Invoke training method with specified parameters
    ############

    train(
        net=net,
        device=device,
        optim=optimizer,
        batchsize=batchsize,
        loss_fn=loss_fn,
        nupdates=nupdates,
        testset_ratio=testset_ratio,
        validset_ratio=validset_ratio,
        num_workers=num_workers,
        seed=seed,
        datapath=datapath,
        resultpath=resultpath,
        collate_fn=None  # TODO: Change if needed otherwise delete parameter from method and delete method from utils
    )


if __name__ == '__main__':
    main()
