from CNN import CNN
from UNet import UNet
from train import train


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
    unet = UNet(n_channels=1, n_classes=1, bilinear=False)  # classes are 1 and 0 (car, no car) therefore 1 class

    # select a model
    net = unet


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    lr = 1e-3 # initial 1e-3

    weight_decay = 1e-5 # initial 1e-5

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    batchsize = 16

    loss_fn = torch.nn.L1Loss()
    loss_fn_new = torch.nn.BCEWithLogitsLoss()

    nupdates = 5000

    testset_ratio = 1 / 5

    validset_ratio = 1 / 5

    num_workers = 0

    seed = 1234 # initial 1234

    resultpath = 'results/unet'

    datapath = os.path.join(os.getcwd(), 'data/new_data.pkl')

    print_stats_at = 100  # print status to tensorboard every x updates
    validate_at = 200  # evaluate model on validation set and check for new best model every x updates
    plot_images_at = 50 # plot model every 100 updates


    ############
    # Invoke training method with specified parameters
    ############

    train(
        net=net,
        device=device,
        optim=optimizer,
        batchsize=batchsize,
        loss_fn=loss_fn_new,
        nupdates=nupdates,
        testset_ratio=testset_ratio,
        validset_ratio=validset_ratio,
        num_workers=num_workers,
        seed=seed,
        datapath=datapath,
        resultpath=resultpath,
        print_stats_at=print_stats_at,
        validate_at = validate_at,
        plot_images_at = plot_images_at,
    )



if __name__ == '__main__':
    main()
