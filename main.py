import os
import pickle
from os.path import join
import numpy as np
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import tensorflow as tf
import tqdm
from CNN import CNN
from ImageDataset import ImageDataset
from utils import collate_fn



#TODO: prototype, improve
def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader):
    model.eval()
    loss_fn = torch.nn.L1Loss()
    loss = 0
    with torch.no_grad():
        for batch in dataloader:
            image_array, target_array = batch

            # change dimensions of image_array to fit CNN
            image_array = image_array.permute(0, 3, 1, 2)
            # move to device
            image_array = image_array.to(device)
            target_array = target_array.to(device)
            # get output
            output = model(image_array)
            loss += loss_fn(output, target_array)

    model.train()
    loss /= len(dataloader)
    return loss


def main(results_path="results", device: torch.device = torch.device("cuda:0"), n_updates: int = int(50000)):
    # TODO: Maybe change this to 3 separate files, but this is easier for now
    image_dataset = ImageDataset(frame_path=join(os.getcwd(), 'data.pkl'))
    train_set = torch.utils.data.Subset(image_dataset, indices=np.arange(int(len(image_dataset) * (3 / 5))))
    valid_set = torch.utils.data.Subset(image_dataset, indices=np.arange(int(len(image_dataset) * (3 / 5)),
                                                                         int(len(image_dataset) * (4 / 5))))
    test_set = torch.utils.data.Subset(image_dataset, indices=np.arange(int(len(image_dataset) * (4 / 5)),
                                                                        int(len(image_dataset))))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, collate_fn=collate_fn,
                                               shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(valid_set, batch_size=1, collate_fn=collate_fn,
                                             shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, collate_fn=collate_fn,
                                              shuffle=True, num_workers=0)

    writer = SummaryWriter(log_dir=os.path.join(results_path, 'tensorboard'))

    print_stats_at = 100  # print status to tensorboard every x updates
    validate_at = 5000  # evaluate model on validation set and check for new best model every x updates
    update = 0  # current update counter
    best_validation_loss = np.inf  # best validation loss so far
    no_update_counter = 0  # counter for how many updates have passed without a new best validation loss

    # Create CNN Network
    net = CNN(n_hidden_layers=5, n_input_channels=3, n_hidden_kernels=64, kernel_size=3, activation_fn=torch.nn.ReLU)
    net.to(device)

    # get l1 loss
    loss_fn = torch.nn.L1Loss()

    # get adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)

    # Save initial model as "best" model (will be overwritten later)
    torch.save(net, os.path.join(results_path, 'best_model.pt'))

    update_progess_bar = tqdm.tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)  # progressbar
    while update < n_updates and no_update_counter < 3:
        for batch in train_loader:
            # get data
            image_array, target_array = batch


            # change dimensions of image_array to fit CNN
            image_array = image_array.permute(0, 3, 1, 2)

            image_array = image_array.to(device)
            target_array = target_array.to(device)
            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            output = net(image_array)

            # Calculate loss
            loss = loss_fn(output, target_array)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Update progress bar
            update_progess_bar.set_description(f"loss: {loss.item():7.5f}")
            update_progess_bar.update(1)

            # Evaluate model on validation set
            if (update + 1) % validate_at == 0 and update > 0:
                val_loss = evaluate_model(net, dataloader=val_loader)
                writer.add_scalar(tag="validation/loss", scalar_value=val_loss, global_step=update)
                # Add weights as arrays to tensorboard
                for i, param in enumerate(net.parameters()):
                    writer.add_histogram(tag=f'validation/param_{i}', values=param,
                                         global_step=update)
                # Add gradients as arrays to tensorboard
                for i, param in enumerate(net.parameters()):
                    writer.add_histogram(tag=f'validation/gradients_{i}',
                                         values=param.grad.cpu(),
                                         global_step=update)
                # Save best model for early stopping
                if best_validation_loss > val_loss:
                    no_update_counter = 0
                    best_validation_loss = val_loss
                    torch.save(net, os.path.join(results_path, 'best_model.pt'))

            # Print current status and score
            if (update + 1) % print_stats_at == 0 and update > 0:
                writer.add_scalar(tag="training/loss",
                                  scalar_value=loss.cpu(),
                                  global_step=update)
            update += 1
            if update >= n_updates or no_update_counter >= 3:
                break

    update_progess_bar.close()
    print('Finished Training!')

    # Load best model and compute score on test set
    print(f"Computing scores for best model")
    net = torch.load(os.path.join(results_path, 'best_model.pt'))
    test_loss = evaluate_model(net, dataloader=test_loader)
    val_loss = evaluate_model(net, dataloader=val_loader)
    train_loss = evaluate_model(net, dataloader=train_loader)

    print(f"Scores:")
    print(f"test loss: {test_loss}")
    print(f"validation loss: {val_loss}")
    print(f"training loss: {train_loss}")
    # Demo Code



if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    main('results', device)
