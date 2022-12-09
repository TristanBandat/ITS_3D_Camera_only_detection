from dataloader import get_dataloaders
from ImageDataset import ImageDataset
from evaluate import evaluate_model

import os
import numpy as np
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import tqdm


def train(net, device, optim, batchsize, loss_fn, nupdates, testset_ratio, validset_ratio, num_workers, seed,
          datapath='data.pkl', resultpath='results', collate_fn=None):
    # Setting seed
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)

    image_dataset = ImageDataset(frame_path=datapath)

    train_loader, valid_loader, test_loader = get_dataloaders(image_dataset, testset_ratio, validset_ratio, batchsize,
                                                              num_workers, collate_fn)

    writer = SummaryWriter(log_dir=os.path.join(resultpath, 'tensorboard'))

    print_stats_at = 100  # print status to tensorboard every x updates
    validate_at = 5000  # evaluate model on validation set and check for new best model every x updates
    update = 0  # current update counter
    best_validation_loss = np.inf  # best validation loss so far
    no_update_counter = 0  # counter for how many updates have passed without a new best validation loss

    # Move network to device
    net.to(device)

    # get loss
    loss_fn = loss_fn

    # get adam optimizer
    optimizer = optim

    # Save initial model as "best" model (will be overwritten later)
    torch.save(net, os.path.join(resultpath, 'best_model.pt'))

    update_progessbar = tqdm.tqdm(total=nupdates, desc=f"loss: {np.nan:7.5f}", position=0)  # progressbar
    while update < nupdates and no_update_counter < 3:
        for batch in train_loader:

            # get data
            image_array = batch['image']
            target_array = batch['boxes']

            # image_array = image_array.permute(0, 3, 1, 2) # TODO: Probably not needed

            image_array = image_array.to(device, dtype=torch.float32)
            target_array = target_array.to(device, dtype=torch.float32)

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
            update_progessbar.set_description(f"loss: {loss.item():7.5f}")
            update_progessbar.update(1)

            # Evaluate model on validation set
            if (update + 1) % validate_at == 0 and update > 0:
                val_loss = evaluate_model(net, dataloader=valid_loader, device=device, loss_fn=loss_fn)
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
                    torch.save(net, os.path.join(resultpath, 'best_model.pt'))

            # Print current status and score
            if (update + 1) % print_stats_at == 0 and update > 0:
                writer.add_scalar(tag="training/loss",
                                  scalar_value=loss.cpu(),
                                  global_step=update)
            update += 1
            if update >= nupdates or no_update_counter >= 3:
                break

    update_progessbar.close()
    print('Finished Training!')

    # Load best model and compute score on test set
    print(f"Computing scores for best model")
    net = torch.load(os.path.join(resultpath, 'best_model.pt'))
    test_loss = evaluate_model(net, dataloader=test_loader, device=device, loss_fn=loss_fn)
    val_loss = evaluate_model(net, dataloader=valid_loader, device=device, loss_fn=loss_fn)
    train_loss = evaluate_model(net, dataloader=train_loader, device=device, loss_fn=loss_fn)

    print(f"Scores:")
    print(f"test loss: {test_loss}")
    print(f"validation loss: {val_loss}")
    print(f"training loss: {train_loss}")
