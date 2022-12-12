import torch


def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device, loss_fn):
    model.eval()
    loss = 0
    with torch.no_grad():
        for batch in dataloader:
            image_array = batch['image']
            target_array = batch['boxes']

            # move to device
            image_array = image_array.to(device, dtype=torch.float32)
            target_array = target_array.to(device, dtype=torch.float32)

            # get output
            output = model(image_array)

            loss += loss_fn(output, target_array)

    model.train() # setting model back to training mode
    loss /= len(dataloader)
    return loss
