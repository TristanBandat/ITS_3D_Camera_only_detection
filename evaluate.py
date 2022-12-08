import torch


# TODO: prototype, improve
def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device):
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
