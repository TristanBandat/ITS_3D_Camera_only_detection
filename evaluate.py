import torch


# TODO: prototype, improve
def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device, loss_fn):
    model.eval()
    loss = 0
    with torch.no_grad():
        for batch in dataloader:
            image_array = batch['image']
            target_array = batch['boxes']

            # change dimensions of image_array to fit CNN
            #image_array = image_array.permute(0, 3, 1, 2)

            # move to device
            image_array = image_array.to(device, dtype=torch.float32)
            target_array = target_array.to(device, dtype=torch.float32)
            # get output
            output = model(image_array)
            loss += loss_fn(output, target_array)

    model.train()
    loss /= len(dataloader)
    return loss
