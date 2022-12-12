import os
import matplotlib.pyplot as plt
import torch

def plot(inputs, targets, predictions, path, update):
    """Plotting the inputs, targets and predictions to file `path`"""
    os.makedirs(path, exist_ok=True)
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

    for ax, data, title in zip(axes, [inputs, targets, predictions], ["Input", "Target", "Prediction"]):
        ax.clear()
        ax.set_title(title)
        ax.imshow(data, interpolation="none")
        # ax.imshow(np.transpose((data[i]), (1, 2, 0)), interpolation="none")
        ax.set_axis_off()
    fig.savefig(os.path.join(path, f"{update+1:07d}.png"), dpi=100)

    plt.close(fig)
