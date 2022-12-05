import os

import numpy as np
import torch
from matplotlib import pyplot as plt

#TODO: only prototype, needs to be changed
def collate_fn(batch_as_list: list):
    #
    # Handle sequences
    #
    # Get the maximum sequence length in the current minibatch
    max_X = np.max([seq.shape[0] for seq in batch_as_list])
    # Allocate a tensor that can fit all padded sequences
    max_Y = np.max([seq.shape[1] for seq in batch_as_list])
    stacked_sequences = torch.zeros(size=(len(batch_as_list), max_X, max_Y, batch_as_list[0].shape[2]), dtype=torch.int32)
    # Write the sequences into the tensor stacked_sequences
    for i, sequence in enumerate(batch_as_list):
        stacked_sequences[i, :len(sequence), :len(sequence[0]), :] = torch.from_numpy(sequence)

    return stacked_sequences

