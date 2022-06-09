import torch
import torch.nn as nn


def generate_mask(sequence_batch, sequence_length):
    # sequence_batch [b, l]
    # sequence_length [b]
    # return: [b, l]
    mask = sequence_batch.new_ones(sequence_batch.shape[0], sequence_batch.shape[1], dtype=torch.float)
    mask[sequence_batch == 1] = 0.0
    return mask


if __name__ == '__main__':
    sequence_batch = torch.tensor([[2, 1, 1], 
                                    [4, 2, 1]])
    sequence_length = torch.tensor([1, 2])
    mask = generate_mask(sequence_batch, sequence_length)
    import pdb
    pdb.set_trace()