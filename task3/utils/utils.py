import torch
import torch.nn as nn


def generate_mask(sequence_batch, sequence_length):
    # sequence_batch [b, l]
    # sequence_length [b]
    # return: [b, l]
    mask = sequence_batch.new_ones(sequence_batch.shape[0], sequence_batch.shape[1], dtype=torch.float)
    mask[sequence_batch == 1] = 0.0
    return mask


def masked_softmax(attn_matrix, mask):
    # attn_matrix [b, l, l]
    # mask [b, l]
    expand_mask = mask.unsqueeze(1).expand_as(attn_matrix).contiguous()  # should be [b, l, l]
    reshape_matrix = attn_matrix.reshape(-1, attn_matrix.shape[-1])  # should be [b*l, l]
    reshape_mask = expand_mask.view(-1, expand_mask.shape[-1])  # should be [b*l, l]

    softmax_result = nn.functional.softmax(reshape_matrix * reshape_mask, dim=-1)  # should be [b*l, l]
    masked_softmax_result = softmax_result * reshape_mask  # should be [b*l, l]
    div_add = 1e-10  # TODO 需要更小吗
    final_softmax_result =  masked_softmax_result / (masked_softmax_result.sum(dim=-1, keepdim=True) + div_add)

    return final_softmax_result.view(attn_matrix.shape)


def weighted_sum(attn_matrix, data_vector, mask):
    # attn_matrix [b, l, l]
    # data_vector [b, l, d]
    # mask [b, l]

    # TODO 需要检查data_vector padding部分，确保全为0
    weighted_vector = attn_matrix.bmm(data_vector)  # should be [b, l, d]
    expand_mask = mask.unsqueeze(-1).expand_as(data_vector).contiguous()
    import pdb
    pdb.set_trace()
    return expand_mask * weighted_vector  # should be [b, l, d]


if __name__ == '__main__':
    sequence_batch = torch.tensor([[2, 1, 1], 
                                    [4, 2, 1]])  # 注意这里都是索引！还没embedding
    sequence_length = torch.tensor([1, 2])
    data_vector = torch.rand((2, 3, 4))
    
    # generate mask test
    mask = generate_mask(sequence_batch, sequence_length)

    # masked_softmax test
    attn_matrix = torch.rand((2, 3, 3))  # [b, l, l]
    final_softmax_result = masked_softmax(attn_matrix, mask)

    # weighted sum test
    weighted_vector = weighted_sum(attn_matrix, data_vector, mask)

    import pdb
    pdb.set_trace()