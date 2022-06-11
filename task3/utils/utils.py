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
    # attn_matrix [b, l1, l2]
    # mask [b, l1]
    expand_mask = mask.unsqueeze(1).expand_as(attn_matrix).contiguous()  # should be [b, l1, l2]
    reshape_matrix = attn_matrix.reshape(-1, attn_matrix.shape[-1])  # should be [b*l1, l2]
    reshape_mask = expand_mask.view(-1, expand_mask.shape[-1])  # should be [b*l1, l2]

    softmax_result = nn.functional.softmax(reshape_matrix * reshape_mask, dim=-1)  # should be [b*l1, l2]
    masked_softmax_result = softmax_result * reshape_mask  # should be [b*l1, l2]
    div_add = 1e-13  # TODO 1e-10 -> 1e-13
    final_softmax_result =  masked_softmax_result / (masked_softmax_result.sum(dim=-1, keepdim=True) + div_add)

    return final_softmax_result.view(attn_matrix.shape)


def weighted_sum(attn_matrix, data_vector, mask):
    # attn_matrix [b, l1, l2]
    # data_vector [b, l2, d] 实际上会有[b, l2, 2*hidden_size]
    # mask [b, l1]

    # TODO 需要检查data_vector padding部分，确保全为0
    weighted_vector = attn_matrix.bmm(data_vector)  # should be [b, l1, d]
    expand_mask = mask.unsqueeze(-1).expand_as(weighted_vector).contiguous()  # should be [b, l1, 1]
    # import pdb
    # pdb.set_trace()
    return expand_mask * weighted_vector  # should be [b, l, d]


def masked_max_pooling(data_vector, mask, value_to_add):
    # 避免padding的0对结果产生影响——因为data_vector有效的词向量部分可能出现负值

    # data_vector should be [b, l, d] 也即[b, l, 2*hidden_size]
    # mask [b, l]
    # value_to_add e.g. -1e7
    # import pdb
    # pdb.set_trace()
    expand_mask = mask.unsqueeze(1).transpose(2, 1)  # [b, l, 1]
    reversed_mask = 1 - expand_mask
    return data_vector * expand_mask + reversed_mask * value_to_add


if __name__ == '__main__':
    sequence_batch = torch.tensor([[2, 1, 1], 
                                    [4, 2, 1]])  # 注意这里都是索引！还没embedding
    sequence_length = torch.tensor([1, 2])
    data_vector = torch.rand((2, 3, 4), requires_grad=True)
    # 构造padding部分
    with torch.no_grad():  
        data_vector[0, 1:, :] = 0.0
        data_vector[1, 2:, :] = 0.0
    
    # generate mask test
    mask = generate_mask(sequence_batch, sequence_length)

    # masked_softmax test
    attn_matrix = torch.rand((2, 3, 3))  # [b, l, l]
    final_softmax_result = masked_softmax(attn_matrix, mask)

    # weighted sum test
    weighted_vector = weighted_sum(attn_matrix, data_vector, mask)

    # masked_max_pooling test
    max_vector =  masked_max_pooling(data_vector, mask, -1e7)
    import pdb
    pdb.set_trace()