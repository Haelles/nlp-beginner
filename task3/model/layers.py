import torch
import torch.nn as nn


class RNN_Dropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(RNN_Dropout, self).__init__()
        self.dropout_rate = dropout_rate
    
    def forward(self, data_vector):
        # ! .detach()
        mask = data_vector.detach().new_ones(data_vector.shape[0], data_vector.shape[-1])
        dropout_mask = nn.functional.dropout(mask, self.dropout_rate, self.training)
        return dropout_mask.unsqueeze(1) * data_vector


class BiLSTM(nn.Module):
    def __init__(self, embedded_dim=300, hidden_size=300, num_layers=1, bias=True, bidirectional=True, dropout_rate=0):
        super(BiLSTM, self).__init__()

        self.embedded_dim = embedded_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate

        self.lstm = nn.LSTM(input_size=embedded_dim, hidden_size=hidden_size, num_layers=num_layers, bias=bias, batch_first=True, dropout=dropout_rate, bidirectional=bidirectional)

    def forward(self, data_vector, data_length):
        # pytorch1.1后提供enforce_sorted参数，不再需要手动排序
        # ordered_lens, index =  data_length.sort(descending=True)
        # ordered_data = data_vector[index]
        # packed_seq_batch = nn.utils.rnn.pack_padded_sequence(ordered_data, ordered_lens, batch_first=True)
        # packed_output, _ = self.lstm(packed_seq_batch)
        # unpacked_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # recover_index = index.argsort()
        # recover_output = unpacked_output[recover_index]

        packed_seq_batch = nn.utils.rnn.pack_padded_sequence(data_vector, data_length, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_seq_batch)
        # 设置total_length为句子最大长度，避免出现第二维(time_step/max_sentence_length)缩短的情况
        unpacked_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=data_vector.shape[1])

        return  unpacked_output  # should be [b, n, 2*hidden_size]


if __name__ == '__main__':
    data_vector = torch.rand((2, 3, 4))
    data_length = torch.tensor([1, 2])
    # rnn_dropout = RNN_Dropout()
    # data_dropout = rnn_dropout(data_vector)
    encoder = BiLSTM(embedded_dim=4, hidden_size=4)
    import pdb
    pdb.set_trace()
    output = encoder(data_vector, data_length)

