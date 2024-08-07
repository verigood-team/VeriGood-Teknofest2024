import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler

class LCFAPCModelList(list):

    from .fast_lcf_bert import FAST_LCF_BERT

   
    FAST_LCF_BERT = FAST_LCF_BERT
  
    def __init__(self):
        model_list = [
            # self.SLIDE_LCFS_BERT,
         
            self.FAST_LCF_BERT,
      
        ]
        super().__init__(model_list)


class APCModelList(LCFAPCModelList):
    pass

class BERT_MLP(nn.Module):
    inputs = ["text_indices"]

    def __init__(self, bert, config):
        super(BERT_MLP, self).__init__()
        self.bert = bert
        self.pooler = BertPooler(bert.config)
        self.dense = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, inputs):
        text_raw_indices = inputs[0]
        last_hidden_state = self.bert(text_raw_indices)["last_hidden_state"]
        pooled_out = self.pooler(last_hidden_state)
        out = self.dense(pooled_out)
        return out


class BERTTCModelList(list):
    
    BERT = BERT_MLP

    def __init__(self):
        super(BERTTCModelList, self).__init__([self.BERT_MLP])


class DynamicLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=True,
        dropout=0,
        bidirectional=False,
        only_use_last_hidden_state=False,
        rnn_type="LSTM",
    ):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        :param rnn_type: {LSTM, GRU, RNN}
        """
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type

        if self.rnn_type == "LSTM":
            self.RNN = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        elif self.rnn_type == "GRU":
            self.RNN = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        elif self.rnn_type == "RNN":
            self.RNN = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional,
            )

    def forward(self, x, x_len):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort

        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""
        x_sort_idx = torch.sort(-x_len)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths=x_len.cpu(), batch_first=self.batch_first
        )

        # process using the selected RNN
        if self.rnn_type == "LSTM":
            out_pack, (ht, ct) = self.RNN(x_emb_p, None)
        else:
            out_pack, ht = self.RNN(x_emb_p, None)
            ct = None
        """unsort: h"""
        ht = torch.transpose(ht, 0, 1)[
            x_unsort_idx
        ]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        ht = torch.transpose(ht, 0, 1)

        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            out = torch.nn.utils.rnn.pad_packed_sequence(
                out_pack, batch_first=self.batch_first
            )  # (sequence, lengths)
            out = out[0]  #
            out = out[x_unsort_idx]
            """unsort: out c"""
            if self.rnn_type == "LSTM":
                ct = torch.transpose(ct, 0, 1)[
                    x_unsort_idx
                ]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
                ct = torch.transpose(ct, 0, 1)

            return out, (ht, ct)


class LSTM(nn.Module):
    inputs = ["text_indices"]

    def __init__(self, embedding_matrix, config):
        super(LSTM, self).__init__()
        self.config = config
        self.embed = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float)
        )
        self.lstm = DynamicLSTM(
            config.embed_dim, config.hidden_dim, num_layers=1, batch_first=True
        )
        self.dense = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, inputs):
        text_raw_indices = inputs[0]
        x = self.embed(text_raw_indices)
        x_len = torch.sum(text_raw_indices != 0, dim=-1)
        _, (h_n, _) = self.lstm(x, x_len)
        out = self.dense(h_n[0])
        return out


class GloVeTCModelList(list):

    LSTM = LSTM

    def __init__(self):
        super(GloVeTCModelList, self).__init__([self.LSTM])
