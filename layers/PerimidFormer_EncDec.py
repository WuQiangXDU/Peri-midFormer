import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from itertools import permutations

# mask for periodic pyramid
def mask_and_indexes(components_per_level):

    all_comp_num = np.lcm.reduce(components_per_level)
    inclu_rela = np.zeros((len(components_per_level), all_comp_num)).astype(int)

    d = 0
    for i, components in enumerate(components_per_level):
        compent_length = all_comp_num // components

        for j in range(components):
            start_idx = j * compent_length
            end_idx = (j + 1) * compent_length
            inclu_rela[i, start_idx:end_idx] = d
            d += 1

    max_ind = inclu_rela[-1, -1]
    mask_metric = np.zeros((max_ind + 1, max_ind + 1)).astype(int)
    mask_metric[0, 0] = 1

    indexes = []
    indexes_layer_all = []

    #
    for i in range(len(components_per_level)):
        if i == 0:
            continue
        unique_elements_above = np.unique(inclu_rela[i - 1, :])
        unique_elements_down = np.unique(inclu_rela[i, :])
        permutations_result = list(permutations(unique_elements_down, 2))
        idx1, idx2 = np.array(permutations_result).T
        mask_metric[idx1, idx2] = 1

        indexes_layer = []

        #  Inclusion relationship of components in neighboring levels
        for j_down, seg_id_down in enumerate(range(unique_elements_down[0], unique_elements_down[-1] + 1)):
            mask_metric[seg_id_down, seg_id_down] = 1
            branch_number = 0
            for j_above, seg_id_above in enumerate(range(unique_elements_above[0], unique_elements_above[-1] + 1)):
                indices = np.where(inclu_rela[i, :] == seg_id_down)[0]
                indices_above = np.where(inclu_rela[i - 1, :] == seg_id_above)[0]
                if (((indices[0] >= indices_above[0]) and (indices[-1] <= indices_above[-1])) or  # The case where a component in the lower level is completely contained in a component in the upper level
                        ((indices[0] <= indices_above[-1]) and (
                                indices[-1] >= indices_above[-1])) or  # The case where a component of the lower level is semi-contained in a component of the upper level
                        ((indices[0] <= indices_above[0]) and (indices[-1] >= indices_above[0]))):  # The case where a lower component and a half are contained in the head of an upper component
                    mask_metric[seg_id_above, seg_id_down] = 1
                    mask_metric[seg_id_down, seg_id_above] = 1
                    branch_number += 1

                    indexes_layer.append([seg_id_above, seg_id_down])

                    continue
                else:
                    continue
        if i == 1:
            indexes = indexes_layer
            indexes_layer_all.append(indexes_layer)
            continue
        indexes_layer_all.append(indexes_layer)

        count_dict = {}
        for sublist in indexes_layer:
            first_element = sublist[0]
            count_dict[first_element] = count_dict.get(first_element, 0) + 1

        indexs_temporary = []

        for ii, index_list in enumerate(indexes):
            key_value = index_list[-1]
            indices_with_first_element = [index for index, sublist in enumerate(indexes_layer) if
                                          sublist[0] == key_value]

            for repeated_index in indices_with_first_element:
                new_list = indexes[ii][0:-1] + indexes_layer[repeated_index]
                indexs_temporary.append(new_list)
        indexes = indexs_temporary

    indexes = np.array(indexes)

    rows, cols = mask_metric.shape
    for i in range(rows):
        for j in range(cols):
            if i < j:
                mask_metric[i, j] = 0

    mask_metric = torch.tensor(mask_metric)
    indexes = torch.tensor(indexes)
    mask_metric = (1 - mask_metric).bool()
    indexes = indexes.unsqueeze(0).unsqueeze(3)

    return mask_metric, indexes


class RegularMask():
    def __init__(self, mask):
        mask = mask.unsqueeze(1).unsqueeze(1)
        self._mask = mask

    @property
    def mask(self):
        return self._mask


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, dropout=0.1, normalize_before=True):
        super(EncoderLayer, self).__init__()

        self.slf_attn = AttentionLayer(
            FullAttention(mask_flag=True, factor=0,
                          attention_dropout=dropout, output_attention=False),
            d_model, n_head)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, slf_attn_mask=None):
        attn_mask = RegularMask(slf_attn_mask)
        enc_output, _ = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=attn_mask)
        enc_output = enc_output + enc_input
        enc_output = self.pos_ffn(enc_output)
        return enc_output


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, configs):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(configs.d_model, configs.d_model, configs.n_heads, dropout=configs.dropout,
                         normalize_before=False) for _ in range(configs.layers)
        ])


    def forward(self, chan_in, components_per_level, task):

        chan_in = chan_in.permute(0, 1, 3, 2)
        channels_num = chan_in.shape[2]
        mask, indexes = mask_and_indexes(components_per_level)
        mask = mask.repeat(len(chan_in), 1, 1).to(chan_in.device)

        # multilayer
        for i in range(len(self.layers)):
            chan_in = self.layers[i](chan_in, mask)

        enc_out = chan_in.permute(0, 2, 1, 3)
        enc_out = torch.reshape(enc_out, (enc_out.shape[0] * enc_out.shape[1], enc_out.shape[2], enc_out.shape[3]))

        # Classification task do not use periodic feature flows
        if task != 'classification':
            num_of_branches = indexes.shape[1]
            indexes = indexes.repeat(enc_out.size(0), 1, 1, enc_out.size(2)).to(enc_out.device)
            indexes = indexes.view(enc_out.size(0), -1, enc_out.size(2))
            all_enc = torch.gather(enc_out, 1, indexes)
            enc_out = all_enc.view(enc_out.size(0), num_of_branches, -1) # [batch_size, num_of_periodic_feature_flows, dimension_of_flow]

        enc_out = torch.reshape(enc_out, (-1, channels_num, enc_out.shape[-2], enc_out.shape[-1]))

        return enc_out


class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x
