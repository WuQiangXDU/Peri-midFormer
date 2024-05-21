import torch
import torch.nn as nn
import torch.fft
from layers.Embed import PositionEmbedding, TemporalEmbedding
from layers.PerimidFormer_EncDec import Encoder
import numpy as np
from utils.tools import series_decomp


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)

    top_list = torch.sort(top_list, descending=False).values

    # Make sure the first level is the original time series itself
    if top_list[0] != 1:
        top_list = torch.cat(
            [torch.tensor([1]).to(top_list.device), top_list[:-1]])

    top_list = top_list.detach().cpu().numpy()
    top_list = top_list[top_list != 0]
    top_list = np.unique(top_list)
    if len(top_list) == 1:
        top_list = np.append(top_list, top_list[0] * 2)
    period = x.shape[1] // top_list

    unique_period, unique_index = np.unique(period, return_index=True)
    unique_period = np.sort(unique_period)[::-1]

    return unique_period


class Peri_mid_const(nn.Module):
    def __init__(self, configs):
        super(Peri_mid_const, self).__init__()
        self.k = configs.top_k
        self.projection = nn.Linear(configs.seq_len, configs.d_model)

    def forward(self, x):
        B, L, C = x.size()
        period_list = FFT_for_Period(x, self.k)

        levels = []
        components_per_level = []
        for i in range(len(period_list)):
            period = period_list[i]

            if L % period != 0:
                remainder = L % period
                length = L - remainder
                x = x[:, :-remainder, :]
            else:
                length = L

            # split into components
            components_num = int(length / period)
            components_per_level.append(components_num)
            components_size = int(length // components_num)
            components = [x[:, i * components_size:(i + 1) * components_size, :] for i in range(components_num)]

            components_uniform_size = []
            for component in components:
                component = component.permute(0, 2, 1)
                component_length = component.shape[-1]
                original_length = L

                # padding to original length
                if component_length < original_length:
                    padding_size = original_length - component_length
                    padding = torch.zeros((component.shape[:-1] + (padding_size,)), device=component.device)
                    component = torch.cat((component, padding), dim=-1)

                component = self.projection(component)
                component = component.permute(0, 2, 1)
                components_uniform_size.append(component)

            levels = levels + components_uniform_size

        peri_mid = torch.stack(levels, dim=-1)
        peri_mid = peri_mid.permute(0, 3, 1, 2)
        return peri_mid, components_per_level


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.top_k = configs.top_k
        self.peri_mid_const = Peri_mid_const(configs)
        self.feature_flows_dim = configs.feature_flows_dim
        self.peri_midformer = Encoder(configs)
        self.position_embedding = PositionEmbedding(configs)
        self.temporal_embedding = TemporalEmbedding(configs)
        self.layer = configs.layers
        self.decompsition = series_decomp(configs.moving_avg)
        self.projection_trend = nn.Linear(self.seq_len, self.label_len + self.pred_len)
        self.dropout = nn.Dropout(p=configs.dropout)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(self.d_model * self.top_k, self.label_len + self.pred_len, bias=True)

        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(self.d_model * self.top_k, self.pred_len + self.seq_len, bias=True)

        if self.task_name == 'classification':
            self.projection = nn.Linear(self.feature_flows_dim, configs.num_class)

    def forecast(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # time series decompsition
        seasonal_part, trend_part = self.decompsition(x_enc)
        enc_in = seasonal_part

        # temporal embedding
        if x_mark_enc is not None:
            enc_in = self.temporal_embedding(enc_in, x_mark_enc)

        # Mapping trend part to target length
        trend_part = trend_part.permute(0, 2, 1)
        trend_part = self.projection_trend(trend_part)
        trend_part = trend_part.permute(0, 2, 1)

        enc_in, components_per_level = self.peri_mid_const(enc_in)

        enc_in = self.position_embedding(enc_in)

        enc_out = self.peri_midformer(enc_in, components_per_level, self.configs.task_name)

        enc_out_dim = enc_out.shape[-1]
        target_dim = self.d_model * self.top_k
        if enc_out_dim < target_dim:
            padding_size = target_dim - enc_out_dim
            padding = torch.zeros((enc_out.shape[:-1] + (padding_size,)), device=enc_out.device)
            enc_out = torch.cat((enc_out, padding), dim=-1)

        periodic_feature_flows = self.projection(enc_out)
        periodic_feature_flows_aggration = torch.mean(periodic_feature_flows, dim=-2)
        dec_out = periodic_feature_flows_aggration.permute(0, 2, 1)

        # add trend part
        dec_out = dec_out + trend_part

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.configs.label_len + self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.configs.label_len + self.pred_len, 1))

        return dec_out

    def imputation(self, x_enc, x_mark_enc):

        #  pre-interpolation
        x_enc_np = x_enc.detach().cpu().numpy()
        zero_indices = np.where(x_enc_np[:, :, :] == 0)
        interpolated_x_enc = np.copy(x_enc_np)
        for sample_idx, time_idx, channel_idx in zip(*zero_indices):
            non_zero_indices = np.nonzero(x_enc_np[sample_idx, :, channel_idx])[0]
            before_non_zero_idx = non_zero_indices[non_zero_indices < time_idx][-1] if len(
                non_zero_indices[non_zero_indices < time_idx]) > 0 else None
            after_non_zero_idx = non_zero_indices[non_zero_indices > time_idx][0] if len(
                non_zero_indices[non_zero_indices > time_idx]) > 0 else None
            if before_non_zero_idx is not None and after_non_zero_idx is not None:
                interpolated_value = (x_enc_np[sample_idx, before_non_zero_idx, channel_idx] + x_enc_np[
                    sample_idx, after_non_zero_idx, channel_idx]) / 2
            elif before_non_zero_idx is None:
                interpolated_value = x_enc_np[sample_idx, after_non_zero_idx, channel_idx]
            elif after_non_zero_idx is None:
                interpolated_value = x_enc_np[sample_idx, before_non_zero_idx, channel_idx]
            interpolated_x_enc[sample_idx, time_idx, channel_idx] = interpolated_value
        x_enc = torch.tensor(interpolated_x_enc).to(x_enc.device)

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # time series decompsition
        seasonal_part, trend_part = self.decompsition(x_enc)
        enc_in = seasonal_part

        # temporal embedding
        if x_mark_enc is not None:
            enc_in = self.temporal_embedding(enc_in, x_mark_enc)

        enc_in, components_per_level = self.peri_mid_const(enc_in)

        enc_in = self.position_embedding(enc_in)

        enc_out = self.peri_midformer(enc_in, components_per_level, self.configs.task_name)

        enc_out_dim = enc_out.shape[-1]
        target_dim = self.d_model * self.top_k
        if enc_out_dim < target_dim:
            padding_size = target_dim - enc_out_dim
            padding = torch.zeros((enc_out.shape[:-1] + (padding_size,)), device=enc_out.device)
            enc_out = torch.cat((enc_out, padding), dim=-1)

        periodic_feature_flows = self.projection(enc_out)
        periodic_feature_flows_aggration = torch.mean(periodic_feature_flows, dim=-2)
        dec_out = periodic_feature_flows_aggration.permute(0, 2, 1)

        # add trend part
        dec_out = dec_out + trend_part

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))

        return dec_out

    def anomaly_detection(self, x_enc, x_mark_enc):

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # time series decompsition
        seasonal_part, trend_part = self.decompsition(x_enc)
        enc_in = seasonal_part

        # temporal embedding
        if x_mark_enc is not None:
            enc_in = self.temporal_embedding(enc_in, x_mark_enc)

        enc_in, components_per_level = self.peri_mid_const(enc_in)

        enc_in = self.position_embedding(enc_in)

        enc_out = self.peri_midformer(enc_in, components_per_level, self.configs.task_name)

        enc_out_dim = enc_out.shape[-1]
        target_dim = self.d_model * self.top_k
        if enc_out_dim < target_dim:
            padding_size = target_dim - enc_out_dim
            padding = torch.zeros((enc_out.shape[:-1] + (padding_size,)), device=enc_out.device)
            enc_out = torch.cat((enc_out, padding), dim=-1)

        periodic_feature_flows = self.projection(enc_out)
        periodic_feature_flows_aggration = torch.mean(periodic_feature_flows, dim=-2)
        dec_out = periodic_feature_flows_aggration.permute(0, 2, 1)

        # add trend part
        dec_out = dec_out + trend_part
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))

        return dec_out

    def classification(self, x_enc):

        enc_in, components_per_level = self.peri_mid_const(x_enc)
        enc_in = self.position_embedding(enc_in)
        enc_out = self.peri_midformer(enc_in, components_per_level, self.configs.task_name)
        enc_out = enc_out.reshape(enc_out.shape[0], -1)

        current_dim = enc_out.shape[-1]
        # In order to reduce the computational effort, the target dimension is set
        target_dim = self.configs.feature_flows_dim
        if current_dim < target_dim:
            padding_size = target_dim - current_dim
            padding = torch.zeros((enc_out.shape[:-1] + (padding_size,)), device=enc_out.device)
            enc_out = torch.cat((enc_out, padding), dim=-1)
        elif current_dim > target_dim:
            if enc_out.dim() == 1:
                enc_out = enc_out.unsqueeze(0)
            adaptive_pool = torch.nn.AdaptiveMaxPool1d(output_size=target_dim)
            enc_out = adaptive_pool(enc_out.unsqueeze(1)).squeeze(1)

        output = self.projection(enc_out)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc)
            return dec_out  # [B, L, C]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc)
            return dec_out  # [B, L, C]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, x_mark_enc)
            return dec_out  # [B, L, C]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
