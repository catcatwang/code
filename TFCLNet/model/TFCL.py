import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AttentionLayer
from .embed import DataEmbedding, PositionalEmbedding
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import numpy as np
import pywt
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer


    def forward(self, x, attn_mask=None):
        # x [B, T, D]
        attlist = []
        for attn_layer in self.attn_layers:
            x, _ = attn_layer(x)
            attlist.append(_)

        if self.norm is not None:
            x = self.norm(x)

        return x, attlist

class Dncoder(nn.Module):
    def __init__(self, attn_layers, d_model=128, d_ff=None, dropout=0.1, activation="relu"):
        super(Dncoder, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu


    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attlist = []
        for attn_layer in self.attn_layers:
            x, _ = attn_layer(x)
            attlist.append(_)
        x = x + self.dropout(x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attlist



class FreEnc(nn.Module):
    def __init__(self, c_in, c_out, d_model, e_layers, win_size, fr):
        super(FreEnc, self).__init__()

        self.emb = DataEmbedding(c_in, d_model)

        self.enc = Encoder(
            [
                    AttentionLayer(d_model) for l in range(e_layers)
            ]
            , norm_layer=nn.LayerNorm(d_model)
        )
        self.dec = Dncoder(
            [
                AttentionLayer(d_model) for l in range(e_layers)
            ]
        )
        self.linear = nn.Linear(d_model,d_model)
        self.pro = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.window_size = 7

        self.mask_token = nn.Parameter(torch.zeros(1,d_model,1, dtype=torch.cfloat))

        self.fr = fr
        self.threshold_value = 0.3
#train
    # SMAP 0.5 0.6
    # SWat 0.2 0.4
    # PSM  0.4 0.2
    # SMD  0.6 0.4
    # MSL  0.8 0.1

    def forward(self, x):
        # x: [B, T, C]
        ex = self.emb(x) # [B, T, D]
        #converting to frequency domain and calculating the mag
        cx = torch.fft.rfft(ex.transpose(1,2))
        mag = torch.sqrt(cx.real ** 2 + cx.imag ** 2) # [B, D, Mag]
        log_mag = torch.log1p(mag)
        idx = torch.argwhere(log_mag < self.threshold_value)
        cx[log_mag < self.threshold_value] = self.mask_token.repeat(ex.shape[0], 1, mag.shape[-1])[
            idx[:, 0], idx[:, 1], idx[:, 2]]
        # converting to time domain
        ix = torch.fft.irfft(cx).transpose(1, 2)
        # encoding tokens
        dx, att = self.enc(ix)
        dxx, attone = self.dec(dx)
        rec = self.pro(dxx)
        att.append(rec)

        return att  # att(list): [B, T, T]



class TemEnc(nn.Module):
    def __init__(self, c_in, c_out, d_model, e_layers, win_size, seq_size, tr):
        super(TemEnc, self).__init__()

        self.emb = DataEmbedding(c_in, d_model)
        self.pos_emb = PositionalEmbedding(d_model)
        self.enc = Encoder(
            [
                    AttentionLayer(d_model) for l in range(e_layers)
            ]
            ,norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.dec = Dncoder(
            [
                    AttentionLayer( d_model) for l in range(e_layers)
            ]
        )
        self.attn = AttentionLayer(d_model)
        self.pro = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.mask_token = nn.Parameter(torch.zeros(1,1,d_model))
        #self.tr = int(tr * win_size)
        self.tr = int(0.4 * win_size)
        self.seq_size = seq_size


    def forward(self, x):

        # x: [B, T, C]
        ex = self.emb(x) # [B, T, D]
        filters = torch.ones(1,1,self.seq_size).to(device)

        diff = torch.abs(ex[:, 1:, :] - ex[:, :-1, :])
        scoree = torch.cat([diff.mean(dim=-1), torch.zeros(ex.size(0), 1, device=ex.device)], dim=1)
        score = 1 / scoree

        # mask time points
        masked_idx, unmasked_idx = score.topk(self.tr, dim=1, sorted=False)[1], (-1*score).topk(x.shape[1]-self.tr, dim=1, sorted=False)[1]
        unmasked_tokens = ex[torch.arange(ex.shape[0])[:,None],unmasked_idx,:]

        # encoding unmasked tokens and getting masked tokens
        ux, _ = self.enc(unmasked_tokens)
        masked_tokens = self.mask_token.repeat(ex.shape[0], masked_idx.shape[1], 1) + self.pos_emb(idx = masked_idx)

        tokens = torch.zeros(ex.shape,device=device)

        tokens[torch.arange(ex.shape[0])[:,None],unmasked_idx,:] = ux
        tokens[torch.arange(ex.shape[0])[:,None],masked_idx,:] = masked_tokens

        # decoding tokens
        dx, att = self.dec(tokens)
        rec = self.pro(dx)
        att.append(rec)

        return att # att(list): [B, T, T]


class TFC(nn.Module):
    def __init__(self, win_size, seq_size, c_in, c_out, d_model=512, e_layers=3, fr=0.4, tr=0.5, dev=None):
        super(TFC, self).__init__()
        global device
        device = dev
        self.win_size = win_size
        self.tem = TemEnc(c_in, c_out, d_model, e_layers, win_size, seq_size, tr)
        self.fre = FreEnc(c_in, c_out, d_model, e_layers, win_size, fr)

    def forward(self, x):
        tematt = self.tem(x) # tematt: [B, T, T]
        freatt = self.fre(x) # freatt: [B, T, T]
        return tematt, freatt