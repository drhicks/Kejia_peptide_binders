import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic

class SeqDiffNetSimple(nn.Module):
    def __init__(self, K=None, L=None,T=None, d_hidden=64, n_hidden=16):
        super(SeqDiffNetSimple, self).__init__()
        self.T = T
        self.L = L
        self.K = K

        input_len = L * K + 1
        self.emb      = nn.Linear(input_len, d_hidden)

        self.linears = nn.ModuleList([nn.Linear(d_hidden, d_hidden) for i in range(n_hidden)])
        self.pred_seq = nn.Linear(d_hidden, L * K)

    def forward(self, seq, t, idx):
        '''
        Args:
            seq (torch.tensor) [L,20]

            t (torch.tensor) one indexed index of current timestep

            idx (torch.tensor) [L] Integer of each positional index

        Returns:
            pseq (torch.tensor) [1, L,20]

        '''
        t = torch.tensor([t.item()])
        #ic(seq.dtype)
        seq_flat = seq.reshape(-1)
        #ic(seq_flat.shape, t.shape)
        seq_flat_t = torch.cat([seq_flat, t])
        #ic(seq_flat_t.shape)
        seq      = self.emb(seq_flat_t) # [L * K + 1]

        for l in self.linears:
            seq = seq + F.relu_(l(seq))

        pred_seq = self.pred_seq(seq) # [L * K]
        pred_seq_reshaped = pred_seq.reshape((self.L, self.K))

        return pred_seq_reshaped[None] # [1, L, K]



