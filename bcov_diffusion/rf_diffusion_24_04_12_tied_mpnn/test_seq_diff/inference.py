import torch
from tqdm.notebook import trange, tqdm
import numpy as np

def run_inference(model, seq_diffuser, L, device='cpu'): 
    x_t = seq_diffuser.get_pi(L)
    seq_stack = []

    for t in trange(seq_diffuser.T,0,-1):

        idx = torch.arange(L).float().to(device)
        t_tens = torch.tensor(t).to(device)
        # t_tens = torch.tensor(t).float().to(device)

        # out = model(x_t.float(), t_tens, idx) # [L,21]
        out = model(x_t[None,...].float(), t_tens, idx)[0] # [L,21]

        if t > 1:
            x_t = seq_diffuser.get_next_sequence(seq_t=x_t, pseq0=out.cpu(), t=t, seq_diffusion_mask=torch.full((L,), False))
        else:
            x_t = out.cpu()
        
        seq = x_t.argmax(dim=-1)
        seq_stack.append(seq)
    return torch.stack(seq_stack)
