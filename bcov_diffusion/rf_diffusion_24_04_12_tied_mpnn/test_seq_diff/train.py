import torch
from tqdm.notebook import trange, tqdm
from icecream import ic
import matplotlib.pyplot as plt
from tqdm.notebook import trange, tqdm
import numpy as np

def calc_loss(true, pred, curr, t, seq_diffuser):
    '''
        true [L]

        pred [B,L,20]

        curr [L]
    '''

    mask = torch.full_like(true, True).bool().to(pred.device)
    mask = None

    #ic(true)
    #ic(pred)
    #ic(curr)

    #ic(true.shape)
    #ic(pred.shape)
    #ic(curr.shape)

    #ic('Discrete Loss')
    '''
        Expects:
        x_t [B,L]
        x_0 [B,L]
        p_logit_x0 [B,L,20]
    '''

    x_t         = torch.argmax(curr, dim=-1)[None].long().to(pred.device)
    x_0         = true[None].long().to(pred.device)
    p_logit_x_0 = pred[:, :,:20]
    #ic(x_0, x_t, p_logit_x_0.argmax(-1))
    #ic(p_logit_x_0.shape)

    #ic(x_t.shape)
    #ic(x_0.shape)
    #ic(p_logit_x_0.shape)

    return seq_diffuser.loss(x_t=x_t, x_0=x_0, p_logit_x_0=p_logit_x_0, t=int(t), diffusion_mask=mask)

    #ic(loss)

    #return loss

def calc_bit_seq_similarity(true, pred):
    '''
        true [L]
        pred [B,L,21]
    '''
    L = true.shape[0]

    # For bit sequence
    int_pred = torch.argmax(pred, dim=-1)

    sim = int(torch.sum(int_pred==true)) / int(L)

    return sim

def calc_seq_similarity(true, pred):
    '''
        true [L]
        pred [B,L,21]
    '''
    L = true.shape[0]

    # For bit sequence
    # int_pred = torch.argmax(pred, dim=-1)

    #ic(pred.shape)
    pred = pred[0,:,:20]
    probs = torch.nn.Softmax(dim=1)(pred)
    #ic(probs)
    int_pred = torch.multinomial(probs, num_samples=1).squeeze(-1)

    sim = int(torch.sum(int_pred==true)) / int(L)

    #ic(probs)
    #ic(probs[0])
    #ic(int_pred)
    #ic(true)

    return sim


def train_cycle(model, dataset, seq_diffuser, optimizer, n_epoch, device='cpu', t_weighting=None, use_optimizer=True):
    if t_weighting is None:
        t_weighting = np.ones(seq_diffuser.T) / seq_diffuser.T
    
    losses = []
    losses_aux = []
    losses_vb = []
    similarity = []
    pseq0s = []
    ts = []
    samples = []
    for epoch in trange(n_epoch):

        for sample in dataset[torch.randperm(len(dataset))]:
            samples.append(sample)

            
            # t        = torch.randint(1,seq_diffuser.T+1, (1,))
            t = np.random.choice(np.arange(1, seq_diffuser.T+1), p=t_weighting)
            t = torch.tensor([t])
            ts.append(t)
            
            seq_pert, _ = seq_diffuser.diffuse_sequence(sample, t_list=[t]) # [n,L,20]
            
            L = sample.shape[0]
            if not seq_diffuser.continuous_seq():
                seq_pert = torch.nn.functional.one_hot(seq_pert, num_classes=seq_diffuser.K)


            seq_pert = seq_pert.squeeze() # [L,20]

            # t = torch.tensor(t).to(device)
            seq_pert = seq_pert.to(device).float()

            idx = torch.arange(L).float().to(device)

            pseq0 = model(seq_pert, t, idx)
            pseq0s.append(pseq0.detach().clone())

            if seq_diffuser.continuous_seq():
                loss = calc_bit_loss(true=sample, pred=pseq0)
                seq_similarity = calc_bit_seq_similarity(true=sample, pred=pseq0)
            else:
                loss, loss_aux, loss_vb = calc_loss(true=sample, pred=pseq0, curr=seq_pert, seq_diffuser=seq_diffuser, t=t)
                seq_similarity = calc_seq_similarity(true=sample, pred=pseq0)
                similarity.append(seq_similarity)
                losses.append(loss.detach())
                losses_aux.append(loss_aux.detach())
                losses_vb.append(loss_vb.detach())

                
            if use_optimizer:
                loss.backward()

        if use_optimizer:
            optimizer.step()
            optimizer.zero_grad()
    return torch.tensor(losses), torch.tensor(losses_aux), torch.tensor(losses_vb), torch.tensor(similarity), torch.stack(pseq0s), torch.stack(samples), torch.stack(ts)

def plot_run(losses, losses_aux, losses_vb, similarity, pseq0s, dataset, show_logits_per_residue=True):
    plt.figure()
    plt.plot(losses, label='loss')
    plt.plot(losses_aux, label='loss_aux')
    plt.plot(losses_vb, label='loss_vb')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.legend()

    plt.figure()
    plt.plot(similarity)
    plt.xlabel('step')
    plt.ylabel('sim')

    preds = pseq0s[:,0,:,:].argmax(dim=2)
    K = pseq0s.shape[-1]
    L = len(dataset[0])
    if show_logits_per_residue:
        for i in range(L):
            plt.figure()
            # lines = plt.plot(pseq0s[:50,0,i,:].numpy())
            for j in range(K):
                lines = plt.plot(pseq0s[:,0,i,j].numpy(),
                                alpha=1 if dataset[0][i] == j else 0.1)
            plt.title(f'residue: {i} logits')
            # plt.legend(torch.arange(20).numpy(), )

        print(f'first pred: {preds[0]}')
        print(f'final pred: {preds[-1]}')


