import torch
from matplotlib import pyplot as plt


def assert_converges(diffuser, L=40):
    # May only be a decent sanity check for uniform rate matrices.
    # seq = torch.arange(0, diffuser.K)
    seq = torch.zeros(L).long()
    f_same = []
    for i in range(100):
        diffused_seq, true_seq = diffuser.diffuse_sequence(seq, diffusion_mask=[])
        # Make sure that this is close to 1/20
        fraction_same = (diffused_seq[-1] == true_seq).float().mean()
        f_same.append(fraction_same)
    mean_f_same = torch.mean(torch.tensor(f_same)).item()
    expected = 1/diffuser.K
    msg = f'fraction same: {mean_f_same:.3f}, expected: {expected:.3f}'
    print(msg)
    assert(abs(mean_f_same - expected) <= 0.02)

    plt.figure()
    plt.imshow(diffused_seq)
    plt.title(f'fraction same: {fraction_same:.2f}')


