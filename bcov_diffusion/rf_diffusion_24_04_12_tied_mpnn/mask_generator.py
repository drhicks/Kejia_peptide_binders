import random
import sys

import torch
import kinematics
import numpy as np
from icecream import ic


#####################################
# Misc functions for mask generation
#####################################

def get_masks(L, min_length, max_length, min_flank, max_flank):
    """
    Makes a random contiguous mask, with (or without) flanking residues masked.
    """
    flank_width = random.randint(min_flank, max_flank)
    max_length = min([max_length, L-2*flank_width - 20]) #require at least 20 visible residues in any masking regime.
    central_width = random.randint(min_length,max_length)
    assert central_width > min_length - 1
    assert max_length > min_length
    
    start = random.randint(flank_width,L-flank_width-central_width-1)
    return (start,start+central_width),flank_width


def get_diffusion_pos(L,min_length, max_length=None):
    """
    Random contiguous mask generation to denote which residues are being diffused 
    and which are not. 

    TODO: This does not support multi-chain diffusion training at the moment 

    Returns:

        start,end : indices between which residues are allowed to be diffused. 
                    Otherwise, residues are held fixed and revealed 
    """
    if (max_length is None) or (max_length > L):
        max_length = L 

    assert min_length <= max_length 

    # choose a length to crop 
    chosen_length = np.random.randint(min_length, max_length)

    # choose a start position - between 0 (inclusive) and L-chosen_length (exclusive)
    start_idx = random.randint(0, L-chosen_length)
    end_idx   = start_idx + chosen_length

    return start_idx, end_idx 

def get_cb_distogram(xyz):
    Cb = kinematics.get_Cb(xyz)
    dist = kinematics.get_pair_dist(Cb, Cb)
    return dist

def get_contacts(xyz, xyz_less_than=5, seq_dist_greater_than=10):
    L = xyz.shape[0]
    dist = get_cb_distogram(xyz)

    is_close_xyz = dist < xyz_less_than

    idx = torch.ones_like(dist).nonzero()
    seq_dist = torch.abs(torch.arange(L)[None] - torch.arange(L)[:,None])
    is_far_seq = torch.abs(seq_dist) > seq_dist_greater_than

    contacts = is_far_seq * is_close_xyz
    return contacts

def sample_around_contact(L, indices, len_low, len_high):
    diffusion_mask = torch.zeros(L).bool()
    for anchor in indices:
        mask_length = int(np.floor(random.uniform(len_low, len_high)))
        l = anchor - mask_length // 2
        r = anchor + (mask_length - mask_length//2)
        l = max(0, l)
        r = min(r, L)
        diffusion_mask[l:r] = True
    return diffusion_mask

def get_double_contact(xyz, full_prop, low_prop, high_prop, broken_prop, xyz_less_than=5, seq_dist_greater_than=25, len_low=5, len_high=10):
    contacts = get_contacts(xyz, xyz_less_than, seq_dist_greater_than)
    if not contacts.any():
        return get_diffusion_mask_simple(xyz, full_prop, low_prop, high_prop, broken_prop)
    contact_idxs = contacts.nonzero()
    contact_idx = np.random.choice(np.arange(len(contact_idxs)))
    indices = contact_idxs[contact_idx]
    L = xyz.shape[0]
    return sample_around_contact(L, indices, len_low, len_high)

def find_third_contact(contacts):
    contact_idxs = contacts.nonzero()
    contact_idxs = contact_idxs[torch.randperm(len(contact_idxs))]
    for i,j in contact_idxs:
        if j < i:
            continue
        K = (contacts[i,:] * contacts[j,:]).nonzero()
        if len(K):
            K = K[torch.randperm(len(K))]
            for k in K:
                return torch.tensor([i,j,k])
    return None

def get_triple_contact(xyz, full_prop, low_prop, high_prop, broken_prop, xyz_less_than=6, seq_dist_greater_than=10, len_low=1, len_high=3):
    contacts = get_contacts(xyz, xyz_less_than, seq_dist_greater_than)
    indices = find_third_contact(contacts)
    if indices is None:
        return get_diffusion_mask_simple(xyz, full_prop, low_prop, high_prop, broken_prop)
    L = xyz.shape[0]
    return sample_around_contact(L, indices, len_low, len_high)

def get_diffusion_mask_simple(xyz, full_prop, low_prop, high_prop, broken_prop):
    """
    Function to make a diffusion mask.
    Options:
        full_prop - proportion of time whole protein is masked.
        low_prop - lower bound on the proportion of the protein masked
        high_prop - upper bound on the proportion of the protein masked
        broken_prop - proportion of the time the mask is in the middle (broken motif), vs at the ends
    Output:
        1D diffusion mask. True is unmasked, False is masked/diffused
    """
    L = xyz.shape[0]
    diffusion_mask = torch.ones(L).bool()
    mask_length = int(np.floor(random.uniform(low_prop, high_prop) * L))
    # decide if mask goes in the middle or the ends
    if random.uniform(0,1) < broken_prop:
        high_start = L-mask_length-1
        start = random.randint(0, high_start)
        diffusion_mask[start:start+mask_length] = False
    else:
        # split mask in two
        split = random.randint(1, mask_length-2)
        diffusion_mask[:split] = False
        diffusion_mask[-(mask_length-split):] = False
    return diffusion_mask

def get_diffusion_mask(
        xyz, full_prop, low_prop, high_prop, broken_prop,
        diff_mask_probs):
    
    L = xyz.shape[0]
    if random.uniform(0,1) < full_prop:
        return torch.zeros(L).bool()

    mask_probs = list(diff_mask_probs.items())
    masks = [m for m, _ in mask_probs]
    props = [p for _, p in mask_probs]
    get_mask = np.random.choice(masks, p=props)
    return get_mask(xyz, full_prop, low_prop, high_prop, broken_prop)


#####################################
# Main mask generator function
#####################################

def generate_masks(msa, task, loader_params, chosen_dataset, full_chain=None, xyz=None): #full_chain is for complexes, to signify which chain is complete
    '''
    Slimmed down function that outputs 1D masks for inputs and loss calculations.
    Input masks are defined as True=(unmasked)/False=masked (except for input_t1dconf, which is a scalar value, and seq2str_mask which is the msa mask for the seq2str task)
    Loss masks are defined as True=(loss applied)/False=(no loss applied)
    
    Input masks:
        -input_seq
        -input_str
        -input_floating = points to be represented as floating points (structure present but side chains masked out)
        -input_t1d_str_conf = scalar to multiply input str t1d confidences by
        -input_t1d_seq_conf = scalar to multiply input seq t1d confidences by

    Output masks:
        -loss_seq
        -loss_str
        -loss_str_2d = additional coordinate pair masking to be applied on top of loss_str 1d masking.
    '''

    L = msa.size()[2]
    input_seq_mask = torch.ones(L).bool()
    input_str_mask = torch.ones(L).bool()
    input_floating_mask = -1
    input_t1d_str_conf_mask = torch.ones(L).bool() * 0.9
    input_t1d_seq_conf_mask = torch.ones(L).bool() * 0.9
    loss_seq_mask = torch.ones(L).bool()
    loss_str_mask = torch.ones(L).bool()
    loss_str_mask_2d = torch.ones(L,L).bool()

    if task == 'seq2str':
        '''
        Classic structure prediction task.
        '''
        #input masks
        # Currently msa loss masking is performed in train_multi_EMA
        #input_seq_mask = torch.clone(seq2str_mask) #this is not 1D
        input_str_mask = torch.ones(L).bool()
        input_floating_mask = torch.ones(L).bool()
        input_t1d_str_conf_mask = torch.ones(L)*0.9 #scale seq2str t1d confidences by 0.9
        input_t1d_seq_conf_mask = torch.ones(L) # Very confident about the true sequence

        #loss masks
        # Currently msa loss masking is performed in train_multi_EMA
        # loss_seq_mask = torch.clone(seq2str_mask) #this is not 1D
        #loss_str_mask = seq2str_str_mask
        #loss_str_mask_2d = seq2_str_mask[None, :] * seq2str_str_mask[:, None]

    # dj - only perform diffusion hal on pdb and fb for now 
    elif task == 'diff' and chosen_dataset not in ['complex','negative']:
        """
        Hal task but created for the diffusion-based training. 
        """ 

        # get start and end of contiguous section of diffused residues 
        #start, end = get_diffusion_pos(L, loader_params['DIFF_MASK_LOW'], loader_params['DIFF_MASK_HIGH'])
        
        ## input masks
        # False is diffused, True is not diffused 
        #input_str_mask[start:end+1] = False 
        #input_seq_mask     = torch.clone(input_str_mask)
        
        #MADE A NEW FUNCTION
        diffusion_mask = get_diffusion_mask(
            xyz,
            full_prop=loader_params['P_UNCOND'],
            low_prop=loader_params['MASK_MIN_PROPORTION'],
            high_prop=loader_params['MASK_MAX_PROPORTION'],
            broken_prop=loader_params['MASK_BROKEN_PROPORTION'],
            diff_mask_probs=loader_params['DIFF_MASK_PROBS'],
            ) 
        input_str_mask = diffusion_mask
        input_seq_mask = diffusion_mask
        # t1dconf scaling will be taken care of by diffuser, so just leave those at 1 here 
        input_t1d_str_conf_mask = torch.ones(L)
        input_t1d_seq_conf_mask = torch.ones(L)

        ## loss masks 
        loss_seq_mask[diffusion_mask] = False  # Dont score where diffusion mask is True (i.e., where things are not diffused)

    elif task == 'diff' and chosen_dataset == 'complex':
        '''
        Diffusion task for complexes. Default is to diffuse the whole of the complete chain.
        Takes full_chain as input, which is [full_chain_start_idx, full_chain_end_idx]
        '''
        assert full_chain[1] is not None
        
        input_str_mask = torch.clone(full_chain)
        input_seq_mask = torch.clone(input_str_mask)

    elif task == 'hal' and chosen_dataset != 'complex':
        '''
        This is Joe's hallucination task, where a contiguous region is masked, along with flanks, and two residues either end are given (but not their angles).
        Scored on everything but the flank regions (may want to change this to only score central inpainted region
        '''
        splice, flank_width = get_masks(L, loader_params['HAL_MASK_LOW'], loader_params['HAL_MASK_HIGH'], loader_params['FLANK_LOW'], loader_params['FLANK_HIGH'])
        hal_mask_len = splice[1]-splice[0]

        #input masks
        input_seq_mask = torch.ones(L).bool()
        input_seq_mask[splice[0]-flank_width:splice[1]+flank_width] = False #mask out flanks and central region
        input_str_mask = torch.clone(input_seq_mask)
        input_str_mask[splice[0]-1] = True #give structure of two flanking residues
        input_str_mask[splice[1]] = True

        input_floating_mask = torch.ones(L).bool()
        input_floating_mask[splice[0]-1] = False #immediate two flanking residues are set to false/floating
        input_floating_mask[splice[1]] = False
        input_t1d_str_conf_mask = torch.ones(L) #t1d confidences are unscaled in hal task
        input_t1d_seq_conf_mask = torch.ones(L) #t1d confidences are unscaled in hal task

        #loss masks
        loss_seq_mask = torch.zeros(L).bool()
        loss_seq_mask[splice[0]:splice[1]] = True #only apply a loss on the central region
        loss_str_mask = torch.ones(L).bool()
        loss_str_mask[splice[0]-flank_width:splice[0]] = False #don't apply a loss in the flanking regions
        loss_str_mask[splice[1]:splice[1] + flank_width] = False
        loss_str_mask_2d[~loss_str_mask,:] = False
        loss_str_mask_2d[:,~loss_str_mask] = False

    elif task == 'hal_ar' and chosen_dataset != 'complex':
        '''
        This is Joe's hallucination task, where a contiguous region is masked, along with flanks, and two residues either end are given (but not their angles).
        This is autoregressive sequence unmasking.
        Scored on everything but the flank regions (may want to change this to only score central inpainted region
        '''
        splice, flank_width = get_masks(L, loader_params['HAL_MASK_LOW'], loader_params['HAL_MASK_HIGH'], loader_params['FLANK_LOW'], loader_params['FLANK_HIGH'])
        hal_mask_len = splice[1]-splice[0]

        to_unmask = random.uniform(0,0.5) #up to 50% of sequence unmasked
        #input masks
        input_seq_mask = torch.where(torch.rand(L) < to_unmask, True, False)
        input_seq_mask[:splice[0]-flank_width] = True
        input_seq_mask[splice[1]+flank_width:] = True
        input_seq_mask[splice[0]-flank_width:splice[0]] = False #mask out flanks
        input_seq_mask[splice[1]:splice[1]+flank_width] = False
        
        input_str_mask[splice[0]-flank_width:splice[1]+flank_width] = False
        input_str_mask[splice[0]-1] = True #give structure of two flanking residues
        input_str_mask[splice[1]] = True

        input_floating_mask = torch.ones(L).bool()
        input_floating_mask[splice[0]-1] = False #immediate two flanking residues are set to false/floating
        input_floating_mask[splice[1]] = False
        input_t1d_str_conf_mask = torch.ones(L) #t1d confidences are unscaled in hal task
        input_t1d_seq_conf_mask = torch.ones(L) #t1d confidences are unscaled in hal task

        #loss masks
        loss_seq_mask = torch.zeros(L).bool()
        loss_seq_mask[splice[0]:splice[1]] = True #only apply a loss on the central region
        loss_str_mask = torch.ones(L).bool()
        loss_str_mask[splice[0]-flank_width:splice[0]] = False #don't apply a loss in the flanking regions
        loss_str_mask[splice[1]:splice[1] + flank_width] = False
        loss_str_mask_2d[~loss_str_mask,:] = False
        loss_str_mask_2d[:,~loss_str_mask] = False

    elif task == 'hal' and chosen_dataset == 'complex':
        ''' 
        This is Joe's complex hal task, where a contiguous region is masked.
        Everything is scored.
        This is for complexes
        '''
        len_full_chain = full_chain[1]-full_chain[0]+1 #full_chain has start and end index
        high_limit = min([loader_params['COMPLEX_HAL_MASK_HIGH'],len_full_chain-20])
        low_limit = min([loader_params['COMPLEX_HAL_MASK_LOW'],high_limit])
        len_to_mask = random.randint(low_limit, high_limit)
        start = random.randint(full_chain[0], len_full_chain-len_to_mask + full_chain[0])
        #input masks
        input_seq_mask = torch.ones(L).bool()
        input_seq_mask[start:start+len_to_mask] = False
        input_str_mask = torch.ones(L).bool()
        input_str_mask[start:start+len_to_mask] = False #not doing flanking masking now
        input_t1d_str_conf_mask = torch.ones(L) #t1d confidences are unscaled in str2seq task
        input_t1d_seq_conf_mask = torch.ones(L) #t1d confidences are unscaled in str2seq task

        #loss masks
        loss_seq_mask = torch.zeros(L).bool()
        loss_seq_mask[start:start+len_to_mask] = True
        loss_str_mask = torch.ones(L).bool()
        loss_str_mask_2d[~loss_str_mask,:] = False
        loss_str_mask_2d[:,~loss_str_mask] = False
   
    elif task == 'hal_ar' and chosen_dataset == 'complex':
        ''' 
        This is Joe's complex hal task, where a contiguous region is masked, but with some sequence tokens visible (to mimic autoregressive unmasking).
        Everything is scored.
        This is for complexes
        '''
        len_full_chain = full_chain[1]-full_chain[0]+1 #full_chain has start and end index
        high_limit = np.min([loader_params['COMPLEX_HAL_MASK_HIGH_AR'],len_full_chain-20])
        low_limit=np.min([loader_params['COMPLEX_HAL_MASK_LOW_AR'],high_limit])
        len_to_mask = random.randint(low_limit, high_limit)
        start = random.randint(full_chain[0], len_full_chain-len_to_mask + full_chain[0])
 
        to_unmask = random.uniform(0,0.5) #up to 50% of sequence unmasked
        #input masks
        input_seq_mask = torch.where(torch.rand(L) < to_unmask, True, False)
        input_seq_mask[:start] = True
        input_seq_mask[start+len_to_mask:] = True
        input_str_mask = torch.ones(L).bool()
        input_str_mask[start:start+len_to_mask] = False #not doing flanking masking now. No AR unmasking of structure
        input_t1d_str_conf_mask = torch.ones(L) #t1d confidences are unscaled in str2seq task
        input_t1d_seq_conf_mask = torch.ones(L) #t1d confidences are unscaled in str2seq task

        #loss masks
        loss_seq_mask = torch.zeros(L).bool()
        loss_seq_mask[start:start+len_to_mask] = True
        loss_str_mask = torch.ones(L).bool()
        loss_str_mask_2d[~loss_str_mask,:] = False
        loss_str_mask_2d[:,~loss_str_mask] = False

    elif task == 'str2seq' and chosen_dataset != 'complex':
        '''
        This is Joe's str2seq task, where a contiguous region is masked, along with flanks.
        Everything, but the flanked regions, is scored
        This is only if the protein is monomeric
        '''
        splice, flank_width = get_masks(L, loader_params['HAL_MASK_LOW'], loader_params['HAL_MASK_HIGH'], loader_params['FLANK_LOW'], loader_params['FLANK_HIGH'])
        hal_mask_len = splice[1]-splice[0]
        #input masks
        input_seq_mask = torch.ones(L).bool()
        input_seq_mask[splice[0]-flank_width:splice[1]+flank_width] = False #mask out flanks and central region
        input_str_mask = torch.ones(L).bool()
        input_str_mask[splice[0] - flank_width:splice[0]] = False #mask out flanks only (i.e. provide structure of the central region)
        input_str_mask[splice[1]:splice[1] + flank_width] = False
        input_floating_mask = torch.ones(L).bool()
        input_t1d_str_conf_mask = torch.ones(L) #t1d confidences are unscaled in str2seq task
        input_t1d_seq_conf_mask = torch.ones(L) #t1d confidences are unscaled in str2seq task

        #loss masks
        loss_seq_mask = torch.zeros(L).bool()
        loss_seq_mask[splice[0]:splice[1]] = True #only apply a loss on sequence recovery in the central region
        loss_str_mask = torch.clone(input_str_mask) #don't apply a structure loss on the flanking regions
        loss_str_mask_2d[~loss_str_mask,:] = False
        loss_str_mask_2d[:,~loss_str_mask] = False 
    
    elif task == 'str2seq' and chosen_dataset == 'complex':
        ''' 
        This is Joe's str2seq task, where a contiguous region is masked.
        Everything is scored
        This is for complexes
        '''
        len_full_chain = full_chain[1]-full_chain[0]+1 #full_chain has start and end index
        high_limit = np.min([loader_params['COMPLEX_HAL_MASK_HIGH'],len_full_chain-20])
        low_limit=np.min([loader_params['COMPLEX_HAL_MASK_LOW'],high_limit])
        len_to_mask = random.randint(low_limit, high_limit)
        start = random.randint(full_chain[0], len_full_chain-len_to_mask + full_chain[0])

        #input masks
        input_seq_mask = torch.ones(L).bool()
        input_seq_mask[start:start+len_to_mask] = False
        input_str_mask = torch.ones(L).bool()
        input_t1d_str_conf_mask = torch.ones(L) #t1d confidences are unscaled in str2seq task
        input_t1d_seq_conf_mask = torch.ones(L) #t1d confidences are unscaled in str2seq task
 
        #loss masks
        loss_seq_mask = torch.zeros(L).bool()
        loss_seq_mask[start:start+len_to_mask] = True
        loss_str_mask = torch.clone(input_str_mask)
        loss_str_mask_2d[~loss_str_mask,:] = False
        loss_str_mask_2d[:,~loss_str_mask] = False

    elif task == 'str2seq_full':
        '''
        This is David's str2seq task, where most (default 90-100%) of sequence is masked.
        Score on everything.
        '''
        #NOTE this has not been implemented yet for e.g. complexes
        rand_prop = random.uniform(loader_params['STR2SEQ_FULL_LOW'], loader_params['STR2SEQ_FULL_HIGH']) #random proportion masked, between two extremes
        
        #input masks
        input_seq_mask = torch.rand(L).bool() < rand_prop
        input_str_mask = torch.ones(L).bool()
        input_floating_mask = torch.ones(L).bool()

        input_t1d_str_conf_mask = torch.ones(L) #t1d confidences are not scaled in str2seq_full task
        input_t1d_seq_conf_mask = torch.ones(L) #t1d confidences are not scaled in str2seq_full task

        #loss masks
        loss_seq_mask = torch.clone(input_seq_mask)
        loss_str_mask = torch.ones(L).bool() #apply a loss on the whole structure        
        loss_str_mask_2d[~loss_str_mask,:] = False
        loss_str_mask_2d[:,~loss_str_mask] = False
 
    else:
        sys.exit(f'Masks cannot be generated for the {task} task!')
    if task != 'seq2str':
        assert torch.sum(~input_seq_mask) > 0, f'Task = {task}, dataset = {chosen_dataset}, full chain = {full_chain}'
    mask_dict = {'input_seq_mask':input_seq_mask,
                'input_str_mask':input_str_mask,
                'input_floating_mask':input_floating_mask,
                'input_t1d_str_conf_mask':input_t1d_str_conf_mask,
                'input_t1d_seq_conf_mask':input_t1d_seq_conf_mask,
                'loss_seq_mask':loss_seq_mask,
                'loss_str_mask':loss_str_mask,
                'loss_str_mask_2d':loss_str_mask_2d}
    
    return mask_dict

