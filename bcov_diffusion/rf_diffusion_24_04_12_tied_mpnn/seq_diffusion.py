# script for sequence diffusion protocols 
import torch 
import numpy as np
import os
from scipy.spatial.transform import Rotation as scipy_R
from scipy.spatial.transform import Slerp 

from util import rigid_from_3_points, get_torsions

from util import torsion_indices as TOR_INDICES 
from util import torsion_can_flip as TOR_CAN_FLIP
from util import reference_angles as REF_ANGLES

from util_module import ComputeAllAtomCoords

from diff_util import th_min_angle, th_interpolate_angles, get_aa_schedule 
from diffusion import get_beta_schedule, cosine_interp

from chemical import INIT_CRDS 
import igso3
import time 

from icecream import ic  


torch.set_printoptions(sci_mode=False)

class DiscreteSeqDiffuser():
    '''
        Class to do discrete diffusion according to the method reported in [1]. This class will
        yield a noised sequence and also will return the "true" probability distributions at
        certain timesteps along the noising trajectory.
        [1] Austin, Jacob, et al. "Structured denoising diffusion models in discrete
        state-spaces." Advances in Neural Information Processing Systems 34 (2021):
        17981-17993.
    '''

    def __init__(self,
                 T,
                 rate_matrix,
                 s_a0=None,
                 s_aT=None,
                 s_b0=None,
                 s_bT=None,
                 schedule_type='linear',
                 schedule_params={},
                 lamda=1,
                 K=20):

        assert (not s_a0 is None and not s_aT is None) ^ (not s_b0 is None and not s_bT is None), \
                f"SequenceDiffuser cannot use alpha and beta schedules at the same time, a_0: {s_a0} a_T: {s_aT} " + \
                f"b_0: {s_b0} b_T: {s_bT}"

        self.K = K

        self.T = T
        self.rate_matrix_type = rate_matrix

        if rate_matrix in ['blosum']:
            assert not s_a0 is None and not s_aT is None

            self.seq_alpha_schedule, self.seq_alpha_bar_schedule = self.get_seq_alpha_schedule(
                    s_a0=s_a0,
                    s_aT=s_aT,
                    schedule_type=schedule_type,
                    **schedule_params)

            self.singly_stochastic_B = self.rowwise_normalize( torch.exp(torch.Tensor(blosum62)) )


        elif rate_matrix in ['uniform']:
            assert not s_b0 is None and not s_bT is None

            self.seq_beta_schedule, self.seq_alpha_schedule, self.seq_alpha_bar_schedule = self.get_seq_beta_schedule(
                    s_b0=s_b0,
                    s_bT=s_bT,
                    schedule_type=schedule_type,
                    **schedule_params)

        else: raise NotImplementedError(f'Received invalid rate matrix for SequenceDiffuser. Rate matrix {rate_matrix} is not implemented.')

        # The number of categories to choose between, we are using the 20 AAs with no gap or mask token
        self.lamda = lamda
        self.softmax = torch.nn.Softmax(dim=2)
        self.rate_matrix = None

        # A field variable that keeps track of whether we are working with a singly or doubly stochastic transition matrix
        self.singly_stochastic = rate_matrix in ['blosum']

    def continuous_seq(self):
        '''
            This type of sequence diffuser does not use a continuous sequence representation so return False.
        '''
        return False
    
    def diffuse_sequence(self,
                         seq,
                         diffusion_mask=None,
                         t_list=None):
        '''
            Given a sequence, do discrete diffusion of the sequence and return the result at the
            specified timepoints.

            Args:

                seq (torch.tensor [L], required): Torch tensor of true sequence to be noised. Integer sequence representation

                diffusion_mask (torch.tensor [L], optional): Tensor of bools, True means NOT diffused at this residue, False means diffused

                t_list (list, optional): If present, only return the diffused coordinates at timesteps t within the list

            Returns:

                diffused_seq (torch.tensor, [t,L])

                true_seq     (torch.tensor, [L])

        '''
        diffusion_mask = diffusion_mask or []

        diffused_seq = self.apply_kernel_recursive(seq, diffusion_mask, t_list) # [t,L]
        true_seq     = seq.clone() # [L]

        assert( torch.all( diffused_seq[:,diffusion_mask] == true_seq[None,diffusion_mask] ) )

        return diffused_seq, true_seq 

    def rowwise_normalize(self, B):
        ''' Given a [K,K] matrix, normalize each of the rows by the elements contained in the rows.'''
        return B / torch.sum(B, dim=1, keepdim=True)

    def get_seq_alpha_schedule(self,
                               s_a0=0.1,
                               s_aT=10,
                               schedule_type='linear'):

        '''
            The noising schedule parameter alpha is defined differently in Austin et al. from Ho et al. We are going to call
            the parameter from Austin et al. seq_alpha (or s_a, s_alpha, etc.) to avoid confusion.
        '''
        if schedule_type == 'linear':
            seq_alpha_schedule = torch.linspace(s_a0, s_aT, self.T)

        elif schedule_type == 'cosine':
            seq_alpha_schedule = cosine_interp(self.T, s_aT, s_a0)

        elif schedule_type == 'exponential':
            seq_alpha_schedule = torch.tensor(np.exp(np.linspace(np.log(s_a0), np.log(s_aT), self.T)))

        else: raise NotImplementedError(f'Received invalid schedule type for seq_alpha_schedule. Schedule {schedule_type} is not implemented.')

        seq_alpha_bar_schedule = torch.cumsum(seq_alpha_schedule, dim=0)

        return seq_alpha_schedule, seq_alpha_bar_schedule

    def get_seq_beta_schedule(self,
                              s_b0=0.0001,
                              s_bT=0.1,
                              schedule_type='linear'):

        if schedule_type == 'linear':
            seq_beta_schedule = torch.linspace(s_b0, s_bT, self.T)

        elif schedule_type == 'cosine':
            seq_beta_schedule = cosine_interp(self.T, s_bT, s_b0)

        else: raise NotImplementedError(f'Received invalid schedule type for seq_beta_schedule. Schedule {schedule_type} is not implemented.')

        seq_alpha_schedule = torch.ones_like(seq_beta_schedule) - seq_beta_schedule
        seq_alpha_bar_schedule = torch.cumprod(seq_alpha_schedule, dim=0)

        return seq_beta_schedule, seq_alpha_schedule, seq_alpha_bar_schedule

    def get_Qt(self, t_idx):

        if self.rate_matrix_type == 'blosum':

            if self.rate_matrix is None:
                # This call probably deserves its own 'prepare blosum matrix' function - NRB
                self.rate_matrix = self.singly_stochastic2rate_matrix( self.rowwise_normalize( torch.exp(torch.Tensor(blosum62)) ) )

            s_alpha_t = self.seq_alpha_schedule[t_idx]

            Qt = torch.matrix_exp(s_alpha_t*self.rate_matrix)

        elif self.rate_matrix_type == 'uniform':
            # Equation A.2.1 from Austin et al. with K = 20

            beta_t = self.seq_beta_schedule[t_idx]

            Qt = ( beta_t / self.K ) * torch.ones(self.K,self.K)

            Qt.fill_diagonal_( 1 - ((self.K-1)/self.K) * beta_t ) # 1 - (K-1)/K * beta_t

        else: raise NotImplementedError(f'Sequence diffusion with rate matrix of type {self.rate_matrix_type} is not implemented')

        return Qt

    def get_bar_Qt(self, t_idx):

        if self.rate_matrix_type == 'blosum':

            if self.rate_matrix is None:
                self.rate_matrix = self.singly_stochastic2rate_matrix( self.rowwise_normalize( torch.exp(torch.Tensor(blosum62)) ) )

            s_alpha_bar_t = self.seq_alpha_bar_schedule[t_idx]

            barQt = torch.matrix_exp(s_alpha_bar_t*self.rate_matrix)

        elif self.rate_matrix_type == 'uniform':

            s_alpha_bar_t = self.seq_alpha_bar_schedule[t_idx]

            barQt = s_alpha_bar_t*torch.eye(self.K) + (1-s_alpha_bar_t)/self.K*torch.ones([self.K,self.K])

        else: raise NotImplementedError(f'Sequence diffusion with rate matrix of type {self.rate_matrix_type} is not implemented')

        return barQt

    def apply_kernel(self, seq, t, diffusion_mask=None):
        '''
            Take the sequence at timestep = 0 and return the noised sequence at timestep = t
            Args:
                seq (torch.Tensor, [L]): Integer representation of sequence at timestep = 0
                t (int): Timestep to sample
                diffusion_mask (torch.Tensor, [L]): Mask
            Returns:
                out_seq (torch.Tensor, [L]): The noised sequence at timestep = t
        '''

        # t_idx is 0-indexed
        t_idx = t - 1

        bar_Qt = self.get_bar_Qt(t_idx) # [K,K]

        # Grab rows from Qt corresponding to the current sequence
        seq_probs_t = bar_Qt[seq] # [L,K]

        # Sample a noised sequence from the probabilities
        out_seq = torch.multinomial(seq_probs_t, num_samples=1).squeeze() # [L]

        # Denoise all positions that are not being noised
        if not diffusion_mask is None:
            out_seq[diffusion_mask] = seq[diffusion_mask]

        return out_seq

    def apply_kernel_recursive(self, seq, diffusion_mask=None, t_list=None):
        '''
            Repeatedly apply self.apply_kernel T times and return all seqs
            Args:

            Returns:
                seq_stack (torch.Tensor, [L,t]): A tensor with the integer representation of the noised sequence at each timestep. The order of dimensions
                                                 is defined like this to match the convention in Euclidean Diffuser
        '''

        # If we want to do actual recursive noising we would enter this block - NRB
        if False:
            cur_seq = torch.clone(seq)

            for t in range(1,self.T+1):
                cur_seq = self.apply_kernel(cur_seq,
                                           t,
                                           diffusion_mask)

                cur_seq.append(cur_seq)

        seq_stack = [self.apply_kernel(seq,t,diffusion_mask) for t in range(1,self.T+1)]

        seq_stack = torch.stack(seq_stack, dim=0) # [t,L]

        if t_list != None:
            t_idx = torch.tensor([t-1 for t in t_list])
            assert(t_idx>=0).all(), 'detected timestep less than 1'

            seq_stack = seq_stack[t_idx,:]

        return seq_stack

    def singly_stochastic2rate_matrix(self, A):
        """rate_matrix computes the rate matrix corresponding to a singly stochastic
        matrix A.
        Args:
            A: single stochastic matrix (e.g. Blossom probabilities) [K, K]

        Returns:
            rate matrix of shape [K, K]
        """
        R = torch.clone(A)
        for i in range(A.shape[0]):
            R[i,i] = -sum(R[i])
            R[i,i] += A[i,i]
        return R

    ########################################
    #### Functions for Loss Calculations ###
    ########################################

    def q_Xt1_given_Xt_and_X0(self, x_t, x_0, t):
        """q_Xt1_given_Xt_and_X0 computes conditional q(x_{t-1} | x_t, x_0).

        Computation of this conditional follows equation 3 of [1].

        For this, the transition matrices are defined via matrix
        exponentials as described in Appendix A.4.2 of [1].

        Args:
            x_t, x_0: index of amino acid at times t and 0
            t: time index (positive integer)

        Reference:
            [1] Austin, Jacob, et al. "Structured denoising diffusion models in discrete
            state-spaces." Advances in Neural Information Processing Systems 34 (2021):
            17981-17993.
        """
        # confirm t > 0
        assert t > 0, "time step t-1=%d is undefined "%(t-1)

        # if t=1, then x_{t-1} = x_0, so return 1-hot at x_0
        if t==1: return np.eye(self.K)[x_0]

        t_idx = t-1

        # Use Equation 3 if t>1
        Qt      = self.get_Qt(t_idx).to(x_t.device)
        bar_Qt  = self.get_bar_Qt(t_idx).to(x_t.device)
        bar_Qt1 = self.get_bar_Qt(t_idx-1).to(x_t.device)

        # compute probabilities
        if self.singly_stochastic: qs = (Qt[:,x_t].transpose(1,0)*bar_Qt1[x_0])/bar_Qt[x_0, x_t][:,None]
        else: qs = (Qt[x_t]*bar_Qt1[x_0])/bar_Qt[x_0, x_t][:,None]

        return qs

    def p_theta_Xt1_given_Xt_from_pX0(self, x_t, px_0, t):
        """p_theta_Xt1_given_Xt_from_pX0 computes model prediction of
        p(X_{t-1}| X_t) from an output that may be understood as a prediction of
        p(X_{0}|X_t).
        Important: this function operates on one residue position at a time, not on the whole seqeunce
        This implementation follows equation 4 of Austin.
        Args:
            x_t: index of amino acid at time t [1]
            px_0: model prediction of probilities of x_0 [20]
            t: time index (positive integer)
        Returns:
            predicted probabilities of x_{t-1} of shape [20]
        """

        t_idx = t-1

        # compute joint probs
        Qt = self.get_Qt(t_idx).to(x_t.device)
        bar_Qt1 = self.get_bar_Qt(t_idx-1).to(x_t.device)

        q_xt1_xt_given_x0 = lambda x_t1_, x_t_, x_0_: bar_Qt1[x_0_, x_t1_] * Qt[x_t1_, x_t_]

        # assemble all conditional probabilities.
        # all_terms[i,j] = q(x_{t-1}=j, x_t | x_0=i)*px_0[i]
        all_terms = torch.stack([
            px_0i * torch.tensor([
                q_xt1_xt_given_x0(x_t1_j, x_t, x_0_i) # [L]
                for x_t1_j in range(self.K)
            ]).to(x_t.device) # [K,L]
            for x_0_i, px_0i in enumerate(px_0)])

        # compute sum over all x_0 (as on the RHS of equation 4)
        p_theta_x_t1 = torch.sum(all_terms, axis=0)

        # normalize because equation is only up to proportionality
        p_theta_x_t1 /= torch.sum(p_theta_x_t1)
        return p_theta_x_t1

    def KL(self, q, p):
        """KL computes the KL divergence for discrete PMFs q to p"""
        return q.dot(torch.log(q) - torch.log(p))

    def loss(self, x_t, x_0, p_logit_x_0, t, diffusion_mask):
        '''
            Compute the loss given the timestep
            This implementation follows equations 1 and 5 of Austin.
            Args:
                x_t: index of amino acid at time t [B,L]
                x_0: index of amino acid at time 0 [B,L]
                p_logit_x_0: model prediction of logits of x_0 [B,L,20]
                t: time index (positive integer)
        '''

        px_0 = self.softmax(p_logit_x_0)
        return self.loss_prob(x_t, x_0, px_0, t, diffusion_mask)

    def loss_prob(self, x_t, x_0, px_0, t, diffusion_mask):
        assert(t>0), f'Received t <= 0 in seq diffuser loss function. t: {t}'
        assert px_0.ndim == 3, f'{px_0.ndim} != 3. {px_0.shape}'
        B, L, K = px_0.shape
        assert K == self.K, f'{K} != {self.K}'
        assert x_t.shape == (B,L), f'{x_t.shape} != {B,L}'
        assert x_0.shape == (B,L), f'{x_0.shape} != {B,L}'

        # Squeeze out batch dimension
        px_0 = px_0.squeeze(0)
        x_t  = x_t.squeeze(0)
        x_0  = x_0.squeeze(0)

        # Losses described in Equation 1
        if t>1:
            q_Xt1 = self.q_Xt1_given_Xt_and_X0(x_t, x_0, t)
            p_theta_Xt1 = torch.stack([self.p_theta_Xt1_given_Xt_from_pX0(x_ti, px_0i, t) for x_ti, px_0i in zip(x_t, px_0)], dim=0).to(x_t.device)

            loss_vb = torch.stack([self.KL(q_Xt1_, p_theta_Xt1_) for q_Xt1_, p_theta_Xt1_ in zip(q_Xt1, p_theta_Xt1)], dim=0)

        elif t==1:
            p_theta_Xt1 = torch.stack([self.p_theta_Xt1_given_Xt_from_pX0(x_ti, px_0i, t) for x_ti, px_0i in zip(x_t, px_0)], dim=0).to(x_t.device)

            # Now get probability of just x0 sequence by indexing by x0 residue indices
            p_theta_X0 = torch.gather(p_theta_Xt1,1,x_0[:,None]).squeeze(1) # [L]

            loss_vb = -1 * torch.log(p_theta_X0)

        # Add in auxiliary loss from Equation 5, this is the same as the loss in Eq 1 at t==1
        # Equation 5 adds in this auxiliary loss weighted by a factor lambda (self.lamda for us)
        p_theta_Xt1 = torch.stack([self.p_theta_Xt1_given_Xt_from_pX0(x_ti, px_0i, t) for x_ti, px_0i in zip(x_t, px_0)], dim=0).to(x_t.device)

        # Now get probability of just x0 sequence by indexing by x0 residue indices
        p_theta_X0 = torch.gather(px_0, 1, x_0[...,None]).squeeze(1)

        loss_aux = -1 * torch.log(p_theta_X0)

        # Add the main and auxiliary loss to get total loss
        loss = loss_vb + self.lamda * loss_aux

        # Default is all diffused
        if diffusion_mask is None: diffusion_mask = torch.zeros_like(loss, dtype=torch.bool)
        assert diffusion_mask.dtype == torch.bool

        # Not here is because True == seq is not diffused
        return torch.mean(loss[~diffusion_mask]), torch.mean(loss_aux[~diffusion_mask]), torch.mean(loss_vb[~diffusion_mask])

    def get_next_sequence(self, seq_t, pseq0, t, seq_diffusion_mask):
        '''
            Function to take the predicted sequence and get the sequence at t-1 to feed to the model

            Args:

                seq_t (torch.tensor): [L,K] One-hot encoding of sequence fed to model at timestep t

                pseq0 (torch.tensor): [L,K] The model's prediction of sequence logits

                t (int): The current timestep

                seq_diffusion_mask (torch.tensor): [L] Whether each position is diffusing sequence or not

            Returns:

                seq_t_1 (torch.tensor): [L,K] One-hot encoding of sequence to feed to model at timestep t-1

        '''
        L = seq_t.shape[0]

        int_seq_t = torch.argmax(seq_t, dim=-1)

        seq_next_diff = []
        for l in range(L):
            seq_tl = int_seq_t[l]
            pseq0_l = pseq0[l] # [K]

            p_xt1 = self.p_theta_Xt1_given_Xt_from_pX0(seq_tl, pseq0_l, t) # [K]
            p_xt1 = torch.clamp(p_xt1, min=0) # Do not allow negative probabilities
            sampled_seq = torch.multinomial(p_xt1, 1) # [1]
            seq_next_diff.append(sampled_seq)
        
        seq_next_diff = torch.tensor(seq_next_diff) # [L]

        seq_next_diff[seq_diffusion_mask] = int_seq_t[seq_diffusion_mask]

        seq_t_1 = torch.nn.functional.one_hot(seq_next_diff, num_classes=self.K)

        return seq_t_1

    def get_pi(self, L):
        '''Return the stationary distribution.'''
        # TODO: compute this explicitly from Q_bar
        if self.rate_matrix_type == 'uniform':
            return torch.nn.functional.one_hot(torch.randint(0, self.K, size=(L,)), num_classes=self.K)
        raise NotImplementedError(f'rate_matrix_type {rate_matrix_type} not supported')

class ContinuousSeqDiffuser():

    '''
        Class to do discrete diffusion according to the method reported in [1]. This class will
        yield a noised sequence and also will return the "true" probability distributions at
        certain timesteps along the noising trajectory.

        [1] Chen, Ting, et al. "Analog Bits: Generating Discrete Data using Diffusion Models
        ith Self-Conditioning." arXiv (2022)
    '''

    def __init__(self,
                 T,
                 s_b0,
                 s_bT,
                 schedule_type='linear',
                 schedule_params={},
                 loss_type='l2_loss'):

        self.K = 20 # Mask and gap characters not allowed

        # analog bits are in range [-self.bitscale, self.bitscale]
        # This is also the std dev of the noise added during the forward process
        self.bitscale = 1 

        self.T = T

        self.loss_type = loss_type

        # make noise/beta schedule
        self.beta_schedule, _, _ = get_beta_schedule(T, s_b0, s_bT, schedule_type, **schedule_params)

    def continuous_seq(self):
        '''
            This type of sequence diffuser uses a continuous sequence representation so return True
        '''
        return True

    def seq2bits(self,
                 seq,
                 K=None):
        '''
            Given a sequence in integer representation return the sequence as analog bit.

            Args:

                seq (torch.tensor [L], required): Torch tensor of integer sequence representation.

            Returns:

                seqbits (torch.tensor [L,K]): Torch tensor of sequence as analog bits [-1,1] at each position

        '''
        
        if K is None: K = self.K

        seqbits = torch.nn.functional.one_hot(seq, num_classes=K).float()

        # Scale and shift one hot encoding to be a bit encoding [-1,1]
        seqbits = (seqbits * 2 - 1) * self.bitscale

        return seqbits

    def diffuse_sequence(self,
                         seq,
                         diffusion_mask=None,
                         t_list=None):
        '''
            Given a sequence, do discrete diffusion of the sequence and return the result at the
            specified timepoints.

            Args:

                seq (torch.tensor [L], required): Torch tensor of true sequence to be noised. Integer sequence representation

                diffusion_mask (torch.tensor [L], optional): Tensor of bools, True means NOT diffused at this residue, False means diffused

                t_list (list, optional): If present, only return the diffused coordinates at timesteps t within the list

            Returns:

                diffused_seq (torch.tensor [t,L,K] )
                true_seq     (torch.tensor [L,K] )

        '''

        true_seq     = self.seq2bits(seq.clone())
        diffused_seq = self.apply_kernel_recursive(seq, diffusion_mask, t_list)

        # Assert non-diffused regions match - NRB
        if not diffusion_mask is None:
            assert(torch.all( diffused_seq[:,diffusion_mask] == true_seq[None,diffusion_mask] ))

        return diffused_seq, true_seq

    def apply_kernel_recursive(self,
                               seq,
                               diffusion_mask,
                               t_list):

        curr_seq = self.seq2bits( seq.clone() )
        seq_stack = []

        for t in range(1,self.T+1):
            curr_seq = self.apply_kernel(curr_seq,
                                         t,
                                         diffusion_mask)

            seq_stack.append(curr_seq)

        seq_stack = torch.stack(seq_stack, dim=0) # [T,L,K]

        if t_list != None:
            t_idx = torch.tensor([t-1 for t in t_list])
            assert(t_idx>=0).all(), 'detected timestep less than 1'

            seq_stack = seq_stack[t_idx]

        return seq_stack # [t,L,K]

    def apply_kernel(self,
                     seq,
                     t,
                     diffusion_mask=None,
                     var_scale=1):
        '''
            seq (torch.tensor, [L,K])

            t (int)

            diffusion_mask (torch.tensor, [L])
        '''

        # t_idx is 0-indexed
        t_idx = t-1

        L,K = seq.shape
        b_t = self.beta_schedule[t_idx]

        # get noise at timestep t
        mean = torch.sqrt(1-b_t)*seq # [L,K]
        var  = torch.ones(L,K) * b_t * var_scale

        sampled_seq = torch.normal(mean, torch.sqrt(var))
        delta = sampled_seq - seq

        if not diffusion_mask is None:
            delta[diffusion_mask, ...] = 0

        out_seq = seq + delta # [L,K]

        return out_seq

    def loss(self,
             seq_true,
             seq_pred,
             diffusion_mask):
        '''
            Wrapper function to decide which loss to run

            Args:
                
                seq_true (torch.tensor, [L]): The true sequence as integers 

                seq_pred (torch.tensor, [L,K]): The predicted sequence in analog bit form
        '''

        if self.loss_type == 'l2_loss':

            true_bits = self.seq2bits(seq_true, K=21)

            return self.l2_loss(true_bits, seq_pred, diffusion_mask)
        else:
            raise NotImplementedError()

    def l2_loss(self,
                seq_true,
                seq_pred,
                diffusion_mask):
        '''
            Loss described in Equation 2 of Chen et al.

            This is an L2 norm of the distance of the true bits from the predicted bits

            Args:
                
                seq_true (torch.tensor, [L,K]): The true sequence in analog bit form

                seq_pred (torch.tensor, [L,K]): The predicted sequence in analog bit form

                diffusion_mask (torch.tensor, [L]): True means NOT diffused at this residue
        '''

        loss = torch.square( seq_true - seq_pred )

        # Sum loss across all categories
        loss = torch.mean(loss, dim=-1)
        loss = loss * diffusion_mask # Zero out entries that are not being diffused

        return loss.sum() / (diffusion_mask.sum() + 1e-8)

    def sigmoid_loss(self,
                     seq_true,
                     seq_pred):
        '''
            Loss described in B.2 of Chen et al.

        '''

        raise NotImplementedError()

    def cce_loss(self,
                 seq_true,
                 seq_pred):
        '''
            Loss described in B.3 of Chen et al.
        '''

        raise NotImplementedError()
