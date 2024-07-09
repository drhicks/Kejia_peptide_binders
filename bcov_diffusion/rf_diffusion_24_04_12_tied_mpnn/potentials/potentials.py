import torch
from icecream import ic 
import numpy as np 
from util import generate_Cbeta

class Potential:
    '''
        Interface class that defines the functions a potential must implement
    '''

    def compute(self, seq, xyz):
        '''
            Given the current sequence and structure of the model prediction, return the current
            potential as a PyTorch tensor with a single entry

            Args:
                seq (torch.tensor, size: [L,?]:    The current sequence of the sample.
                                                     TODO: determine whether this is one hot or an 
                                                     integer representation
                xyz (torch.tensor, size: [L,27,3]: The current coordinates of the sample
            
            Returns:
                potential (torch.tensor, size: [1]): A potential whose value will be MAXIMIZED
                                                     by taking a step along it's gradient
        '''
        raise NotImplementedError('Potential compute function was not overwritten')


    def raw_delta_ca(self, seq, xyz, t, T):
        '''
            Given the current sequence and structure of the model prediction, return atomic deltas

            Args:
                seq (torch.tensor, size: [L,?]:    The current sequence of the sample.
                                                     TODO: determine whether this is one hot or an 
                                                     integer representation
                xyz (torch.tensor, size: [L,27,3]: The current coordinates of the sample
                t (int, required):                  The current timestep
                T (int, required):                  Big T

            
            Returns:
                deltas xyz (torch.tensor, size: [L,3])): Deltas to add to CA
        '''
        return torch.zeros((len(xyz), 3))



class monomer_ROG(Potential):
    '''
        Radius of Gyration potential for encouraging monomer compactness

        Written by DJ and refactored into a class by NRB
    '''

    def __init__(self, weight=1, min_dist=15):

        self.weight   = weight
        self.min_dist = min_dist

    def compute(self, seq, xyz):
        Ca = xyz[:,1] # [L,3]

        centroid = torch.mean(Ca, dim=0, keepdim=True) # [1,3]

        dgram = torch.cdist(Ca[None,...].contiguous(), centroid[None,...].contiguous(), p=2) # [1,L,1,3]

        dgram = torch.maximum(self.min_dist * torch.ones_like(dgram.squeeze(0)), dgram.squeeze(0)) # [L,1,3]

        rad_of_gyration = torch.sqrt( torch.sum(torch.square(dgram)) / Ca.shape[0] ) # [1]

        return -1 * self.weight * rad_of_gyration

class binder_ROG(Potential):
    '''
        Radius of Gyration potential for encouraging binder compactness

        Author: NRB
    '''

    def __init__(self, binderlen, weight=1, min_dist=15):

        self.binderlen = binderlen
        self.min_dist  = min_dist
        self.weight    = weight

    def compute(self, seq, xyz):
        
        # Only look at binder residues
        Ca = xyz[:self.binderlen,1] # [Lb,3]

        centroid = torch.mean(Ca, dim=0, keepdim=True) # [1,3]

        # cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca[None,...].contiguous(), centroid[None,...].contiguous(), p=2) # [1,Lb,1,3]

        dgram = torch.maximum(self.min_dist * torch.ones_like(dgram.squeeze(0)), dgram.squeeze(0)) # [Lb,1,3]

        rad_of_gyration = torch.sqrt( torch.sum(torch.square(dgram)) / Ca.shape[0] ) # [1]

        return -1 * self.weight * rad_of_gyration


class dimer_ROG(Potential):
    '''
        Radius of Gyration potential for encouraging compactness of both monomers when designing dimers

        Author: PV
    '''

    def __init__(self, binderlen, weight=1, min_dist=15):

        self.binderlen = binderlen
        self.min_dist  = min_dist
        self.weight    = weight

    def compute(self, seq, xyz):

        # Only look at monomer 1 residues
        Ca_m1 = xyz[:self.binderlen,1] # [Lb,3]
        
        # Only look at monomer 2 residues
        Ca_m2 = xyz[self.binderlen:,1] # [Lb,3]

        centroid_m1 = torch.mean(Ca_m1, dim=0, keepdim=True) # [1,3]
        centroid_m2 = torch.mean(Ca_m1, dim=0, keepdim=True) # [1,3]

        # cdist needs a batch dimension - NRB
        #This calculates RoG for Monomer 1
        dgram_m1 = torch.cdist(Ca_m1[None,...].contiguous(), centroid_m1[None,...].contiguous(), p=2) # [1,Lb,1,3]
        dgram_m1 = torch.maximum(self.min_dist * torch.ones_like(dgram_m1.squeeze(0)), dgram_m1.squeeze(0)) # [Lb,1,3]
        rad_of_gyration_m1 = torch.sqrt( torch.sum(torch.square(dgram_m1)) / Ca_m1.shape[0] ) # [1]

        # cdist needs a batch dimension - NRB
        #This calculates RoG for Monomer 2
        dgram_m2 = torch.cdist(Ca_m2[None,...].contiguous(), centroid_m2[None,...].contiguous(), p=2) # [1,Lb,1,3]
        dgram_m2 = torch.maximum(self.min_dist * torch.ones_like(dgram_m2.squeeze(0)), dgram_m2.squeeze(0)) # [Lb,1,3]
        rad_of_gyration_m2 = torch.sqrt( torch.sum(torch.square(dgram_m2)) / Ca_m2.shape[0] ) # [1]

        #Potential value is the average of both radii of gyration (is avg. the best way to do this?)
        return -1 * self.weight * (rad_of_gyration_m1 + rad_of_gyration_m2)/2

class binder_ncontacts(Potential):
    '''
        Differentiable way to maximise number of contacts within a protein
        
        Motivation is given here: https://www.plumed.org/doc-v2.7/user-doc/html/_c_o_o_r_d_i_n_a_t_i_o_n.html

        Author: PV
    '''

    def __init__(self, binderlen, weight=1, r_0=8, d_0=4):

        self.binderlen = binderlen
        self.r_0       = r_0
        self.weight    = weight
        self.d_0       = d_0

    def compute(self, seq, xyz):

        # Only look at binder Ca residues
        Ca = xyz[:self.binderlen,1] # [Lb,3]
        
        #cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca[None,...].contiguous(), Ca[None,...].contiguous(), p=2) # [1,Lb,Lb]
        divide_by_r_0 = (dgram - self.d_0) / self.r_0
        numerator = torch.pow(divide_by_r_0,6)
        denominator = torch.pow(divide_by_r_0,12)
        binder_ncontacts = (1 - numerator) / (1 - denominator)
        
        print("BINDER CONTACTS:", binder_ncontacts.sum())
        #Potential value is the average of both radii of gyration (is avg. the best way to do this?)
        return self.weight * binder_ncontacts.sum()

    
class dimer_ncontacts(Potential):

    '''
        Differentiable way to maximise number of contacts for two individual monomers in a dimer
        
        Motivation is given here: https://www.plumed.org/doc-v2.7/user-doc/html/_c_o_o_r_d_i_n_a_t_i_o_n.html

        Author: PV
    '''


    def __init__(self, binderlen, weight=1, r_0=8, d_0=4):

        self.binderlen = binderlen
        self.r_0       = r_0
        self.weight    = weight
        self.d_0       = d_0

    def compute(self, seq, xyz):

        # Only look at binder Ca residues
        Ca = xyz[:self.binderlen,1] # [Lb,3]
        #cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca[None,...].contiguous(), Ca[None,...].contiguous(), p=2) # [1,Lb,Lb]
        divide_by_r_0 = (dgram - self.d_0) / self.r_0
        numerator = torch.pow(divide_by_r_0,6)
        denominator = torch.pow(divide_by_r_0,12)
        binder_ncontacts = (1 - numerator) / (1 - denominator)
        #Potential is the sum of values in the tensor
        binder_ncontacts = binder_ncontacts.sum()

        # Only look at target Ca residues
        Ca = xyz[self.binderlen:,1] # [Lb,3]
        dgram = torch.cdist(Ca[None,...].contiguous(), Ca[None,...].contiguous(), p=2) # [1,Lb,Lb]
        divide_by_r_0 = (dgram - self.d_0) / self.r_0
        numerator = torch.pow(divide_by_r_0,6)
        denominator = torch.pow(divide_by_r_0,12)
        target_ncontacts = (1 - numerator) / (1 - denominator)
        #Potential is the sum of values in the tensor
        target_ncontacts = target_ncontacts.sum()
        
        print("DIMER NCONTACTS:", (binder_ncontacts+target_ncontacts)/2)
        #Returns average of n contacts withiin monomer 1 and monomer 2
        return self.weight * (binder_ncontacts+target_ncontacts)/2

class interface_ncontacts(Potential):

    '''
        Differentiable way to maximise number of contacts between binder and target
        
        Motivation is given here: https://www.plumed.org/doc-v2.7/user-doc/html/_c_o_o_r_d_i_n_a_t_i_o_n.html

        Author: PV
    '''


    def __init__(self, binderlen, weight=1, r_0=8, d_0=6):

        self.binderlen = binderlen
        self.r_0       = r_0
        self.weight    = weight
        self.d_0       = d_0

    def compute(self, seq, xyz):

        # Extract binder Ca residues
        Ca_b = xyz[:self.binderlen,1] # [Lb,3]

        # Extract target Ca residues
        Ca_t = xyz[self.binderlen:,1] # [Lt,3]

        #cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca_b[None,...].contiguous(), Ca_t[None,...].contiguous(), p=2) # [1,Lb,Lt]
        divide_by_r_0 = (dgram - self.d_0) / self.r_0
        numerator = torch.pow(divide_by_r_0,6)
        denominator = torch.pow(divide_by_r_0,12)
        interface_ncontacts = (1 - numerator) / (1 - denominator)
        #Potential is the sum of values in the tensor
        interface_ncontacts = interface_ncontacts.sum()

        print("INTERFACE CONTACTS:", interface_ncontacts.sum())

        return self.weight * interface_ncontacts


class monomer_contacts(Potential):
    '''
        Differentiable way to maximise number of contacts within a protein

        Motivation is given here: https://www.plumed.org/doc-v2.7/user-doc/html/_c_o_o_r_d_i_n_a_t_i_o_n.html
        Author: PV

        NOTE: This function sometimes produces NaN's -- added check in reverse diffusion for nan grads
    '''

    def __init__(self, weight=1, r_0=8, d_0=2, eps=1e-6):

        self.r_0       = r_0
        self.weight    = weight
        self.d_0       = d_0
        self.eps       = eps

    def compute(self, seq, xyz):

        Ca = xyz[:,1] # [L,3]

        #cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca[None,...].contiguous(), Ca[None,...].contiguous(), p=2) # [1,Lb,Lb]
        divide_by_r_0 = (dgram - self.d_0) / self.r_0
        numerator = torch.pow(divide_by_r_0,6)
        denominator = torch.pow(divide_by_r_0,12)

        ncontacts = (1 - numerator) / ((1 - denominator))


        #Potential value is the average of both radii of gyration (is avg. the best way to do this?)
        return self.weight * ncontacts.sum()


def make_contact_matrix(nchain, contact_string=None):
    """
    Calculate a matrix of inter/intra chain contact indicators
    
    Parameters:
        nchain (int, required): How many chains are in this design 
        
        contact_str (str, optional): String denoting how to define contacts, comma delimited between pairs of chains
            '!' denotes repulsive, '&' denotes attractive
    """
    alphabet   = [a for a in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
    letter2num = {a:i for i,a in enumerate(alphabet)}
    
    contacts   = np.zeros((nchain,nchain))
    written    = np.zeros((nchain,nchain))
    
    contact_list = contact_string.split(',') 
    for c in contact_list:
        if not len(c) == 3:
            raise SyntaxError('Invalid contact(s) specification')

        i,j = letter2num[c[0]],letter2num[c[2]]
        symbol = c[1]
        
        # denote contacting/repulsive
        assert symbol in ['!','&']
        if symbol == '!':
            contacts[i,j] = -1
            contacts[j,i] = -1
        else:
            contacts[i,j] = 1
            contacts[j,i] = 1
            
    return contacts 


class olig_contacts(Potential):
    """
    Applies PV's num contacts potential within/between chains in symmetric oligomers 

    Author: DJ 
    """

    def __init__(self, 
                 contact_matrix, 
                 weight_intra=1, 
                 weight_inter=1,
                 r_0=8, d_0=2):
        """
        Parameters:
            chain_lengths (list, required): List of chain lengths, length is (Nchains)

            contact_matrix (torch.tensor/np.array, required): 
                square matrix of shape (Nchains,Nchains) whose (i,j) enry represents 
                attractive (1), repulsive (-1), or non-existent (0) contact potentials 
                between chains in the complex

            weight (int/float, optional): Scaling/weighting factor
        """
        print('This is chain contact matrix you are using')
        ic(contact_matrix)
        self.contact_matrix = contact_matrix
        self.weight_intra = weight_intra 
        self.weight_inter = weight_inter 
        self.r_0 = r_0
        self.d_0 = d_0

        # check contact matrix only contains valid entries 
        assert all([i in [-1,0,1] for i in contact_matrix.flatten()]), 'Contact matrix must contain only 0, 1, or -1 in entries'
        # assert the matrix is square and symmetric 
        shape = contact_matrix.shape 
        assert len(shape) == 2 
        assert shape[0] == shape[1]
        for i in range(shape[0]):
            for j in range(shape[1]):
                assert contact_matrix[i,j] == contact_matrix[j,i]
        self.nchain=shape[0]

         
    #   self._compute_chain_indices()

    # def _compute_chain_indices(self):
    #     # make list of shape [i,N] for indices of each chain in total length
    #     indices = []
    #     start   = 0
    #     for l in self.chain_lengths:
    #         indices.append(torch.arange(start,start+l))
    #         start += l
    #     self.indices = indices 

    def _get_idx(self,i,L):
        """
        Returns the zero-indexed indices of the residues in chain i
        """
        assert L%self.nchain == 0
        Lchain = L//self.nchain
        return i*Lchain + torch.arange(Lchain)


    def compute(self, seq, xyz):
        """
        Iterate through the contact matrix, compute contact potentials between chains that need it,
        and negate contacts for any 
        """
        L = len(seq.squeeze())

        all_contacts = 0
        start = 0
        for i in range(self.nchain):
            for j in range(self.nchain):
                # only compute for upper triangle, disregard zeros in contact matrix 
                if (i <= j) and (self.contact_matrix[i,j] != 0):

                    # get the indices for these two chains 
                    idx_i = self._get_idx(i,L)
                    idx_j = self._get_idx(j,L)

                    Ca_i = xyz[idx_i,1]  # slice out crds for this chain 
                    Ca_j = xyz[idx_j,1]  # slice out crds for that chain 
                    dgram           = torch.cdist(Ca_i[None,...].contiguous(), Ca_j[None,...].contiguous(), p=2) # [1,Lb,Lb]

                    divide_by_r_0   = (dgram - self.d_0) / self.r_0
                    numerator       = torch.pow(divide_by_r_0,6)
                    denominator     = torch.pow(divide_by_r_0,12)
                    ncontacts       = (1 - numerator) / (1 - denominator)

                    # weight, don't double count intra 
                    scalar = (i==j)*self.weight_intra/2 + (i!=j)*self.weight_inter

                    #                 contacts              attr/repuls          relative weights 
                    all_contacts += ncontacts.sum() * self.contact_matrix[i,j] * scalar 

        return all_contacts 
                    

class olig_intra_contacts(Potential):
    """
    Applies PV's num contacts potential for each chain individually in an oligomer design 

    Author: DJ 
    """

    def __init__(self, chain_lengths, weight=1):
        """
        Parameters:

            chain_lengths (list, required): Ordered list of chain lengths 

            weight (int/float, optional): Scaling/weighting factor
        """
        self.chain_lengths = chain_lengths 
        self.weight = weight 


    def compute(self, seq, xyz):
        """
        Computes intra-chain num contacts potential
        """
        assert sum(self.chain_lengths) == len(seq.squeeze), 'given chain lengths do not match total sequence length'

        all_contacts = 0
        start = 0
        for Lc in self.chain_lengths:
            Ca = xyz[start:start+Lc]  # slice out crds for this chain 
            dgram = torch.cdist(Ca[None,...].contiguous(), Ca[None,...].contiguous(), p=2) # [1,Lb,Lb]
            divide_by_r_0 = (dgram - self.d_0) / self.r_0
            numerator = torch.pow(divide_by_r_0,6)
            denominator = torch.pow(divide_by_r_0,12)
            ncontacts = (1 - numerator) / (1 - denominator)

            # add contacts for this chain to all contacts 
            all_contacts += ncontacts.sum()

            # increment the start to be at the next chain 
            start += Lc 


        return self.weight * all_contacts


class binder_distance_ReLU(Potential):
    '''
        Given the current coordinates of the diffusion trajectory, calculate a potential that is the distance between each residue
        and the closest target residue.

        This potential is meant to encourage the binder to interact with a certain subset of residues on the target that 
        define the binding site.

        Author: NRB
    '''

    def __init__(self, binderlen, hotspot_res, weight=1, min_dist=15, use_Cb=False):

        self.binderlen   = binderlen
        self.hotspot_res = [res + binderlen for res in hotspot_res]
        self.weight      = weight
        self.min_dist    = min_dist
        self.use_Cb      = use_Cb

    def compute(self, seq, xyz):
        binder = xyz[:self.binderlen,:,:] # (Lb,27,3)
        target = xyz[self.hotspot_res,:,:] # (N,27,3)

        if self.use_Cb:
            N  = binder[:,0]
            Ca = binder[:,1]
            C  = binder[:,2]

            Cb = generate_Cbeta(N,Ca,C) # (Lb,3)

            N_t  = target[:,0]
            Ca_t = target[:,1]
            C_t  = target[:,2]

            Cb_t = generate_Cbeta(N_t,Ca_t,C_t) # (N,3)

            dgram = torch.cdist(Cb[None,...], Cb_t[None,...], p=2) # (1,Lb,N)

        else:
            # Use Ca dist for potential

            Ca = binder[:,1] # (Lb,3)

            Ca_t = target[:,1] # (N,3)

            dgram = torch.cdist(Ca[None,...], Ca_t[None,...], p=2) # (1,Lb,N)

        closest_dist = torch.min(dgram.squeeze(0), dim=1)[0] # (Lb)

        # Cap the distance at a minimum value
        min_distance = self.min_dist * torch.ones_like(closest_dist) # (Lb)
        potential    = torch.maximum(min_distance, closest_dist) # (Lb)

        # torch.Tensor.backward() requires the potential to be a single value
        potential    = torch.sum(potential, dim=-1)
        
        return -1 * self.weight * potential

class binder_any_ReLU(Potential):
    '''
        Given the current coordinates of the diffusion trajectory, calculate a potential that is the minimum distance between
        ANY residue and the closest target residue.

        In contrast to binder_distance_ReLU this potential will only penalize a pose if all of the binder residues are outside
        of a certain distance from the target residues.

        Author: NRB
    '''

    def __init__(self, binderlen, hotspot_res, weight=1, min_dist=15, use_Cb=False):

        self.binderlen   = binderlen
        self.hotspot_res = [res + binderlen for res in hotspot_res]
        self.weight      = weight
        self.min_dist    = min_dist
        self.use_Cb      = use_Cb

    def compute(self, seq, xyz):
        binder = xyz[:self.binderlen,:,:] # (Lb,27,3)
        target = xyz[self.hotspot_res,:,:] # (N,27,3)

        if use_Cb:
            N  = binder[:,0]
            Ca = binder[:,1]
            C  = binder[:,2]

            Cb = generate_Cbeta(N,Ca,C) # (Lb,3)

            N_t  = target[:,0]
            Ca_t = target[:,1]
            C_t  = target[:,2]

            Cb_t = generate_Cbeta(N_t,Ca_t,C_t) # (N,3)

            dgram = torch.cdist(Cb[None,...], Cb_t[None,...], p=2) # (1,Lb,N)

        else:
            # Use Ca dist for potential

            Ca = binder[:,1] # (Lb,3)

            Ca_t = target[:,1] # (N,3)

            dgram = torch.cdist(Ca[None,...], Ca_t[None,...], p=2) # (1,Lb,N)


        closest_dist = torch.min(dgram.squeeze(0)) # (1)

        potential    = torch.maximum(min_dist, closest_dist) # (1)

        return -1 * self.weight * potential





_atom_record_format = (
    "ATOM  {atomi:5d} {atomn:^4}{idx:^1}{resn:3s} {chain:1}{resi:4d}{insert:1s}   "
    "{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}{seg:-4d}{elem:2s}\n"
)


def format_atom(
        atomi=0,
        atomn='ATOM',
        idx=' ',
        resn='RES',
        chain='A',
        resi=0,
        insert=' ',
        x=0,
        y=0,
        z=0,
        occ=1,
        b=0,
        seg=1,
        elem=''
):
    return _atom_record_format.format(**locals())

def dump_pts(pts, name):
    with open(name, "w") as f:
        for ivert, vert in enumerate(pts):
            f.write(format_atom(ivert%100000, resi=ivert%10000, x=vert[0], y=vert[1], z=vert[2]))

def dump_line(start, direction, length, name):
    dump_lines([start], [direction], length, name)

def dump_lines(starts, directions, length, name):

    starts = np.array(starts)
    if ( len(starts.shape) == 1 ):
        starts = np.tile(starts, (len(directions), 1))

    directions = np.array(directions)

    vec = np.linspace(0, length, 80)

    pt_collections = []

    for i in range(len(starts)):
        start = starts[i]
        direction = directions[i]

        pts = start + direction*vec[:,None]
        pt_collections.append(pts)

    pts = np.concatenate(pt_collections)

    dump_pts(pts, name)




def parse_pt_string(pt_string, cas):

    parts = pt_string.split("~")
    assert(parts[0] == "T")
    parts.pop(0)

    # print(len(parts))
    assert(len(parts) == 4)

    at1, weight1 = parts[0].split("@")
    at2, weight2 = parts[1].split("@")
    at3, weight3 = parts[2].split("@")

    at1 = int(at1)
    at2 = int(at2)
    at3 = int(at3)

    weight1 = float(weight1)
    weight2 = float(weight2)
    weight3 = float(weight3)
    height = float(parts[3])


    # npose = nu.npose_from_file(pdb)

    # cas = nu.extract_atoms(npose, [nu.CA])[:,:3]


    pre_xyz1 = cas[at1-1]
    pre_xyz2 = cas[at2-1]
    pre_xyz3 = cas[at3-1]



    def get_xyz(our_xyz, weight, other1, other2):

        if ( weight > 0 ):
            return our_xyz

        midpoint = (other1 + other2)/2

        to_mid = midpoint - our_xyz

        return midpoint + to_mid


    xyz1 = get_xyz(pre_xyz1, weight1, pre_xyz2, pre_xyz3)
    xyz2 = get_xyz(pre_xyz2, weight2, pre_xyz1, pre_xyz3)
    xyz3 = get_xyz(pre_xyz3, weight3, pre_xyz1, pre_xyz2)

    weight1 = abs(weight1)
    weight2 = abs(weight2)
    weight3 = abs(weight3)

    total_weight = weight1 + weight2 + weight3


    plane_pt = (xyz1 * weight1 + xyz2 * weight2 + xyz3 * weight3) / total_weight

    # right hand rule on the new points determines up
    vec1 = xyz2 - xyz1
    vec2 = xyz3 - xyz2

    up_unit = torch.cross(vec1, vec2)
    up_unit /= torch.sqrt(torch.sum(torch.square( up_unit)))

    final_pt = plane_pt + up_unit * height

    return final_pt



class move_span_com(Potential):
    '''
        Try to move the COM of a span to a specific spot


        Author: NRB
    '''

    def __init__(self, binderlen, weight=1, spans=[], pt_strings=[], weights=None):

        self.binderlen   = binderlen
        # self.hotspot_res = [res + binderlen for res in hotspot_res]
        self.weight      = weight
        self.pt_strings  = pt_strings

        self.spans = []
        for span in spans:
            assert("_" in span)
            sp = span.split("_")
            assert(len(sp) == 2)
            lb = int(sp[0])
            ub = int(sp[1])
            assert(lb <= ub)

            if ( lb > 0 ):
                lb -= 1
            if ( ub > 0 ):
                ub -= 1

            self.spans.append([lb, ub])

        if ( weights is None ):
            self.weights = torch.ones(len(self.spans))
        else:
            assert(len(weights) == len(self.spans))
            self.weights = weights


        assert( len(self.spans) == len(self.pt_strings))

    def compute(self, seq, xyz):
        
        # constant because bcov is bad at torch
        return xyz[0,0,0] / xyz[0,0,0]


    def raw_delta_ca(self, seq, xyz, t, T):

        # Extract binder Ca residues
        Ca_b = xyz[:self.binderlen,1] # [Lb,3]

        # Extract target Ca residues
        Ca_t = xyz[self.binderlen:,1] # [Lt,3]

        deltas = torch.zeros((len(xyz), 3))

        target_pts = []
        coms = []

        for ispan in range(len(self.spans)):

            target_pt = parse_pt_string(self.pt_strings[ispan], Ca_t)
            target_pts.append(target_pt)

            mask = torch.zeros(len(Ca_b), dtype=bool)

            lb, ub = self.spans[ispan]
            if ( ub != -1 ):
                mask[lb:ub] = True
            else:
                mask[lb:] = True



            com = torch.mean(Ca_b[mask], axis=0)
            goal_dist = torch.cdist(com[None,:], target_pt[None,:])[0,0]

            unit = target_pt - com
            unit /= goal_dist

            coms.append(com)

            full_mask = torch.zeros(len(xyz), dtype=bool)
            full_mask[:self.binderlen] = mask

            move_ratio = goal_dist / ( self.weights[ispan] * self.weight )
            if ( move_ratio < 1 ):
                unit *= move_ratio

            deltas[full_mask,:] += unit * self.weights[ispan]


        dump_pts(target_pts, "target_pts.pdb")
        dump_pts(coms, "coms.pdb")

        return deltas * self.weight


import sklearn.decomposition
import scipy.spatial.transform


class linearize_spans(Potential):
    '''
        try to rotate all the cas in a span into a line


        Author: bcov
    '''

    def __init__(self, binderlen, weight=1, spans=[], weights=None):

        self.binderlen   = binderlen
        # self.hotspot_res = [res + binderlen for res in hotspot_res]
        self.weight      = weight
        # self.pt_strings  = pt_strings

        self.spans = []
        for span in spans:
            assert("_" in span)
            sp = span.split("_")
            assert(len(sp) == 2)
            lb = int(sp[0])
            ub = int(sp[1])
            assert(lb <= ub)

            if ( lb > 0 ):
                lb -= 1
            if ( ub > 0 ):
                ub -= 1

            self.spans.append([lb, ub])

        if ( weights is None ):
            self.weights = torch.ones(len(self.spans))
        else:
            assert(len(weights) == len(self.spans))
            self.weights = weights


        # assert( len(self.spans) == len(self.pt_strings))

    def compute(self, seq, xyz):
        
        # constant because bcov is bad at torch
        return xyz[0,0,0] / xyz[0,0,0]


    def raw_delta_ca(self, seq, xyz, t, T):

        # Extract binder Ca residues
        Ca_b = xyz[:self.binderlen,1] # [Lb,3]

        # Extract target Ca residues
        Ca_t = xyz[self.binderlen:,1] # [Lt,3]

        deltas = torch.zeros((len(xyz), 3))

        starts = []
        ends = []

        for ispan in range(len(self.spans)):

            # get a mask for this span
            mask = torch.zeros(len(Ca_b), dtype=bool)
            lb, ub = self.spans[ispan]
            if ( ub != -1 ):
                mask[lb:ub] = True
            else:
                mask[lb:] = True


            # get cas, com, and principle axis of cas
            span_cas = Ca_b[mask]
            com = torch.mean(span_cas, axis=0)
            pca = sklearn.decomposition.PCA()
            pca.fit(span_cas)
            axis = torch.tensor(pca.components_[0], dtype=xyz.dtype)

            # we rotate the points towards the pca unless they are within 10deg of the equator
            equator_close = np.radians(10)

            
            # figure out the curreent angles and distances to com
            to_com = com - span_cas
            dist_to_com = torch.cdist(com[None,:], span_cas)[0]
            to_com_unit = to_com / dist_to_com[:,None]
            cos_dots = torch.sum( to_com_unit * axis, axis=-1 )
            angle_to_com = torch.arccos( cos_dots )

            is_equitorial = torch.abs( angle_to_com - np.pi/2 ) < equator_close


            # figure out how many degrees we'd need to rotate them
            # define positive rotations as aligning against pca vector
            translation_amount = self.weight * self.weights[ispan]

            # each_circumference = dist_to_com * 2 * np.pi
            translation_radians = translation_amount / dist_to_com
            translation_radians_with_dir = -torch.sign(cos_dots) * translation_radians

            # don't rotate past the axis, need special cases for both rotation directions (clip to 0 and pi)
            is_rotating_against = cos_dots < 0
            raw_final_angle = translation_radians_with_dir + angle_to_com

            realized_translation_radians = translation_radians_with_dir

            tmp_mask = is_rotating_against & (raw_final_angle > np.pi)
            realized_translation_radians[ tmp_mask ] = np.pi - angle_to_com[tmp_mask]

            tmp_mask = is_rotating_against & (raw_final_angle < 0)
            realized_translation_radians[ tmp_mask ] = - angle_to_com[tmp_mask]


            # actually get the rotation matrices
            rotation_units = torch.cross(axis.broadcast_to((len(to_com_unit), -1)), -to_com_unit)
            rotation_units /= torch.sqrt(torch.sum(torch.square(rotation_units), axis=-1))[:,None]

            # dump_lines(com.broadcast_to(len(rotation_units), 3), rotation_units, 1, "rotation_units.pdb")

            # don't ask me why the -1 is there, but I checked the outputs and that's how it works
            rotation_matrices = scipy.spatial.transform.Rotation.from_rotvec( -rotation_units * realized_translation_radians[:,None] ).as_matrix()
            rotation_matrices = torch.tensor(rotation_matrices, dtype=xyz.dtype)

            # import IPython
            # IPython.embed()

            final_pts = torch.zeros( (len(span_cas), 3), dtype=xyz.dtype)
            for i in range(len(final_pts)):
                final_pts[i] = rotation_matrices[i] @ (span_cas[i] - com) + com

            rotation_as_translation = final_pts - span_cas
            rotation_as_translation[is_equitorial,:] = 0


            # for i in range(len(rotation_as_translation)):
            #     print(torch.sqrt(torch.sum(torch.square(rotation_as_translation[i]))))

            # import IPython
            # IPython.embed()

            starts.append(span_cas)
            ends.append(final_pts)


            full_mask = torch.zeros(len(xyz), dtype=bool)
            full_mask[:self.binderlen] = mask


            deltas[full_mask,:] += rotation_as_translation


        # starts = torch.cat(starts)
        # ends = torch.cat(ends)

        # dump_lines(starts, ends-starts, 10, "linearize_vectors.pdb")
        # dump_pts(coms, "coms.pdb")

        return deltas #* self.weight



class bend_adj_spans(Potential):
    '''
        prevent adjacent spans from being colinear


        Author: bcov
    '''

    def __init__(self, binderlen, weight=1, spans=[], weights=None, min_angle=90):

        self.binderlen   = binderlen
        # self.hotspot_res = [res + binderlen for res in hotspot_res]
        self.weight      = weight
        self.min_angle_rad = np.radians(min_angle)
        # self.pt_strings  = pt_strings

        self.spans = []
        for span in spans:
            assert("_" in span)
            sp = span.split("_")
            assert(len(sp) == 2)
            lb = int(sp[0])
            ub = int(sp[1])
            assert(lb <= ub)

            if ( lb > 0 ):
                lb -= 1
            if ( ub > 0 ):
                ub -= 1

            self.spans.append([lb, ub])

        if ( weights is None ):
            self.weights = torch.ones(len(self.spans))
        else:
            assert(len(weights)*2 == len(self.spans))
            self.weights = weights

        assert(len(spans)%2 == 0)


        # assert( len(self.spans) == len(self.pt_strings))

    def compute(self, seq, xyz):
        
        # constant because bcov is bad at torch
        return xyz[0,0,0] / xyz[0,0,0]


    # figure out the "start" and "end" of the span
    #  this gets tricky when the span is a ball of points
    #  just use the com of first and second halfs and then pretend
    #  everything is linear
    def extract_span_stuff(self, ispan, Ca_b):

        mask = torch.zeros(len(Ca_b), dtype=bool)
        lb, ub = self.spans[ispan]
        if ( ub != -1 ):
            mask[lb:ub] = True
        else:
            mask[lb:] = True

        indices = torch.where(mask)[0]
        half_size = len(indices)//2

        span_cas = Ca_b[mask]
        com = torch.mean(span_cas, axis=0)

        first_com = torch.mean(Ca_b[indices[:half_size]], axis=0)
        second_com = torch.mean(Ca_b[indices[-half_size:]], axis=0)

        half_length_vec = second_com - first_com
        start = com - half_length_vec
        end = com + half_length_vec

        return mask, span_cas, com, start, end



    def raw_delta_ca(self, seq, xyz, t, T):

        # Extract binder Ca residues
        Ca_b = xyz[:self.binderlen,1] # [Lb,3]

        # Extract target Ca residues
        Ca_t = xyz[self.binderlen:,1] # [Lt,3]

        deltas = torch.zeros((len(xyz), 3))

        starts = []
        ends = []

        num_pairs = len(self.spans)//2

        for ipair in range(num_pairs):

            mask1, cas1, com1, start1, end1 = self.extract_span_stuff(ipair*2, Ca_b)
            mask2, cas2, com2, start2, end2 = self.extract_span_stuff(ipair*2+1, Ca_b)

            hinge_point = (end1 + start2)/2

            com1_hinge_unit = com1 - hinge_point
            com1_hinge_unit /= torch.sqrt(torch.sum(torch.square(com1_hinge_unit)))

            com2_hinge_unit = com2 - hinge_point
            com2_hinge_unit /= torch.sqrt(torch.sum(torch.square(com2_hinge_unit)))


            angle_rad = torch.arccos( torch.sum( com1_hinge_unit * com2_hinge_unit ) )
            if ( angle_rad < self.min_angle_rad ):
                continue

            rotation_axis = torch.cross(com1_hinge_unit, com2_hinge_unit)

            # if they are colinear
            if ( torch.sqrt( torch.sum(torch.square(rotation_axis))) == 0 ):
                rotation_axis = torch.cross(com1_hinge_unit, torch.tensor([1, 0, 0]))

                # if they are colinear along the x-axis
                if ( torch.sqrt( torch.sum(torch.square(rotation_axis))) == 0 ):
                    rotation_axis = torch.cross(com1_hinge_unit, torch.tensor([0, 1, 0]))

            norm = torch.sqrt( torch.sum(torch.square(rotation_axis)))
            assert( norm > 0)
            rotation_axis /= norm


            # we want to make the angle between them smaller as calculated by cos
            #  If we pretend that span1 is the x-axis of a 2d graph
            #     span1 needs to increase
            #     span2 needs to decrease

            hinge_to_cas1 = cas1 - com1[None,:]
            hinge_to_cas2 = cas2 - com2[None,:]

            hinge_to_cas1_norm = torch.sqrt(torch.sum(torch.square(hinge_to_cas1), axis=-1))
            hinge_to_cas2_norm = torch.sqrt(torch.sum(torch.square(hinge_to_cas2), axis=-1))

            translation_amount = self.weight * self.weights[ipair]
            radians1 = translation_amount / hinge_to_cas1_norm
            radians2 = -translation_amount / hinge_to_cas2_norm


            # import IPython
            # IPython.embed()

            # I swear this uses left-hand rule hence the -1
            rotation_matrices1 = scipy.spatial.transform.Rotation.from_rotvec( -rotation_axis[None,:] * radians1[:,None] ).as_matrix()
            rotation_matrices1 = torch.tensor(rotation_matrices1, dtype=xyz.dtype)

            rotation_matrices2 = scipy.spatial.transform.Rotation.from_rotvec( -rotation_axis[None,:] * radians2[:,None] ).as_matrix()
            rotation_matrices2 = torch.tensor(rotation_matrices2, dtype=xyz.dtype)


            final_pts1 = torch.zeros( (len(cas1), 3), dtype=xyz.dtype)
            for i in range(len(final_pts1)):
                final_pts1[i] = rotation_matrices1[i] @ (cas1[i] - com1) + com1

            final_pts2 = torch.zeros( (len(cas2), 3), dtype=xyz.dtype)
            for i in range(len(final_pts2)):
                final_pts2[i] = rotation_matrices2[i] @ (cas2[i] - com2) + com2

            rotation_as_translation1 = final_pts1 - cas1
            rotation_as_translation2 = final_pts2 - cas2


            # for i in range(len(rotation_as_translation1)):
            #     print(torch.sqrt(torch.sum(torch.square(rotation_as_translation1[i]))))
            # for i in range(len(rotation_as_translation2)):
            #     print(torch.sqrt(torch.sum(torch.square(rotation_as_translation2[i]))))

            # import IPython
            # IPython.embed()

            starts.append(cas1)
            ends.append(final_pts1)
            starts.append(cas2)
            ends.append(final_pts2)


            full_mask1 = torch.zeros(len(xyz), dtype=bool)
            full_mask1[:self.binderlen] = mask1
            deltas[full_mask1,:] += rotation_as_translation1

            full_mask2 = torch.zeros(len(xyz), dtype=bool)
            full_mask2[:self.binderlen] = mask2
            deltas[full_mask2,:] += rotation_as_translation2


        # starts = torch.cat(starts)
        # ends = torch.cat(ends)

        # dump_lines(starts, ends-starts, 10, "bend_vectors.pdb")
        # dump_pts(coms, "coms.pdb")

        return deltas #* self.weight



class init_on_hemicircle(Potential):
    '''
        shove the coordinates into a hemicircle lol


        Author: bcov
    '''

    def __init__(self, binderlen, scale=1, start_pt=None, stop_pt=None, plane_pt=None, centered_at_com=0, maintain_noise=0,
                            sine_wave_scale=0, sine_wave_cycles=8, num_frames=1, maintain_rog=0):

        self.binderlen   = binderlen
        self.scale       = scale
        self.start_pt    = start_pt
        self.stop_pt     = stop_pt
        self.plane_pt    = plane_pt
        self.centered_at_com = bool(centered_at_com)
        self.maintain_noise = bool(maintain_noise)
        self.sine_wave_scale = sine_wave_scale
        self.sine_wave_cycles = sine_wave_cycles
        self.num_frames  = num_frames
        self.maintain_rog  = bool(num_frames)

        assert(not self.start_pt is None)
        assert(not self.stop_pt is None)
        assert(not self.plane_pt is None)

        # assert( len(self.spans) == len(self.pt_strings))

    def compute(self, seq, xyz):
        
        # constant because bcov is bad at torch
        return xyz[0,0,0] / xyz[0,0,0]



    def raw_delta_ca(self, seq, xyz, t, T):

        deltas = torch.zeros((len(xyz), 3))

        if ( t < T - (self.num_frames -1) ):
            return deltas


        # Extract binder Ca residues
        Ca_b = xyz[:self.binderlen,1] # [Lb,3]

        # Extract target Ca residues
        Ca_t = xyz[self.binderlen:,1] # [Lt,3]



        binder_com = torch.mean(Ca_b, axis=0)


        start_pt = parse_pt_string(self.start_pt, Ca_t)
        stop_pt = parse_pt_string(self.stop_pt, Ca_t)
        plane_pt = parse_pt_string(self.plane_pt, Ca_t)


        circle_center = (start_pt + stop_pt)/2

        rotation_vec = torch.cross( start_pt - circle_center, plane_pt - circle_center )
        rotation_vec /= torch.sqrt(torch.sum(torch.square(rotation_vec)))

        ind_radians = torch.linspace(0, np.pi, len(Ca_b))

        rotation_matrices = scipy.spatial.transform.Rotation.from_rotvec( rotation_vec[None,:] * ind_radians[:,None] ).as_matrix()
        rotation_matrices = torch.tensor(rotation_matrices, dtype=xyz.dtype)

        # first we construct a unit hemicircle around the origin


        start_unit = start_pt - circle_center
        real_hemicircle_size = torch.sqrt(torch.sum(torch.square(start_unit)))
        start_unit /= real_hemicircle_size

        unit_hemicircle = torch.zeros((len(Ca_b), 3), dtype=xyz.dtype)

        for i in range(len(Ca_b)):
            unit_hemicircle[i] = rotation_matrices[i] @ start_unit



        scaled_hemicircle = unit_hemicircle * real_hemicircle_size * self.scale






        if ( self.sine_wave_scale > 0 ):
            max_sine_angle = self.sine_wave_cycles * 2 * np.pi
            sine_angles = torch.linspace(0, max_sine_angle, len(Ca_b))
            sine_fracs = torch.cos(sine_angles)

            scaled_sine = sine_fracs * real_hemicircle_size * self.sine_wave_scale * self.scale

            sine_translations = rotation_vec[None,:] * scaled_sine[:,None]

            real_hemicircle += sine_translations


        scaled_com = torch.mean(scaled_hemicircle, axis=0)
        old_rog2 = torch.sum( torch.square( Ca_b - binder_com ) ) / len( Ca_b )

        if ( self.maintain_rog ):


            tol = 0.001
            scalar = 100.0
            cur_scale = 0.5

            # this makes sure that the hemicircle doesn't move the com for this calculation
            this_center = binder_com - scaled_com

            # binary search in exp space
            for i in range(1000):
                if ( self.maintain_noise ):
                    to_add = (scaled_hemicircle - scaled_com[None,:]) * cur_scale
                else:
                    to_add = scaled_hemicircle * cur_scale + this_center[None,:] - Ca_b

                this_final = Ca_b + to_add
                this_com = torch.mean(this_final, axis=0)
                this_rog2 = torch.sum( torch.square( this_final - this_com) ) / len( Ca_b )

                print("%8.1f %8.1f %8.5f %8.5f"%(this_rog2, old_rog2, cur_scale, scalar))
                if ( abs(1 - this_rog2 / old_rog2) < tol ):
                    break

                if ( this_rog2 > old_rog2 ):
                    cur_scale /= scalar
                else:
                    cur_scale *= scalar

                scalar = np.sqrt(scalar)

            scaled_hemicircle *= cur_scale



        using_center = binder_com - scaled_com if self.centered_at_com else circle_center
        real_hemicircle = scaled_hemicircle + using_center


        if ( self.maintain_noise ):
            translations = real_hemicircle - binder_com[None,:]
        else:
            translations = real_hemicircle - Ca_b


        if ( self.maintain_rog ):

            test_coords = translations + Ca_b
            new_com = torch.mean(test_coords, axis=0)
            new_rog2 = torch.sum(torch.square(test_coords - new_com)) / len( Ca_b )

            assert( abs( 1 - new_rog2 / old_rog2 ) < 0.1 )


        print("Hemisphere t", t)


        full_mask = torch.zeros(len(xyz), dtype=bool)
        full_mask[:self.binderlen] = mask
        deltas[full_mask,:] += rotation_as_translation

        # starts = torch.cat(starts)
        # ends = torch.cat(ends)

        # dump_lines(starts, ends-starts, 10, "bend_vectors.pdb")
        dump_pts([start_pt, plane_pt, stop_pt], "my_points.pdb")
        dump_pts(real_hemicircle, "my_hemicircle.pdb")

        return deltas #* self.weight








class init_on_hemicylinder(Potential):
    '''
        shove the coordinates into a hemicylinder while maintaining rog and com

        scale is what fraction of the way to move them, scale=1 results in all points on a hemicylinder

        Author: bcov
    '''

    def __init__(self, binderlen, scale=1, start_pt=None, stop_pt=None, plane_pt=None, num_frames=1, num_degrees=180, radius=None):

        self.binderlen   = binderlen
        self.scale       = scale
        self.start_pt    = start_pt
        self.stop_pt     = stop_pt
        self.plane_pt    = plane_pt
        self.num_frames  = num_frames
        self.num_degrees = num_degrees
        self.radius      = radius

        assert(not self.start_pt is None)
        assert(not self.stop_pt is None)
        assert(not self.plane_pt is None)

        # assert( len(self.spans) == len(self.pt_strings))

    def compute(self, seq, xyz):
        
        # constant because bcov is bad at torch
        return xyz[0,0,0] / xyz[0,0,0]



    def raw_delta_ca(self, seq, xyz, t, T):

        deltas = torch.zeros((len(xyz), 3))

        if ( t < T - (self.num_frames -1) ):
            return deltas


        # Extract binder Ca residues
        Ca_b = xyz[:self.binderlen,1] # [Lb,3]

        # Extract target Ca residues
        Ca_t = xyz[self.binderlen:,1] # [Lt,3]



        binder_com = torch.mean(Ca_b, axis=0)


        start_pt = parse_pt_string(self.start_pt, Ca_t)
        stop_pt = parse_pt_string(self.stop_pt, Ca_t)
        plane_pt = parse_pt_string(self.plane_pt, Ca_t)


        circle_center = (start_pt + stop_pt)/2

        rotation_vec = torch.cross( start_pt - circle_center, plane_pt - circle_center )
        rotation_vec /= torch.sqrt(torch.sum(torch.square(rotation_vec)))

        if ( self.num_degrees == 180 ):
            ind_radians = torch.linspace(0, np.pi, len(Ca_b))
        else:
            excess_radians = np.radians(self.num_degrees - 180 ) / 2
            ind_radians = torch.linspace(-excess_radians, np.pi + excess_radians, len(Ca_b))

        rotation_matrices = scipy.spatial.transform.Rotation.from_rotvec( rotation_vec[None,:] * ind_radians[:,None] ).as_matrix()
        rotation_matrices = torch.tensor(rotation_matrices, dtype=xyz.dtype)

        # first we construct a unit hemicircle around the origin


        start_unit = start_pt - circle_center
        real_hemicircle_size = torch.sqrt(torch.sum(torch.square(start_unit)))
        start_unit /= real_hemicircle_size

        unit_hemicircle = torch.zeros((len(Ca_b), 3), dtype=xyz.dtype)

        for i in range(len(Ca_b)):
            unit_hemicircle[i] = rotation_matrices[i] @ start_unit


        def plane_rog2(pts, vector, power=1):

            com = torch.mean(pts, axis=0)

            from_com = pts - com[None,:]

            projections = vector[None,:] * torch.sum( from_com * vector[None,:], axis=-1)[:,None]
            rejections = from_com - projections

            assert( torch.allclose( torch.mean(rejections, axis=0), torch.zeros(3, dtype=rejections.dtype), atol=0.001 ) )

            rog2 = torch.sum( torch.pow( torch.abs(rejections), power ) ) / len(pts)

            return rog2

        def interval_com_plane_rog2(input_pts, vector, power=2, sections=10, actually_plane=False):

            com = torch.mean(input_pts, axis=0)

            ranges = np.linspace(0, sections-0.000001, len(input_pts)).astype(int)
            pts = torch.zeros((len(input_pts), 3), dtype=input_pts.dtype)

            for i in range(sections):
                pts[i] = torch.mean(input_pts[ranges == i])

            from_com = pts - com[None,:]

            projections = vector[None,:] * torch.sum( from_com * vector[None,:], axis=-1)[:,None]

            if ( actually_plane ):
                rejections = from_com - projections
            else:
                rejections = from_com

            # assert( torch.allclose( torch.mean(rejections, axis=0), torch.zeros(3, dtype=rejections.dtype), atol=0.001 ) )

            rog2 = torch.sum( torch.pow( torch.abs(rejections), power ) ) / len(pts)

            return rog2


        # we adjust the hemicircle size until moving the scale-fraction of the distance
        #  results in the correct rog



        # scaled_com = torch.mean(unit_hemicircle, axis=0)
        old_rog2 = interval_com_plane_rog2( Ca_b, rotation_vec ) #torch.sum( torch.square( Ca_b - binder_com ) ) / len( Ca_b )




        # the distance along rotation_vec from the binder_com plane will never change
        scaled_com_to_Ca_b = (Ca_b - binder_com) 
        Ca_b_cyl_height = torch.sum( scaled_com_to_Ca_b * rotation_vec[None,:], axis=-1 )

        # shift_amount = 6
        # linear_shift = torch.linspace(-1, 1, len(Ca_b))[:,None] * rotation_vec[None,:]

        tol = 0.001
        scalar = 100.0
        cur_scale = 0.5

        # try to make the point cluster not change size by interval_com_plane_rog2
        # unless radius is set in which case just use that

        # binary search in exp space
        for i in range(1000):

            # now, we need to make the com(translations + Ca_b) == com( Ca_b )

            # com( (goal_pts - Ca_b ) * self.scale + Ca_b ) = com( Ca_b )
            # com( (goal_pts - Ca_b) * self.scale ) = 0
            # com( goal_pts ) = com( Ca_b )

            if ( self.radius ):
                cur_scale = 3

            scaled_hemicircle = unit_hemicircle * cur_scale
            scaled_com = torch.mean( scaled_hemicircle, axis=0)
            scaled_hemicircle += binder_com - scaled_com

            goal_pts = scaled_hemicircle + Ca_b_cyl_height[:,None] * rotation_vec[None,:]

            translations = (goal_pts - Ca_b) * self.scale #+ linear_shift

            prospective_pts = translations + Ca_b
            prospective_com = torch.mean(prospective_pts, axis=0)

            assert( torch.allclose( prospective_com, binder_com, atol=0.001 ) )

            prospective_rog2 = interval_com_plane_rog2( prospective_pts, rotation_vec ) #torch.sum( torch.square( prospective_pts - prospective_com) ) / len( Ca_b )

            # print("%8.1f %8.1f %8.5f %8.5f"%(prospective_rog2, old_rog2, cur_scale, scalar))
            if ( abs(1 - prospective_rog2 / old_rog2) < tol ):
                break

            if ( self.radius ):
                break

            if ( prospective_rog2 > old_rog2 ):
                cur_scale /= scalar
            else:
                cur_scale *= scalar

            scalar = np.sqrt(scalar)


        # dump_pts(goal_pts, "hemicylinder_goal.pdb")

        deltas[:self.binderlen,:] += translations

        return deltas #* self.weight





# Dictionary of types of potentials indexed by name of potential. Used by PotentialManager.
# If you implement a new potential you must add it to this dictionary for it to be used by
# the PotentialManager
implemented_potentials = { 'monomer_ROG':          monomer_ROG,
                           'binder_ROG':           binder_ROG,
                           'binder_distance_ReLU': binder_distance_ReLU,
                           'binder_any_ReLU':      binder_any_ReLU,
                           'dimer_ROG':            dimer_ROG,
                           'binder_ncontacts':     binder_ncontacts,
                           'dimer_ncontacts':      dimer_ncontacts,
                           'interface_ncontacts':  interface_ncontacts,
                           'monomer_contacts':     monomer_contacts,
                           'olig_intra_contacts':  olig_intra_contacts,
                           'olig_contacts':        olig_contacts,
                           'move_span_com':        move_span_com,
                           'linearize_spans':      linearize_spans,
                           'bend_adj_spans':       bend_adj_spans,
                           'init_on_hemicircle':   init_on_hemicircle,
                           'init_on_hemicylinder': init_on_hemicylinder,
                           }

require_binderlen      = { 'binder_ROG',
                           'binder_distance_ReLU',
                           'binder_any_ReLU',
                           'dimer_ROG',
                           'binder_ncontacts',
                           'dimer_ncontacts',
                           'interface_ncontacts',
                           'move_span_com',
                           'linearize_spans',
                           'bend_adj_spans',
                           'init_on_hemicylinder',
                           }

require_hotspot_res    = { 'binder_distance_ReLU',
                           'binder_any_ReLU' }

