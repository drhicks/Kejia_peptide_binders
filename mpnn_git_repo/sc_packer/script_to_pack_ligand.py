from __future__ import print_function
import json, time, os, sys, glob
import shutil

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import random
from dateutil import parser
import csv
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint
import queue


import random
from dateutil import parser
import csv
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint
import queue

import torch.distributions as D

sys.path.append("/home/justas/openfold")


import sys

#Arguments
input_path = sys.argv[1]
output_path = sys.argv[2]

import math

import numpy as np
import torch
import torch.nn as nn
from typing import Dict


from openfold.np import protein
import openfold.np.residue_constants as rc
from openfold.utils.rigid_utils import Rotation, Rigid
from openfold.utils.tensor_utils import (
     batched_gather,
     one_hot,
     tree_map,
     tensor_tree_map,
 )

from openfold.utils import feats

from openfold.data.data_transforms import make_atom14_masks, atom37_to_torsion_angles

from openfold.np.residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
)

from openfold.np.protein import Protein, to_pdb, from_pdb_string


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


PATH = '/projects/ml/struc2seq/data_for_complexes/training_scripts/2022/latest_sc/model_w/ligand/lig_002.pt'


epochs_to_train = 1

num_mix = 3

num_examples = 50000000

batch_size_valid = 5000
max_length_valid = 5000

NUM_ATOMS = 25
K_DNA = 10
NUM_H = 128
num_encoder_layers=3
num_decoder_layers=3
k_neighbors=48
dropout=0.1
augment_eps=0.00
Y_eps = 0.00
class StructureDataset_old():
    def __init__(self, jsonl_file, verbose=True, truncate=None, max_length=100,
        alphabet='ACDEFGHIKLMNPQRSTVWYX-'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        with open(jsonl_file) as f:
            self.data = []

            lines = f.readlines()
            start = time.time()
            for i, line in enumerate(lines):
                entry = json.loads(line)
                seq = entry['seq']
                name = entry['name']

                # Check if in alphabet
                bad_chars = set([s for s in seq]).difference(alphabet_set)
                if len(bad_chars) == 0:
                    if len(entry['seq']) <= max_length:
                        self.data.append(entry)
                    else:
                        discard_count['too_long'] += 1
                else:
                    #print(name, bad_chars, entry['seq'])
                    discard_count['bad_chars'] += 1

                # Truncate early
                if truncate is not None and len(self.data) == truncate:
                    return

                if verbose and (i + 1) % 1000 == 0:
                    elapsed = time.time() - start
                    #print('{} entries ({} loaded) in {:.1f} s'.format(len(self.data), i+1, elapsed))

            #print('Discarded', discard_count)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

#Load data from a list of dictionaries
class StructureDataset():
    def __init__(self, pdb_dict_list, verbose=True, truncate=None, max_length=100,
        alphabet='ACDEFGHIKLMNPQRSTVWYX-'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        self.data = []

        start = time.time()
        for i, entry in enumerate(pdb_dict_list):
            seq = entry['seq']
            name = entry['name']

            bad_chars = set([s for s in seq]).difference(alphabet_set)
            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length:
                    self.data.append(entry)
                else:
                    discard_count['too_long'] += 1
            else:
                #print(name, bad_chars, entry['seq'])
                discard_count['bad_chars'] += 1

            # Truncate early
            if truncate is not None and len(self.data) == truncate:
                return

            if verbose and (i + 1) % 1000 == 0:
                elapsed = time.time() - start
                #print('{} entries ({} loaded) in {:.1f} s'.format(len(self.data), i+1, elapsed))

            #print('Discarded', discard_count)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

#Data loader
class StructureLoader():
    def __init__(self, dataset, batch_size=100, shuffle=True,
        collate_fn=lambda x:x, drop_last=False):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch
#------------------------------
#MAKING PDB ASSEMBLIES

ref_atype_to_element = {'CNH2': 'C', 'COO': 'C', 'CH0': 'C', 'CH1': 'C', 'CH2': 'C', 'CH3': 'C', 'aroC': 'C', 'Ntrp': 'N', 'Nhis': 'N', 'NtrR': 'N', 'NH2O': 'N', 'Nlys': 'N', 'Narg': 'N', 'Npro': 'N', 'OH': 'O', 'OW': 'O', 'ONH2': 'O', 'OOC': 'O', 'Oaro': 'O', 'Oet2': 'O', 'Oet3': 'O', 'S': 'S', 'SH1': 'S', 'Nbb': 'N', 'CAbb': 'C', 'CObb': 'C', 'OCbb': 'O', 'Phos': 'P', 'Pbb': 'P', 'Hpol': 'H', 'HS': 'H', 'Hapo': 'H', 'Haro': 'H', 'HNbb': 'H', 'Hwat': 'H', 'Owat': 'O', 'Opoint': 'O', 'HOH': 'O', 'Bsp2': 'B', 'F': 'F', 'Cl': 'CL', 'Br': 'BR', 'I': 'I', 'Zn2p': 'ZN', 'Co2p': 'CO', 'Cu2p': 'CU', 'Fe2p': 'FE', 'Fe3p': 'FE', 'Mg2p': 'MG', 'Ca2p': 'CA', 'Pha': 'P', 'OPha': 'O', 'OHha': 'O', 'Hha': 'H', 'CO3': 'C', 'OC3': 'O', 'Si': 'Si', 'OSi': 'O', 'Oice': 'O', 'Hice': 'H', 'Na1p': 'NA', 'K1p': 'K', 'He': 'HE', 'Li': 'LI', 'Be': 'BE', 'Ne': 'NE', 'Al': 'AL', 'Ar': 'AR', 'Sc': 'SC', 'Ti': 'TI', 'V': 'V', 'Cr': 'CR', 'Mn': 'MN', 'Ni': 'NI', 'Ga': 'GA', 'Ge': 'GE', 'As': 'AS', 'Se': 'SE', 'Kr': 'KR', 'Rb': 'RB', 'Sr': 'SR', 'Y': 'Y', 'Zr': 'ZR', 'Nb': 'NB', 'Mo': 'MO', 'Tc': 'TC', 'Ru': 'RU', 'Rh': 'RH', 'Pd': 'PD', 'Ag': 'AG', 'Cd': 'CD', 'In': 'IN', 'Sn': 'SN', 'Sb': 'SB', 'Te': 'TE', 'Xe': 'XE', 'Cs': 'CS', 'Ba': 'BA', 'La': 'LA', 'Ce': 'CE', 'Pr': 'PR', 'Nd': 'ND', 'Pm': 'PM', 'Sm': 'SM', 'Eu': 'EU', 'Gd': 'GD', 'Tb': 'TB', 'Dy': 'DY', 'Ho': 'HO', 'Er': 'ER', 'Tm': 'TM', 'Yb': 'YB', 'Lu': 'LU', 'Hf': 'HF', 'Ta': 'TA', 'W': 'W', 'Re': 'RE', 'Os': 'OS', 'Ir': 'IR', 'Pt': 'PT', 'Au': 'AU', 'Hg': 'HG', 'Tl': 'TL', 'Pb': 'PB', 'Bi': 'BI', 'Po': 'PO', 'At': 'AT', 'Rn': 'RN', 'Fr': 'FR', 'Ra': 'RA', 'Ac': 'AC', 'Th': 'TH', 'Pa': 'PA', 'U': 'U', 'Np': 'NP', 'Pu': 'PU', 'Am': 'AM', 'Cm': 'CM', 'Bk': 'BK', 'Cf': 'CF', 'Es': 'ES', 'Fm': 'FM', 'Md': 'MD', 'No': 'NO', 'Lr': 'LR', 'SUCK': 'Z', 'REPL': 'Z', 'REPLS': 'Z', 'HREPS': 'Z', 'VIRT': 'X', 'MPct': 'X', 'MPnm': 'X', 'MPdp': 'X', 'MPtk': 'X'}

chem_elements = ['C','N','O','P','S','AC','AG','AL','AM','AR','AS','AT','AU','B','BA','BE','BI','BK','BR','CA','CD','CE','CF','CL','CM','CO','CR','CS','CU','DY','ER','ES','EU','F','FE','FM','FR','GA','GD','GE','H','HE','HF','HG','HO','I','IN','IR','K','KR','LA','LI','LR','LU','MD','MG','MN','MO','NA','NB','ND','NE','NI','NO','NP','OS','PA','PB','PD','PM','PO','PR','PT','PU','RA','RB','RE','RH','RN','RU','SB','SC','SE','SM','SN','SR','Si','TA','TB','TC','TE','TH','TI','TL','TM','U','V','W','X','XE','Y','YB','Z','ZN','ZR']


ref_atypes_dict = dict(zip(chem_elements, range(len(chem_elements))))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, IDs, loader, train_dict, params):
        self.IDs = IDs
        self.train_dict = train_dict
        self.loader = loader
        self.params = params

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        sel_idx = np.random.randint(0, len(self.train_dict[ID]))
        out = self.loader(self.train_dict[ID][sel_idx], self.params)
        return out


pdb_dict_valid = []
with open(input_path, 'r') as json_file:
    json_list = list(json_file)
for json_str in json_list:
    dict_out = json.loads(json_str)
    pdb_dict_valid.append(dict_out)


dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=max_length_valid)
loader_valid =StructureLoader(dataset_valid, batch_size=batch_size_valid)


def get_DNA_mask(X, mask, Y, Y_m,  eps=1e-6):
    #Y - shape - [B, L, num_atoms, 3]
    #Y_m - shape - [B, L, num_atoms]
    b = X[:,:,1,:] - X[:,:,0,:]
    c = X[:,:,2,:] - X[:,:,1,:]
    a = torch.cross(b, c, dim=-1)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
    
    D = torch.sqrt(torch.sum((Cb[:,:,None,:] - Y)**2,-1) + 1e-6) #[B, L, num_atoms]
    D_max = 1000.0
    D_adjust = D + D_max * (1. - Y_m) + D_max * (1. - mask[:,:,None])
    DNA_mask = ((D_adjust < 8.0).sum(-1) > 0).float()
    return DNA_mask



def get_pdbs(data_loader, repeat=1, max_length=10000, num_units=1000000):
    dna_list = 'atcg'
    rna_list = 'dryu'
    ligand_list = 'J'
    protein_list = 'ARNDCQEGHILKMFPSTWYVX'
    protein_list_check = 'ARNDCQEGHILKMFPSTWYV'

    dna_rna_list = dna_list + rna_list + 'X'
    dna_rna_num = dict(zip(dna_rna_list, range(len(dna_rna_list))))

    dna_rna_dict = {
    "a" : ['P', 'OP1', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", 'N1', 'C2', 'N3', 'C4', 'C5', 'C6', 'N7', 'C8','N9', "", ""],
    "t" : ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "C5", "C6", "C7", "", "", ""],
    "c" : ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6", "", "", ""],
    "g" : ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "N2", "N3", "C4", "C5", "C6", "N7", "C8", "N9", ""],
    "d" : ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9", ""],
    "r" : ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6", "", ""],
    "y" : ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6", "", ""],
    "u" : ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "N2", "N3", "C4", "C5", "C6", "N7", "C8", "N9"],
    "X" : 22*[""]}

    dna_rna_atom_types = np.array(["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "N2", "N3", "C4", "C5", "C6", "N7", "C8", "N9", "O4", "O2", "N4", "C7", ""])

    idxAA_22_to_27 = np.zeros((9, 22), np.int32)
    for i, AA in enumerate(dna_rna_dict.keys()):
        for j, atom in enumerate(dna_rna_dict[AA]):
            idxAA_22_to_27[i,j] = int(np.argwhere(atom==dna_rna_atom_types)[0][0])



    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet
    c = 0
    c1 = 0
    pdb_dict_list = []
    t0 = time.time()

    RES_NAMES_1 = 'ARNDCQEGHILKMFPSTWYVX'
    
    atom_types = np.array([
        'N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
        'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
        'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
        'CZ3', 'NZ', ''])
    
    restype_name_to_atom14_names = {
        'ALA': ['N', 'CA', 'C', 'O', 'CB', '',    '',    '',    '',    '',    '',    '',    '',    ''],
        'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2', '',    '',    ''],
        'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'ND2', '',    '',    '',    '',    '',    ''],
        'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'OD2', '',    '',    '',    '',    '',    ''],
        'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG',  '',    '',    '',    '',    '',    '',    '',    ''],
        'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'NE2', '',    '',    '',    '',    ''],
        'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'OE2', '',    '',    '',    '',    ''],
        'GLY': ['N', 'CA', 'C', 'O', '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],
        'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2', '',    '',    '',    ''],
        'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '',    '',    '',    '',    '',    ''],
        'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', '',    '',    '',    '',    '',    ''],
        'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'CE',  'NZ',  '',    '',    '',    '',    ''],
        'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'SD',  'CE',  '',    '',    '',    '',    '',    ''],
        'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  '',    '',    ''],
        'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  '',    '',    '',    '',    '',    '',    ''],
        'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG',  '',    '',    '',    '',    '',    '',    '',    ''],
        'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
        'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
        'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH',  '',    ''],
        'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
        'UNK': ['',  '',   '',  '',  '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],
    }
    
    RES_NAMES = [
        'ALA','ARG','ASN','ASP','CYS',
        'GLN','GLU','GLY','HIS','ILE',
        'LEU','LYS','MET','PHE','PRO',
        'SER','THR','TRP','TYR','VAL', 'UNK',
    ]
    
    idxAA_14_to_37 = np.zeros((21, 14), np.int32)
    for i, AA in enumerate(RES_NAMES):
        for j, atom in enumerate(restype_name_to_atom14_names[AA]):
            idxAA_14_to_37[i,j] = int(np.argwhere(atom==atom_types)[0][0])

    for _ in range(repeat):
        for step,t in enumerate(data_loader):
            if step > num_units:
                break
            if type(t) == dict:
                t = {k:v[0] for k,v in t.items()}
                c1 += 1
                if 'label' in list(t):
                    my_dict = {}
                    s = 0
                    concat_seq = ''
                    concat_seq_DNA = ''
                    concat_N = []
                    concat_CA = []
                    concat_C = []
                    concat_O = []
                    concat_mask = []
                    coords_dict = {}
                    mask_list = []
                    visible_list = []
                    Cb_list = []
                    P_list = []
                    dna_atom_list = []
                    dna_atom_mask_list = []
                    chain_list = []
                    ligand_atom_list = []
                    ligand_atype_list = []
                    ligand_total_length = 0
                    if len(list(np.unique(t['idx']))) < 352:
                        for idx in list(np.unique(t['idx'])):
                            letter = chain_alphabet[idx]
                            res = np.argwhere(t['idx']==idx)
                            if res.shape[0] == 0:
                                continue
                            try:
                                initial_sequence= "".join(list(np.array(list(t['seq']))[res][0,]))
                            except:
                                continue
                            if initial_sequence[-6:] == "HHHHHH":
                                res = res[:,:-6]
                            if initial_sequence[0:6] == "HHHHHH":
                                res = res[:,6:]
                            if initial_sequence[-7:-1] == "HHHHHH":
                                res = res[:,:-7]
                            if initial_sequence[-8:-2] == "HHHHHH":
                                res = res[:,:-8]
                            if initial_sequence[-9:-3] == "HHHHHH":
                                res = res[:,:-9]
                            if initial_sequence[-10:-4] == "HHHHHH":
                                res = res[:,:-10]
                            if initial_sequence[1:7] == "HHHHHH":
                                res = res[:,7:]
                            if initial_sequence[2:8] == "HHHHHH":
                                res = res[:,8:]
                            if initial_sequence[3:9] == "HHHHHH":
                                res = res[:,9:]
                            if initial_sequence[4:10] == "HHHHHH":
                                res = res[:,10:]
                            if True:
                                seq_t = "".join(list(np.array(list(t['seq']))[res][0,]))
                                protein_seq_flag = any([(item in seq_t) for item in protein_list_check])
                                dna_seq_flag = any([(item in seq_t) for item in dna_list])
                                rna_seq_flag = any([(item in seq_t) for item in rna_list])
                                ligand_seq_flag = all([(item in seq_t) for item in ligand_list])
                                if protein_seq_flag:
                                    if idx == t['target_chain'] or type(t['target_chain'])==str:
                                        CA_coords = np.array(t['xyz'][res,])[0,:,1,:] #[L, 3]
                                        CA_mask = np.isfinite(np.sum(CA_coords,-1)) #[L,]
                                        res_true = np.argwhere(CA_mask == True)[:,0]
                                        if CA_mask.sum() == 0:
                                            pass
                                        else:
                                            my_dict['seq_chain_'+letter]= "".join(list(np.array(list(t['seq']))[res][0,res_true]))
                                            concat_seq += my_dict['seq_chain_'+letter]
                                            #if idx in t['masked']:
                                            mask_list.append(letter)
                                            #else:
                                            #    visible_list.append(letter)
                                            chain_list.append(letter)
                                            coords_dict_chain = {}
                                            idx_array = np.arange(CA_coords.shape[0])[res_true]
                                            all_atoms = np.array(t['xyz'][res,])[0,res_true,:14] #[L, 14, 3]
                                            seq_ = my_dict['seq_chain_'+letter]
                                            coords_dict_chain['all_atoms_chain_'+letter]=all_atoms.tolist() #do not use the last 37th entry
                                            coords_dict_chain['idx_chain_'+letter] = idx_array.tolist()
                                            my_dict['coords_chain_'+letter]=coords_dict_chain

                                            b = all_atoms[:,1,:] - all_atoms[:,0,:]
                                            c = all_atoms[:,2,:] - all_atoms[:,1,:]
                                            a = np.cross(b, c, -1)
                                            Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + all_atoms[:,1,:] #virtual
                                            Cb_list.append(Cb)
                                elif dna_seq_flag or rna_seq_flag:

                                    all_atoms = np.array(t['xyz'][res,])[0,]
                                    P_list.append(all_atoms[:,0,:])
                                    all_atoms_ones = np.ones((all_atoms.shape[0], 22))
                                    seq_ = "".join(list(np.array(list(t['seq']))[res][0,]))
                                    concat_seq_DNA += seq_
                                    all_atoms27 = np.empty((len(seq_),27,3))
                                    all_atoms27_mask = np.zeros((len(seq_), 27))
                                    all_atoms27[:] = np.NaN
                                    idx = np.array([idxAA_22_to_27[np.argwhere(AA==np.array(list(dna_rna_dict.keys())))[0][0]] for AA in seq_])
                                    np.put_along_axis(all_atoms27, idx[:,:,None], all_atoms, 1)
                                    np.put_along_axis(all_atoms27_mask, idx, all_atoms_ones, 1)
                                    dna_atom_list.append(all_atoms27)
                                    dna_atom_mask_list.append(all_atoms27_mask)

                                elif ligand_seq_flag:
                                    all_atoms = np.array(t['xyz'][res,])[0,]
                                    ligand_atype = np.array(t['atype'][res])[0,]
                                    if (1-np.isnan(all_atoms[:,0,:])).sum() != 0:
                                        tmp_idx = np.argwhere(1-np.isnan(all_atoms[:,0,:].mean(-1))==1.0)[-1][0] + 1
                                        ligand_atom_list.append(all_atoms[:tmp_idx,:1,:])
                                        ligand_atype_list.append(ligand_atype[:tmp_idx])
                                        ligand_total_length += tmp_idx
                        if len(P_list) > 0 and len(Cb_list) > 0:
                            Cb_stack = np.concatenate(Cb_list, 0) #[L, 3]
                            P_stack = np.concatenate(P_list, 0) #[K, 3]
                            dna_atom_stack = np.concatenate(dna_atom_list, 0)
                            dna_atom_mask_stack = np.concatenate(dna_atom_mask_list, 0)
                            DNA_seq_num = np.array([dna_rna_num[letter] for letter in concat_seq_DNA])
                            D = np.sqrt(((Cb_stack[:,None,:]-P_stack[None,:,:])**2).sum(-1) + 1e-7)
                            idx_dna = np.argsort(D,-1)[:,:K_DNA] #top 10 neighbors per residue

                            dna_atom_selected = dna_atom_stack[idx_dna]
                            dna_atom_mask_selected = dna_atom_mask_stack[idx_dna]
                            my_dict['dna_context'] = dna_atom_selected[:,:,:-1,:].tolist()
                            my_dict['dna_context_mask'] = dna_atom_mask_selected[:,:,:-1].tolist()
                        else:
                            my_dict['dna_context'] = 'no_DNA'
                            my_dict['dna_context_mask'] = 'no_DNA'
                        if ligand_atom_list:
                            ligand_atom_stack = np.concatenate(ligand_atom_list, 0)
                            ligand_atype_stack = np.concatenate(ligand_atype_list, 0)
                            my_dict['ligand_context'] = ligand_atom_stack.tolist()
                            my_dict['ligand_atype'] = ligand_atype_stack.tolist()
                        else:
                            my_dict['ligand_context'] = 'no_ligand'
                            my_dict['ligand_atype'] = 'no_ligand'
                        my_dict['ligand_length'] = ligand_total_length
                        my_dict['name']= t['label']
                        my_dict['chain_list'] = chain_list
                        my_dict['masked_list']= mask_list
                        my_dict['visible_list']= visible_list
                        my_dict['num_of_chains'] = len(mask_list) + len(visible_list)
                        my_dict['seq'] = concat_seq
                        my_dict['seq_DNA'] = concat_seq_DNA
                        if len(concat_seq) <= max_length:
                            if concat_seq:
                                pdb_dict_list.append(my_dict)
    return pdb_dict_list


def featurize(batch, device, train_all=False):
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    alphabet_af2 = 'ARNDCQEGHILKMFPSTWYVX'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32) #sum of chain seq lengths
    L_max = max([len(b['seq']) for b in batch])

    L_ligand_max = max([int(b['ligand_length']) for b in batch]) + 1
    Z = np.zeros(shape=(B,  L_ligand_max, 3))
    Z_m = np.zeros(shape=(B,  L_ligand_max))
    Z_t = np.zeros(shape=(B,  L_ligand_max))
    X = np.zeros([B, L_max, 37, 3])
    Y = np.zeros([B, L_max, K_DNA, 26, 3])
    Y_m = np.zeros([B, L_max, K_DNA, 26])
    residue_idx = -100*np.ones([B, L_max], dtype=np.int32) #residue idx with jumps across chains
    chain_M = np.zeros([B, L_max], dtype=np.int32) #1.0 for the bits that need to be predicted, 0.0 for the bits that are given
    mask_self = np.ones([B, L_max, L_max], dtype=np.int32) #for interface loss calculation - 0.0 for self interaction, 1.0 for other
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32) #integer encoding for chains 0, 0, 0,...0, 1, 1,..., 1, 2, 2, 2...
    S = np.zeros([B, L_max], dtype=np.int32) #sequence AAs integers
    S_af2 = np.zeros([B, L_max], dtype=np.int32) #sequence AAs integers
    L_ligand_list = []
    for i, b in enumerate(batch):
        context = b['dna_context']
        context_mask = b['dna_context_mask']

        ligand_context = b['ligand_context']
        ligand_types = b['ligand_atype']

        if context != 'no_DNA':
            y = np.array(context)
            y_m = np.array(context_mask)
        else:
            y = np.zeros([L_max, K_DNA, 26, 3])
            y_m = np.zeros([L_max, K_DNA, 26])

        if ligand_context != 'no_ligand':
            z = np.array(ligand_context)[:,0,:]
            z_t = np.array(ligand_types)
        else:
            z = np.full([L_ligand_max, 3], np.nan)
            z_t = np.zeros([L_ligand_max])

        masked_chains = b['masked_list']
        visible_chains = b['visible_list']
        all_chains = b['chain_list']
        #random.shuffle(all_chains) #randomly shuffle chain order
        num_chains = b['num_of_chains']
        mask_dict = {}
        x_chain_list = []
        x_m_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        l0 = 0
        l1 = 0
        if train_all:
            for step, letter in enumerate(all_chains):
                if letter in visible_chains:
                    chain_seq = b[f'seq_chain_{letter}']
                    chain_seq = ''.join([a if a!='-' else 'X' for a in chain_seq])

                    chain_length = len(chain_seq)
                    chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                    chain_mask = np.ones(chain_length) #0.0 for visible chains

                    x_chain = np.stack(chain_coords[f'all_atoms_chain_{letter}'], 1).transpose(1,0,2) #[L, 36, 3]
                    x_chain_list.append(x_chain)

                    chain_mask_list.append(chain_mask)
                    chain_seq_list.append(chain_seq)
                    chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                    l1 += chain_length
                    mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                    
                    idx_ = np.array(chain_coords[f'idx_chain_{letter}'])
                    residue_idx[i, l0:l1] = 100*(c-1) + idx_ - idx_[0] + (l0-1>0)*(residue_idx[i, l0-1]+1)

                    l0 += chain_length
                    c+=1
                elif letter in masked_chains:
                    chain_seq = b[f'seq_chain_{letter}']
                    chain_seq = ''.join([a if a!='-' else 'X' for a in chain_seq])

                    chain_length = len(chain_seq)
                    chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                    chain_mask = np.ones(chain_length) #0.0 for visible chains
                    
                    x_chain = np.stack(chain_coords[f'all_atoms_chain_{letter}'], 1).transpose(1,0,2) #[L, 36, 3]
                    x_chain_list.append(x_chain)

                    chain_mask_list.append(chain_mask)
                    chain_seq_list.append(chain_seq)
                    chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                    l1 += chain_length
                    mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                    
                    idx_ = np.array(chain_coords[f'idx_chain_{letter}'])
                    residue_idx[i, l0:l1] = 100*(c-1) + idx_ - idx_[0] + (l0-1>0)*(residue_idx[i, l0-1]+1)
                    
                    l0 += chain_length
                    c+=1        
        else:
            for step, letter in enumerate(all_chains):
                if letter in visible_chains:
                    chain_seq = b[f'seq_chain_{letter}']
                    chain_seq = ''.join([a if a!='-' else 'X' for a in chain_seq])

                    chain_length = len(chain_seq)
                    chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                    chain_mask = np.zeros(chain_length) #0.0 for visible chains
                    
                    x_chain = np.stack(chain_coords[f'all_atoms_chain_{letter}'], 1).transpose(1,0,2) #[L, 36, 3]
                    x_chain_list.append(x_chain)

                    chain_mask_list.append(chain_mask)
                    chain_seq_list.append(chain_seq)
                    chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                    l1 += chain_length
                    mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                    
                    idx_ = np.array(chain_coords[f'idx_chain_{letter}'])
                    residue_idx[i, l0:l1] = 100*(c-1) + idx_ - idx_[0] + (l0-1>0)*(residue_idx[i, l0-1]+1)

                    l0 += chain_length
                    c+=1
                elif letter in masked_chains: 
                    chain_seq = b[f'seq_chain_{letter}']
                    chain_seq = ''.join([a if a!='-' else 'X' for a in chain_seq])

                    chain_length = len(chain_seq)
                    chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                    chain_mask = np.ones(chain_length) #0.0 for visible chains
                    
                    x_chain = np.stack(chain_coords[f'all_atoms_chain_{letter}'], 1).transpose(1,0,2) #[L, 36, 3]
                    x_chain_list.append(x_chain)

                    chain_mask_list.append(chain_mask)
                    chain_seq_list.append(chain_seq)
                    chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                    l1 += chain_length
                    mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                    
                    idx_ = np.array(chain_coords[f'idx_chain_{letter}'])
                    residue_idx[i, l0:l1] = 100*(c-1) + idx_ - idx_[0] + (l0-1>0)*(residue_idx[i, l0-1]+1)
                    l0 += chain_length
                    c+=1
        x = np.concatenate(x_chain_list,0) #[L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(chain_mask_list,0) #[L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list,0)

        l = len(all_sequence)
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad

        y_pad = np.pad(y, [[0,L_max-y.shape[0]], [0,K_DNA-y.shape[1]], [0,0], [0, 0]], 'constant', constant_values=(np.nan, ))
        Y[i,:,:,:] = y_pad

        y_m_pad = np.pad(y_m, [[0,L_max-y.shape[0]], [0,K_DNA-y.shape[1]], [0,0]], 'constant', constant_values=(np.nan, ))
        Y_m[i,] = y_m_pad


        z_pad = np.pad(z, [[0,L_ligand_max-z.shape[0]], [0, 0]], 'constant', constant_values=(np.nan, ))
        Z[i,:,:] = z_pad


        z_t_pad = np.pad(z_t, [[0,L_ligand_max-z_t.shape[0]]], 'constant', constant_values=(0, ))
        Z_t[i,:] = z_t_pad

        m_pad = np.pad(m, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_M[i,:] = m_pad

        chain_encoding_pad = np.pad(chain_encoding, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_encoding_all[i,:] = chain_encoding_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
        S[i, :l] = indices

        indices = np.asarray([alphabet_af2.index(a) for a in all_sequence], dtype=np.int32)
        S_af2[i, :l] = indices


    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X[:,:,:3,:],(2,3))).astype(np.float32)
    mask_sc = np.isfinite(np.sum(X[:,:,:,:],-1)).astype(np.float32)
    #X[isnan] = 0.

    isnan = np.isnan(Y)
    Y[isnan] = 0.

    isnan = np.isnan(Y_m)
    Y_m[isnan] = 0.


    isnan = np.isnan(Z)
    Z_m = np.isfinite(np.sum(Z,-1)).astype(np.float32)
    Z[isnan] = 0.

    # Conversion
    jumps = ((residue_idx[:,1:]-residue_idx[:,:-1])==1).astype(np.float32)
    phi_mask = np.pad(jumps, [[0,0],[1,0]])
    psi_mask = np.pad(jumps, [[0,0],[0,1]])
    omega_mask = np.pad(jumps, [[0,0],[0,1]])
    dihedral_mask = np.concatenate([phi_mask[:,:,None], psi_mask[:,:,None], omega_mask[:,:,None]], -1) #[B,L,3]
    dihedral_mask = torch.from_numpy(dihedral_mask).to(dtype=torch.float32, device=device)
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long,device=device)
    S = torch.from_numpy(S).to(dtype=torch.long,device=device)
    S_af2 = torch.from_numpy(S_af2).to(dtype=torch.long,device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    Y = torch.from_numpy(Y).to(dtype=torch.float32, device=device)
    Y_m = torch.from_numpy(Y_m).to(dtype=torch.float32, device=device)

    Z = torch.from_numpy(Z).to(dtype=torch.float32, device=device)
    Z_m = torch.from_numpy(Z_m).to(dtype=torch.float32, device=device)
    Z_t = torch.from_numpy(Z_t).to(dtype=torch.int32, device=device)

    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    mask_sc = torch.from_numpy(mask_sc).to(dtype=torch.float32, device=device)
    mask_self = torch.from_numpy(mask_self).to(dtype=torch.float32, device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device)
    return X, mask_sc, S, S_af2, mask, lengths, chain_M, residue_idx, mask_self, dihedral_mask, chain_encoding_all, Y, Y_m, Z, Z_m, Z_t 


def sidechain_torsion_loss(true_dict, mean, concentration, mix_logits, external_mask):

    mix = D.Categorical(logits=mix_logits)
    comp = D.VonMises(mean, concentration)
    pred_dist = D.MixtureSameFamily(mix, comp)

    true_torsions = torch.clone(true_dict['torsion_angles_sin_cos'][:,:,3:])
    alt_torsions = torch.clone(true_dict['alt_torsion_angles_sin_cos'][:,:,3:])
    torsion_angle_mask = torch.clone(true_dict['torsion_angles_mask'][:,:,3:])

    true_torsions_mask = ~torch.isnan(true_torsions)
    true_torsions[~true_torsions_mask] = 0.0

    alt_torsions_mask = ~torch.isnan(alt_torsions)
    alt_torsions[~alt_torsions_mask] = 0.0

    torsions_mask = true_torsions_mask[:,:,:,0]*external_mask[:,:,None]*torsion_angle_mask

    true_torsion_angles = torch.atan2(true_torsions[:,:,:,-2], true_torsions[:,:,:,-1])
    alt_torsion_angles = torch.atan2(alt_torsions[:,:,:,-2], alt_torsions[:,:,:,-1])

    L1= -pred_dist.log_prob(true_torsion_angles)
    L2= -pred_dist.log_prob(alt_torsion_angles)

    Lmin = torch.minimum(L1,L2) #[B, L, 4]
    Ltorsion = torch.sum(Lmin*torsions_mask)/(2000.0*4.0)

    predicted_real_values = torch.gather(mean, -1, torch.argmax(mix_logits,-1)[...,None])[...,0]

    torsion_pred_unit = torch.cat([torch.sin(predicted_real_values[:,:,:,None]), torch.cos(predicted_real_values[:,:,:,None])], -1)

    angle_1 = torch.acos(torch.clamp(torch.einsum('blms, blms -> blm', true_torsions, torsion_pred_unit), min=-0.99, max=0.99))
    angle_2 = torch.acos(torch.clamp(torch.einsum('blms, blms -> blm', alt_torsions, torsion_pred_unit), min=-0.99, max=0.99))

    angle_min = torch.minimum(angle_1, angle_2)


    return Ltorsion, angle_min, torsions_mask, Lmin, true_torsion_angles, alt_torsion_angles, torsions_mask



#HELPER FUNCTIONS
def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    S_argmaxed = torch.argmax(log_probs,-1) #[B, L]
    true_false = (S == S_argmaxed).float()
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av, true_false


def loss_smoothed(S, log_probs, mask, weight=0.1):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S, 21).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / 2000.0 #fixed 
    return loss, loss_av


# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.reshape((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.reshape(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn

def get_interface_weights(N, Ca, C, mask, mask_self, top_k=5, eps=1e-6):
    "N, Ca, C - [B, L, 3], mask - [B, L], chain_Ls_list_list - [[64, 78, 40], [[65, 124], ...]"
    b = Ca - N
    c = C - Ca
    a = torch.cross(b, c, dim=-1)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca
    mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
    dX = torch.unsqueeze(Cb,1) - torch.unsqueeze(Cb,2)
    D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
    D_max = 1000.0
    D_adjust = D + D_max * (1. - mask_2D) + D_max * (1. - mask_self)
    interface_contacts_mask = ((D_adjust < 8.0).sum(-1) > 0).float()
    interface_weights = 1.0+0.0*interface_contacts_mask
    return interface_weights, interface_contacts_mask

#------------------------------
#------------------------------

#MODEL CLASSES
class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        #h_EV = self.norm1(h_EV)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        #dh = self.norm2(h_V)
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)

        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        #h_EV = self.norm3(h_EV)

        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))

        return h_V, h_E


class MPNNLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(MPNNLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V



class MPNNLayerJ(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(MPNNLayerJ, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,-1, h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V



class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden + num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, 2*num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)
        self.dense_msg = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_bw, h_encoder, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """


        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_EV = mask_bw * h_EV + h_encoder

        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        #h_EV = self.norm3(h_EV)

        h_message_edge = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message_edge))




        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_EV = mask_bw * h_EV + h_encoder

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        #h_EV = self.norm1(h_EV)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        #dh = self.norm2(h_V)
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        return h_V, h_E


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature)*mask + (1-mask)*(2*self.max_relative_feature+1)
        d_onehot = torch.nn.functional.one_hot(d, 2*self.max_relative_feature+1+1)
        E = self.linear(d_onehot.float())
        return E
    
class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, num_rbf_sc=8,top_k=30, augment_eps=0., num_chain_embeddings=16, device=None, side_residue_num=32, atom_context_num=25):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.num_rbf_side = num_rbf_sc
        self.atom_context_num = atom_context_num
        self.num_positional_embeddings = num_positional_embeddings
        self.side_residue_num = side_residue_num
        # Positional encoding
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        # Normalization and embedding
        node_in, edge_in = 6, num_positional_embeddings + num_rbf*25
        self.node_project_down = nn.Linear(5*num_rbf+105, node_features, bias=False)
        #self.node_embedding = nn.Linear(128, node_features, bias=False) #NOT USED
        self.j_nodes = nn.Linear(105, node_features, bias=False)
        self.j_edges = nn.Linear(num_rbf, node_features, bias=False)


        self.norm_j_edges = nn.LayerNorm(node_features)
        self.norm_j_nodes = nn.LayerNorm(node_features)


        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_nodes = nn.LayerNorm(node_features)
        self.norm_edges = nn.LayerNorm(edge_features)
        self.edge_embedding_s = nn.Linear(num_rbf_sc*31*5, edge_features, bias=False)

        #element_dict = {"C": 0, "N": 1, "O": 2, "P": 3, "S": 4}
        #dna_rna_atom_types = ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "N2", "N3", "C4", "C5", "C6", "N7", "C8", "N9", "O4", "O2", "N4", "C7"]
        #atom_list = [
        #'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
        #'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
        #'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
        #'CZ3', 'NZ', 'OXT']
        self.DNA_RNA_types = torch.tensor([3, 2, 2, 2, 0, 0, 2, 0, 2, 0, 2, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 2, 2, 1, 0], device=device)
        self.side_chain_atom_types = torch.tensor([1, 0, 0, 2, 0, 0, 0, 0, 2, 2, 4, 0, 0, 0, 1, 1, 2, 2, 4, 0, 0, 0, 0, 1, 1, 1, 2, 2, 0, 1, 1, 2, 0, 0, 0, 1, 2], device=device)
       
    def _dist(self, X, mask, top_k_sample=True, eps=1E-6):
        """ Pairwise euclidean distances """
        # Convolutional network on NCHW
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

        # Identify k nearest neighbors (including self)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        if top_k_sample:
            sampled_top_k = np.random.randint(32,self.top_k+1)
        else:
            sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)
        return D_neighbors, E_idx, mask_neighbors

    def _rbf_side(self, D):
        # Distance radial basis function
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf_side
        D_mu = torch.linspace(D_min, D_max, D_count).to(device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _rbf(self, D):
        # Distance radial basis function
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count).to(device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def _get_rbf_side(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf_side(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, Z, Z_m, Z_t, X, Y, Y_m, L, mask, atom_mask, residue_idx, dihedral_mask, chain_labels, top_k_sample=False, mask_for_sc=None):
        """ Featurize coordinates as an attributed graph """
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
            Y = Y + Y_eps * torch.randn_like(Y)        
            Z = Z + Y_eps * torch.randn_like(Z)

        b = X[:,:,1,:] - X[:,:,0,:]
        c = X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
        Ca = X[:,:,1,:]
        N = X[:,:,0,:]
        C = X[:,:,2,:]
        O = X[:,:,4,:]
 
        #N, Ca, C, O, Cb - are five atoms representing a residue

        #Get neighbors
        D_neighbors, E_idx, mask_neighbors = self._dist(Ca, mask, top_k_sample)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) #Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx)) #N-N
        RBF_all.append(self._get_rbf(C, C, E_idx)) #C-C
        RBF_all.append(self._get_rbf(O, O, E_idx)) #O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx)) #Cb-Cb

        RBF_all.append(self._get_rbf(Ca, N, E_idx)) #Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx)) #Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx)) #Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx)) #Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx)) #N-C
        RBF_all.append(self._get_rbf(N, O, E_idx)) #N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx)) #N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx)) #Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx)) #Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx)) #O-C

        RBF_all.append(self._get_rbf(N, Ca, E_idx)) #N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx)) #C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx)) #O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx)) #Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx)) #C-N
        RBF_all.append(self._get_rbf(O, N, E_idx)) #O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx)) #Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx)) #C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx)) #O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx)) #C-O



        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = residue_idx[:,:,None]-residue_idx[:,None,:]
        offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0] #[B, L, K]

        d_chains = 1+0*((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long()
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        E_positional = self.embeddings(offset, E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)



        E_idx_sub = E_idx[:,:,:self.side_residue_num] #[B, L, 15]
        RBF_sidechain = []
        dropout_mask = mask_for_sc
        atom_mask = atom_mask*dropout_mask[:,:,None] #[B, L, 36]
        R_m = gather_nodes(atom_mask[:,:,5:], E_idx_sub) #[B, L, K, 31]
        X_sidechain = X[:,:,5:,:].view(X.shape[0], X.shape[1], -1)
        R = gather_nodes(X_sidechain, E_idx_sub).view(X.shape[0], X.shape[1], self.side_residue_num, -1, 3) #[B, L, 15, 9, 3]

        Y_t = self.DNA_RNA_types[None,None,None,:].repeat(Y.shape[0], Y.shape[1], Y.shape[2], 1) #[B, L, 10, 26]
        R_t = self.side_chain_atom_types[None,None,None,5:].repeat(X.shape[0], X.shape[1], self.side_residue_num, 1) #[B, L, 25, 46]
        #R - [B, L, 15, 31, 3]
        #R_m - [B, L, 15, 31]
        #R_t - [B, L, 15, 31]

        #Y - [B, L, 10, 26, 3]
        #Y_m  - [B, L, 10, 26]
        #Y_t - [B, L, 10, 26]

        R = R.view(X.shape[0], X.shape[1], -1, 3)
        R_m = R_m.view(X.shape[0], X.shape[1], -1)
        R_t = R_t.view(X.shape[0], X.shape[1], -1)
        
        Y = Y.view(X.shape[0], X.shape[1], -1, 3)
        Y_m = Y_m.view(X.shape[0], X.shape[1], -1)
        Y_t = Y_t.view(X.shape[0], X.shape[1], -1)
        
        Y = torch.cat([Y, Z[:,None,:,:].repeat(1,Y.shape[1], 1, 1)], -2)
        Y_m = torch.cat([Y_m, Z_m[:,None,:].repeat(1, Y.shape[1], 1)], -1)
        Y_t = torch.cat([Y_t, Z_t[:,None,:].repeat(1, Y.shape[1], 1)], -1)
        
        J = torch.cat((R, Y), 2) #[B, L, atoms, 3]
        J_m = torch.cat((R_m, Y_m), 2) #[B, L, atoms]
        J_t = torch.cat((R_t, Y_t), 2) #[B, L, atoms]

        Cb_J_distances = torch.sqrt(torch.sum((Cb[:,:,None,:] - J)**2,-1) + 1e-6) #[B, L, num_atoms]
        mask_J = mask[:,:,None]*J_m
        Cb_J_distances_adjusted = Cb_J_distances*mask_J+(1. - mask_J)*10000.0
        D_J, E_idx_J = torch.topk(Cb_J_distances_adjusted, self.atom_context_num, dim=-1, largest=False) #pick 25 closest atoms
        
        mask_far_atoms = (D_J < 20.0).float() #[B, L, K]

        J_picked_ = torch.gather(J, 2, E_idx_J[:,:,:,None].repeat(1,1,1,3)) #[B, L, 50, 3]
        num_atoms_batch = J_picked_.shape[2]
        J_t_picked_ = torch.gather(J_t, 2, E_idx_J) #[B, L, 50]
        J_m_picked_ = torch.gather(mask_J, 2, E_idx_J) #[B, L, 50]
        J_t_1hot_ = torch.nn.functional.one_hot(J_t_picked_, 105) #N, C, O, P, S #[B, L, 50, 4]

        J_picked = torch.zeros([X.shape[0], X.shape[1], NUM_ATOMS, 3], device=device)
        J_m_picked = torch.zeros([X.shape[0], X.shape[1], NUM_ATOMS], device=device)
        J_t_1hot = torch.zeros([X.shape[0], X.shape[1], NUM_ATOMS, 105], device=device)

        J_picked[:,:,:num_atoms_batch,:] = J_picked_
        J_m_picked[:,:,:num_atoms_batch] = J_m_picked_
        J_t_1hot[:,:,:num_atoms_batch,:] = J_t_1hot_
        
        intermediate = torch.sqrt(torch.sum((J_picked[:,:,:,None,:] - J_picked[:,:,None,:,:])**2,-1) + 1e-5) 
        J_edges = self._rbf(torch.sqrt(torch.sum((J_picked[:,:,:,None,:] - J_picked[:,:,None,:,:])**2,-1) + 1e-6)) #[B, L, 50, 50, num_bins]

        RBF_DNA = []

        D_N_J = self._rbf(torch.sqrt(torch.sum((N[:,:,None,:] - J_picked)**2,-1) + 1e-6)) #[B, L, 50, num_bins]
        D_Ca_J = self._rbf(torch.sqrt(torch.sum((Ca[:,:,None,:] - J_picked)**2,-1) + 1e-6))
        D_C_J = self._rbf(torch.sqrt(torch.sum((C[:,:,None,:] - J_picked)**2,-1) + 1e-6))
        D_O_J = self._rbf(torch.sqrt(torch.sum((O[:,:,None,:] - J_picked)**2,-1) + 1e-6))
        D_Cb_J = self._rbf(torch.sqrt(torch.sum((Cb[:,:,None,:] - J_picked)**2,-1) + 1e-6))

        D_all = torch.cat((D_N_J, D_Ca_J, D_C_J, D_O_J, D_Cb_J, J_t_1hot), dim=-1) #[B,L,25,5*num_bins+5]
        #D_all = torch.cat((D_N_J, D_Ca_J, D_C_J, D_O_J, D_Cb_J), dim=-1) #[B,L,25,5*num_bins+5]
        D_all = D_all*J_m_picked[:,:,:,None]*mask_far_atoms[:,:,:,None]

        V = self.node_project_down(D_all) #[B, L, 50, 32]
        V = self.norm_nodes(V)
         
        J_node_mask = J_m_picked*mask_far_atoms


        J_edges =  self.j_edges(J_edges)
        J_nodes = self.j_nodes(J_t_1hot.float())

        J_edges = self.norm_j_edges(J_edges)
        J_nodes = self.norm_j_nodes(J_nodes)


        return V, E, E_idx, J_node_mask, J_nodes, J_edges


class Struct2Seq(nn.Module):
    def __init__(self, num_letters, node_features, edge_features,
        hidden_dim, num_encoder_layers=3, num_decoder_layers=3, atom_context_num = 25,
        vocab=21, k_neighbors=64, augment_eps=0.05, dropout=0.1, device=None, num_mix=3):
        super(Struct2Seq, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_mix = num_mix
        # Featurization layers
        self.features = ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps, device=device, atom_context_num=atom_context_num)

        # Embedding layers
        #self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_c = nn.Linear(hidden_dim, hidden_dim, bias=True)
        

        self.W_nodes_j = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_edges_j = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.W_torsions = nn.Linear(hidden_dim, 4*3*num_mix, bias=True) #out=W@x+b
        self.V_C = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_C_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.softplus = nn.Softplus(beta=1, threshold=20)
        self.softmax = nn.LogSoftmax(-1)
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            MPNNLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        

        self.context_encoder_layers = nn.ModuleList([
            MPNNLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(2)
        ])
      
        self.j_context_encoder_layers = nn.ModuleList([
            MPNNLayerJ(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(2)
        ])


        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, Y, Y_m, Z, Z_m, Z_t, S, L, mask, X_m, chain_M, residue_idx, dihedral_mask, chain_encoding_all, top_k_sample=False, mask_for_sc=None):
        """ Graph-conditioned sequence model """
        
        V,  E, E_idx, J_m, J_nodes, J_edges = self.features(Z, Z_m, Z_t, X, Y, Y_m, L, mask, X_m, residue_idx, dihedral_mask, chain_encoding_all, top_k_sample, mask_for_sc)
        h_V = self.W_s(S) 
        h_E = self.W_e(E)
        h_E_context = self.W_v(V)
        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)
       
        h_V_C = self.W_c(h_V)
        J_m_edges = J_m[:,:,:,None]*J_m[:,:,None,:]
        J_nodes = self.W_nodes_j(J_nodes)
        J_edges = self.W_edges_j(J_edges)
        for i in range(len(self.context_encoder_layers)):
            J_nodes = torch.utils.checkpoint.checkpoint(self.j_context_encoder_layers[i], J_nodes, J_edges, J_m, J_m_edges)
            h_E_context_cat = torch.cat([h_E_context, J_nodes], -1)
            h_V_C = torch.utils.checkpoint.checkpoint(self.context_encoder_layers[i], h_V_C, h_E_context_cat, mask, J_m)


        h_V_C = self.V_C(h_V_C)
        h_V = h_V + self.V_C_norm(self.dropout(h_V_C))
        # Concatenate sequence embeddings for autoregressive decoder

        for layer in self.decoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_EV, mask)
        torsions = self.W_torsions(h_V)
        torsions = torsions.reshape(h_V.shape[0],h_V.shape[1], 4, self.num_mix, 3)
        mean = torsions[:,:,:,:,0].float()
        concentration = 0.1 + self.softplus(torsions[:,:,:,:,1]).float()
        mix_logits = torsions[:,:,:,:,2].float()
        return mean, concentration, mix_logits


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_std_opt(parameters, d_model, step):
    return NoamOpt(
        d_model, 2, 4000, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step
    )

#------------------------------
#------------------------------
# Initialize model
model = Struct2Seq(atom_context_num=NUM_ATOMS, num_letters=21, node_features=NUM_H, edge_features=NUM_H, hidden_dim=NUM_H, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, k_neighbors=k_neighbors, dropout=dropout, augment_eps=augment_eps, device=device, num_mix=num_mix)
model.to(device)
print('Number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))

checkpoint = torch.load(PATH, map_location=device)
total_step = checkpoint['step'] #write total_step from the checkpoint
epoch = checkpoint['epoch'] #write epoch from the checkpoint
model.load_state_dict(checkpoint['model_state_dict'])

start_train = time.time()
epoch_losses_train, epoch_losses_valid = [], []
epoch_checkpoints = []

model.eval()
with torch.no_grad():
    valid_sum1, valid_sum2, validation_weights = 0., 0., 0.
    validation_angle, validation_angle_mask = np.zeros(4), np.zeros(4)
    for _, batch in enumerate(loader_valid):
        xyz37, X_m, S, S_af2, mask, lengths, chain_M, residue_idx, mask_self, dihedral_mask, chain_encoding_all, Y, Y_m, Z, Z_m, Z_t = featurize(batch, device, True) #shuffle fraction can be specified
        xyz37[:,:,6]= torch.tensor(np.nan)
        xyz37_clone = torch.clone(xyz37)
        masks14_37 = make_atom14_masks({"aatype": S_af2})
        temp_dict = {"aatype": S_af2,
                                "all_atom_positions": xyz37,
                                "all_atom_mask": masks14_37['atom37_atom_exists']}
        torsion_dict = atom37_to_torsion_angles("")(temp_dict)
        
        print(torsion_dict['torsion_angles_sin_cos'][:,:,-2,0])   

        xyz37_m = torch.isfinite(torch.sum(xyz37,-1))
        xyz37[~xyz37_m] = 0.0

        xyz37_m = X_m*masks14_37['atom37_atom_exists']*mask[:,:,None]
        mask_for_loss = mask*chain_M

        print(torsion_dict['torsion_angles_sin_cos'][:,:,-2,0])

      
        Y_updated = torch.cat((Y.view(xyz37.shape[0], xyz37.shape[1], -1, 3), Z[:,None,:,:].repeat([1, xyz37.shape[1], 1, 1])), -2)
        Y_m_updated = torch.cat((Y_m.view(xyz37.shape[0], xyz37.shape[1], -1), Z_m[:,None,:].repeat([1, xyz37.shape[1], 1])), -1)
        DNA_mask = get_DNA_mask(xyz37, mask, Y_updated, Y_m_updated)

        mask_for_sc = (torch.rand(chain_M.shape, device=device) > 1.8).float()
        mask_for_sc_loss = 1. - mask_for_sc
        mask_for_loss = mask*chain_M*mask_for_sc_loss

        mean, concentration, mix_logits = model(xyz37, Y, Y_m, Z, Z_m, Z_t, S, lengths, mask, xyz37_m, chain_M, residue_idx, dihedral_mask, chain_encoding_all, top_k_sample=False, mask_for_sc=mask_for_sc)
        
        mask_for_loss = mask*chain_M


        loss_sc, angle_min, torsions_mask, Lmin, true_torsion_angles, alt_torsion_angles, torsions_mask_loss = sidechain_torsion_loss(torsion_dict, mean, concentration, mix_logits, mask_for_loss)

        print('Loss', loss_sc)

        mix = D.Categorical(logits=mix_logits)
        comp = D.VonMises(mean, concentration)
        pred_dist = D.MixtureSameFamily(mix, comp)

        predicted_samples = pred_dist.sample([100])
        print(predicted_samples.shape)

        log_probs_of_samples = pred_dist.log_prob(predicted_samples)

        predicted_real_values = torch.gather(predicted_samples, dim=0, index=torch.argmax(log_probs_of_samples,0)[None,])[0,]



        torsion_pred_unit = torch.cat([torch.sin(predicted_real_values[:,:,:,None]), torch.cos(predicted_real_values[:,:,:,None])], -1)

        if True:
            torsions_mask_ = temp_dict["torsion_angles_mask"][:,:,3:]
            log_probs_raw = pred_dist.log_prob(predicted_real_values)
            log_probs = torch.sum(log_probs_raw*torsions_mask_,-1)/(torch.sum(torsions_mask_,-1)+1e-5)
            torsions_mask_sum = (temp_dict["torsion_angles_mask"][:,:,3:].sum(-1)==0)
            log_probs = log_probs + torsions_mask_sum*2.0
            b_factor_pred = log_probs.detach().cpu().numpy()[:,:,None]
            b_factor_pred = np.repeat(b_factor_pred, 37, -1)


    
            rigids = Rigid.make_transform_from_reference(
                             n_xyz=xyz37[:,:,0,:],
                             ca_xyz=xyz37[:,:,1,:],
                             c_xyz=xyz37[:,:,2,:],
                             eps=1e-9)
    
            true_frames = feats.torsion_angles_to_frames(rigids,
                                                            torsion_dict['torsion_angles_sin_cos'],
                                                            S_af2,
                                                            torch.tensor(restype_rigid_group_default_frame, device=device))
                    
            torsions_update = torch.clone(torsion_dict['torsion_angles_sin_cos'])
            torsions_update[:,:,3:] = torsion_pred_unit
            pred_frames = feats.torsion_angles_to_frames(rigids,
                                                            torsions_update,
                                                            S_af2,
                                                            torch.tensor(restype_rigid_group_default_frame, device=device))
    
    
            atom14_true = feats.frames_and_literature_positions_to_atom14_pos(true_frames, 
                                                                                 S_af2,
                                                                                 torch.tensor(restype_rigid_group_default_frame, device=device),
                                                                                 torch.tensor(restype_atom14_to_rigid_group, device=device),
                                                                                 torch.tensor(restype_atom14_mask, device=device),
                                                                                 torch.tensor(restype_atom14_rigid_group_positions, device=device))
            atom14_pred = feats.frames_and_literature_positions_to_atom14_pos(pred_frames,
                                                                                 S_af2,
                                                                                 torch.tensor(restype_rigid_group_default_frame, device=device),
                                                                                 torch.tensor(restype_atom14_to_rigid_group, device=device),
                                                                                 torch.tensor(restype_atom14_mask, device=device),
                                                                                 torch.tensor(restype_atom14_rigid_group_positions, device=device))
    
            xyz37_true = feats.atom14_to_atom37(atom14_true, masks14_37)
            xyz37_pred = feats.atom14_to_atom37(atom14_pred, masks14_37)
                    
            B_size = xyz37_true.shape[0]
            for k in range(B_size):
                name_ = batch[k]["name"]
                idx = np.argwhere(mask[k].detach().cpu().numpy()==1)

                if True:
                    xi_mask = temp_dict["torsion_angles_mask"][:,:,3:].detach().cpu().numpy()
                    np.savez(f'{output_path}/{name_}', input_log_prob_density=-Lmin[k][idx][:,0],
                                                       input_angles=true_torsion_angles[k][idx][:,0].detach().cpu().numpy(),
                                                       input_alt_angles=alt_torsion_angles[k][idx][:,0].detach().cpu().numpy(),
                                                       repacked_log_prob_density=log_probs_raw[k][idx][:,0].detach().cpu().numpy(),
                                                       repacked_angles=predicted_real_values[k][idx][:,0].detach().cpu().numpy(),
                                                       repacked_angles_all=predicted_samples[:,k][:,idx][:,:,0].detach().cpu().numpy(), 
                                                       mask=xi_mask[k][idx][:,0],
                                                       torsions_sin_cos=torsion_dict['torsion_angles_sin_cos'][k][idx][:,0],
                                                       mask_loss=torsions_mask_loss[k][idx][:,0].detach().cpu().numpy(),
                                                       ca_mask=mask[k][idx][:,0].detach().cpu().numpy(),
                                                       sequence=S_af2[k][idx][:,0].detach().cpu().numpy(),
                                                       chain_encoding=chain_encoding_all[k][idx][:,0].detach().cpu().numpy(),
                                                       index=idx,
                                                       xyz37_clone=xyz37_clone[k][idx][:,0].detach().cpu().numpy(),
                                                       xyz37_mask_t=masks14_37['atom37_atom_exists'][k][idx][:,0].detach().cpu().numpy(),
                                                       model_mean=mean[k][idx][:,0].detach().cpu().numpy(),
                                                       model_concentration=concentration[k][idx][:,0].detach().cpu().numpy(),
                                                       model_mix=mix_logits[k][idx][:,0].detach().cpu().numpy())




                protein_true = Protein(
                                    atom_positions=np.array(xyz37_true[k].cpu().data.numpy()),
                                    atom_mask=np.array(xyz37_m[k].cpu().data.numpy()),  #all_res_mask or X_sc_37 instead??
                                    aatype=np.array(S_af2[k].cpu().data.numpy()),
                                    residue_index=np.arange(S_af2.shape[1]),
                                    chain_index=chain_encoding_all.detach().cpu().numpy()[k,],
                                    b_factors=np.zeros([S_af2.shape[1],37])
                                )      
                        
                protein_pred = Protein(
                                    atom_positions=np.array(xyz37_pred[k].cpu().data.numpy()),
                                    atom_mask=np.array(xyz37_m[k].cpu().data.numpy()),  #all_res_mask or X_sc_37 instead??
                                    aatype=np.array(S_af2[k].cpu().data.numpy()),
                                    residue_index=np.arange(S_af2.shape[1]),
                                    chain_index=chain_encoding_all.detach().cpu().numpy()[k,],
                                    b_factors=b_factor_pred[k]
                                )   
        
                with open(f'{output_path}/{name_}_lig_true.pdb', 'w') as fp:
                    fp.write(to_pdb(protein_true))
        
                with open(f'{output_path}/{name_}_lig_pred.pdb', 'w') as fp:
                    fp.write(to_pdb(protein_pred))
    
    
