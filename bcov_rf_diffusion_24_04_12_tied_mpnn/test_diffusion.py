#!/home/dimaio/.conda/envs/SE3nv/bin/python
import diffusion
import torch 
import numpy as np 

from util import writepdb_multi
from util_module import ComputeAllAtomCoords

from icecream import ic 

def parse_pdb(filename, **kwargs):
    '''extract xyz coords for all heavy atoms'''
    lines = open(filename,'r').readlines()
    return parse_pdb_lines(lines, **kwargs)

def parse_pdb_lines(lines, parse_hetatom=False, ignore_het_h=True):
    # indices of residues observed in the structure
    res = [(l[22:26],l[17:20]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
    seq = [util.aa2num[r[1]] if r[1] in util.aa2num.keys() else 20 for r in res]
    pdb_idx = [( l[21:22].strip(), int(l[22:26].strip()) ) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]  # chain letter, res num

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(res), 14, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        chain, resNo, atom, aa = l[21:22], int(l[22:26]), ' '+l[12:16].strip().ljust(3), l[17:20]
        idx = pdb_idx.index((chain,resNo))
        for i_atm, tgtatm in enumerate(util.aa2long[util.aa2num[aa]]):
            if tgtatm is not None and tgtatm.strip() == atom.strip(): # ignore whitespace
                xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[...,0]))
    xyz[np.isnan(xyz[...,0])] = 0.0

    # remove duplicated (chain, resi)
    new_idx = []
    i_unique = []
    for i,idx in enumerate(pdb_idx):
        if idx not in new_idx:
            new_idx.append(idx)
            i_unique.append(i)

    pdb_idx = new_idx
    xyz = xyz[i_unique]
    mask = mask[i_unique]
    seq = np.array(seq)[i_unique]

    out = {'xyz':xyz, # cartesian coordinates, [Lx14]
            'mask':mask, # mask showing which atoms are present in the PDB file, [Lx14]
            'idx':np.array([i[1] for i in pdb_idx]), # residue numbers in the PDB file, [L]
            'seq':np.array(seq), # amino acid sequence, [L]
            'pdb_idx': pdb_idx,  # list of (chain letter, residue number) in the pdb file, [L]
           }

    # heteroatoms (ligands, etc)
    if parse_hetatom:
        xyz_het, info_het = [], []
        for l in lines:
            if l[:6]=='HETATM' and not (ignore_het_h and l[77]=='H'):
                info_het.append(dict(
                    idx=int(l[7:11]),
                    atom_id=l[12:16],
                    atom_type=l[77],
                    name=l[16:20]
                ))
                xyz_het.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])

        out['xyz_het'] = np.array(xyz_het)
        out['info_het'] = info_het

    return out


parsed = parse_pdb('/mnt/home/davidcj/projects/expert-potato/expert-potato/1qys.pdb')
xyz    = parsed['xyz']
xyz    = torch.from_numpy( (xyz - xyz[:,:1,:].mean(axis=0)[None,...]) )

seq = torch.from_numpy( parsed['seq'] )
atom_mask = torch.from_numpy( parsed['mask'] )

diffusion_mask = torch.zeros(len(seq.squeeze())).to(dtype=bool)
diffusion_mask[:20] = True

T = 200
b_0 = 0.001
b_T = 0.1

kwargs = {'T'  : T,
          'b_0': b_0,
          'b_T': b_T,
          'schedule_type':'cosine',
          'schedule_kwargs':{},
          'so3_type':'slerp',
          'chi_type':'interp',
          'var_scale':1.,
          'crd_scale':1/15,
          'aa_decode_steps':100}


diffuser = diffusion.Diffuser(**kwargs)

diffused_T,\
deltas,\
diffused_frame_crds,\
diffused_frames,\
diffused_torsions,\
diffused_FA_crds,\
aa_masks = diffuser.diffuse_pose(xyz, seq, atom_mask, diffusion_mask=diffusion_mask, diffuse_sidechains=True)


# print('Writing translation pdb')
# outpath1 = './translation_only.pdb'
# seq = torch.from_numpy(seq)
# writepdb_multi(outpath1, diffused_T.transpose(0,1), torch.ones_like(seq), seq, backbone_only=True)



# print('Writing slerp pdb')
# outpath1 = './slerp_only.pdb'

# writepdb_multi(outpath1, torch.from_numpy(diffused_frame_crds).transpose(0,1), torch.ones_like(seq), seq, backbone_only=True)




# print('Writing combo slerp / translation pdb')
# cum_delta = deltas.cumsum(dim=1)
# ic(torch.is_tensor(diffused_frame_crds))
# ic(torch.is_tensor(cum_delta))
# translated_slerp = torch.from_numpy(diffused_frame_crds) + cum_delta[:,:,None,:]

# ic(cum_delta[0,10])
# ic(diffused_T[0,10])

# outpath1 = './slerp_and_translate.pdb'
# writepdb_multi(outpath1, translated_slerp.transpose(0,1), torch.ones_like(seq), seq, backbone_only=True)



## Create full atom crds from chi-only diffusion 
# diffused_torsions_sincos = torch.stack( [torch.cos(diffused_torsions), torch.sin(diffused_torsions)], dim=-1 )
# get_allatom = ComputeAllAtomCoords()
# fullatom_stack = []
# for alphas in diffused_torsions_sincos.transpose(0,1):

#     _,full_atoms = get_allatom(seq[None], xyz[None, :,:3], alphas[None])

#     fullatom_stack.append(full_atoms.squeeze())


# print('Writing chi angle interpolation only')
# outpath1 = './chi_interp.pdb'
# writepdb_multi(outpath1, fullatom_stack, torch.ones_like(seq), seq, backbone_only=False)


# Create full atom coords from combined diffusion
print('Writing combined diffusion pdb...')
outpath1 = './diffuse_all.pdb'
# ic(diffused_FA_crds.shape)
writepdb_multi(outpath1, diffused_FA_crds.squeeze(), torch.ones_like(seq), seq, backbone_only=False)
