import argparse
import os.path

def main(args):
    import json, time, os, sys, glob
    import shutil
    import numpy as np
    import torch
    from torch import optim
    import random
    from dateutil import parser 
    import csv
    import copy
    import math
    from typing import Dict
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.utils
    import torch.utils.checkpoint
    import queue
    import torch.distributions as D

    sys.path.append("/home/justas/openfold")

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
            restype_atom37_to_rigid_group
	)
    from openfold.np.protein import Protein, to_pdb, from_pdb_string

    from sc_utils import featurize, parse_PDB
    from sc_utils import StructureDataset, StructureDatasetPDB, Packer



    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model = Packer(num_mix=3, 
                   node_features=128, 
                   edge_features=128, 
                   hidden_dim=128, 
                   num_encoder_layers=3, 
                   num_decoder_layers=3, 
                   k_neighbors=32, 
                   augment_eps=0.00)

    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()


    base_folder = args.output_folder_path
    if base_folder[-1] != '/':
        base_folder = base_folder + '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)


    if args.pdb_path:
        pdb_dict_list = parse_PDB(args.pdb_path)
        dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=args.max_length)
    else:
        dataset_valid = StructureDataset(args.jsonl_path, truncate=None, max_length=args.max_length)
    

    sss = args.sample_size
    with torch.no_grad():
        for ix, protein in enumerate(dataset_valid):
            batch_clones = [copy.deepcopy(protein) for i in range(args.batch_size)]
            xyz37, S, S_af2, mask, mask_sc, lengths, chain_M, residue_idx, mask_self, dihedral_mask, chain_encoding_all = featurize(batch_clones, device)

            masks14_37 = make_atom14_masks({"aatype": S_af2})
            temp_dict = {"aatype": S_af2,
                         "all_atom_positions": xyz37,
                         "all_atom_mask": masks14_37['atom37_atom_exists']}
            torsion_dict = atom37_to_torsion_angles("")(temp_dict)

            true_torsions = torsion_dict['torsion_angles_sin_cos'][:,:,3:]
            alt_torsions = torsion_dict['alt_torsion_angles_sin_cos'][:,:,3:]
            torsion_angle_mask = torsion_dict['torsion_angles_mask'][:,:,3:]

            true_torsions_mask = ~torch.isnan(true_torsions)
            true_torsions[~true_torsions_mask] = 0.0

            alt_torsions_mask = ~torch.isnan(alt_torsions)
            alt_torsions[~alt_torsions_mask] = 0.0

            xyz37_m = torch.isfinite(torch.sum(xyz37,-1))
            xyz37[~xyz37_m] = 0.0
            xyz37_m = xyz37_m*masks14_37['atom37_atom_exists']*mask[:,:,None]
            
            full_torsion_mask = torsion_dict["torsion_angles_mask"][:,:,3:]*true_torsions_mask[...,0]*true_torsions_mask[...,1]
            mean, concentration, mix_logits, predicted_real_values, xi_log_probs_out = model(true_torsions, full_torsion_mask, xyz37, xyz37_m, S, lengths, mask, chain_M, residue_idx, dihedral_mask, chain_encoding_all, use_true_context=args.score_only)

            torsions_mask_ = temp_dict["torsion_angles_mask"][:,:,3:]
            xi_log_probs_mean = torch.sum(xi_log_probs_out*torsions_mask_,-1)/(torch.sum(torsions_mask_,-1)+1e-6) #[B, L]
            
            xi_value, xi_idx = torch.sort(xi_log_probs_mean, dim=0, descending=True) 
            
            print(mean.shape, concentration.shape, mix_logits.shape, predicted_real_values.shape)
            mean = torch.gather(mean, dim=0, index=xi_idx[:sss,][...,None,None].repeat(1,1,4,3))
            concentration = torch.gather(concentration, dim=0, index=xi_idx[:sss,][...,None,None].repeat(1,1,4,3))
            mix_logits = torch.gather(mix_logits, dim=0, index=xi_idx[:sss,][...,None,None].repeat(1,1,4,3))
            predicted_real_values = torch.gather(predicted_real_values, dim=0, index=xi_idx[:sss,][...,None].repeat(1,1,4))

            print(mean.shape, concentration.shape, mix_logits.shape, predicted_real_values.shape)
            mix = D.Categorical(logits=mix_logits)
            comp = D.VonMises(mean, concentration)
            pred_dist = D.MixtureSameFamily(mix, comp)

            torsion_pred_unit = torch.cat([torch.sin(predicted_real_values[:,:,:,None]), torch.cos(predicted_real_values[:,:,:,None])], -1)

            true_torsion_angles = torch.atan2(true_torsions[:sss,:,:,-2], true_torsions[:sss,:,:,-1])
            alt_torsion_angles = torch.atan2(alt_torsions[:sss,:,:,-2], alt_torsions[:sss,:,:,-1])

            L1_t = pred_dist.log_prob(true_torsion_angles)
            L2_t = pred_dist.log_prob(alt_torsion_angles)
            Lmax_t = torch.maximum(L1_t,L2_t)
            Lmax_p = pred_dist.log_prob(predicted_real_values) #[sss,L,4]
            
            tmp_types = torch.tensor(restype_atom37_to_rigid_group, device=device)[S_af2[:sss,]]
            mask_for_37_apply = (torch.clone(tmp_types)==0) #[B,L,37]
            tmp_types[tmp_types<4]=4
            tmp_types -= 4
            atom_types_for_b_factor = torch.nn.functional.one_hot(tmp_types,4) #[B, L, 37, 4] 
                                                        


            torsions_mask_sum = (torsion_dict["torsion_angles_mask"][:sss,:,3:].sum(-1)==0).float()

            uncertainty = Lmax_p[:,:,None,:]*atom_types_for_b_factor #[B,L,37,4]
            b_factor_pred = uncertainty.sum(-1) #[B, L, 37]
            b_factor_pred = (1-torsions_mask_sum[:,:,None])*b_factor_pred + torsions_mask_sum[:,:,None]*1.0
            b_factor_pred = np.clip(b_factor_pred, -1.0, 2.0)


            uncertainty = Lmax_t[:,:,None,:]*atom_types_for_b_factor #[B,L,37,4]
            b_factor_true = uncertainty.sum(-1) #[B, L, 37]
            b_factor_true = (1-torsions_mask_sum[:,:,None])*b_factor_true + torsions_mask_sum[:,:,None]*1.0
            b_factor_true = np.clip(b_factor_true, -1.0, 2.0)
 
            rigids = Rigid.make_transform_from_reference(
                             n_xyz=xyz37[:sss,:,0,:],
                             ca_xyz=xyz37[:sss,:,1,:],
                             c_xyz=xyz37[:sss,:,2,:],
                             eps=1e-9)
    
            true_frames = feats.torsion_angles_to_frames(rigids,
                                                            torsion_dict['torsion_angles_sin_cos'][:sss,],
                                                            S_af2[:sss,],
                                                            torch.tensor(restype_rigid_group_default_frame, device=device))
                    
            torsions_update = torch.clone(torsion_dict['torsion_angles_sin_cos'])[:sss,]
            torsions_update[:,:,3:] = torsion_pred_unit
            pred_frames = feats.torsion_angles_to_frames(rigids,
                                                            torsions_update,
                                                            S_af2[:sss,],
                                                            torch.tensor(restype_rigid_group_default_frame, device=device))
    
    
            atom14_true = feats.frames_and_literature_positions_to_atom14_pos(true_frames, 
                                                                                 S_af2[:sss,],
                                                                                 torch.tensor(restype_rigid_group_default_frame, device=device),
                                                                                 torch.tensor(restype_atom14_to_rigid_group, device=device),
                                                                                 torch.tensor(restype_atom14_mask, device=device),
                                                                                 torch.tensor(restype_atom14_rigid_group_positions, device=device))
            atom14_pred = feats.frames_and_literature_positions_to_atom14_pos(pred_frames,
                                                                                 S_af2[:sss],
                                                                                 torch.tensor(restype_rigid_group_default_frame, device=device),
                                                                                 torch.tensor(restype_atom14_to_rigid_group, device=device),
                                                                                 torch.tensor(restype_atom14_mask, device=device),
                                                                                 torch.tensor(restype_atom14_rigid_group_positions, device=device))
            masks14_37_sss = make_atom14_masks({"aatype": S_af2[:sss,]}) 
            xyz37_true = feats.atom14_to_atom37(atom14_true, masks14_37_sss)
            xyz37_pred = feats.atom14_to_atom37(atom14_pred, masks14_37_sss)
            print(b_factor_pred.shape) 
            for k in range(sss):
                protein_true = Protein(
                                    atom_positions=np.array(xyz37_true[k].cpu().data.numpy()),
                                    atom_mask=np.array(xyz37_m[k].cpu().data.numpy()),  #all_res_mask or X_sc_37 instead??
                                    aatype=np.array(S_af2[k].cpu().data.numpy()),
                                    residue_index=np.arange(S_af2.shape[1]),
                                    chain_index=chain_encoding_all.detach().cpu().numpy()[k,],
                                    b_factors=b_factor_true[k]
                                )      
                        
                protein_pred = Protein(
                                    atom_positions=np.array(xyz37_pred[k].cpu().data.numpy()),
                                    atom_mask=np.array(xyz37_m[k].cpu().data.numpy()),  #all_res_mask or X_sc_37 instead??
                                    aatype=np.array(S_af2[k].cpu().data.numpy()),
                                    residue_index=np.arange(S_af2.shape[1]),
                                    chain_index=chain_encoding_all.detach().cpu().numpy()[k,],
                                    b_factors=b_factor_pred[k]
                                )   
        
                name_ = batch_clones[k]["name"]
                
                idx = np.argwhere(mask[k].detach().cpu().numpy()==1)
                                
                if args.output_npz:
                    xi_mask = temp_dict["torsion_angles_mask"][:,:,3:].detach().cpu().numpy()
                    np.savez(f'{base_folder}/{name_}_{k}', 
                                                       mask=xi_mask[k][idx][:,0], 
                                                       ca_mask=mask[k][idx][:,0].detach().cpu().numpy(),
                                                       sequence=S_af2[k][idx][:,0].detach().cpu().numpy(),
                                                       input_angles=true_torsion_angles[k][idx][:,0].detach().cpu().numpy(),
                                                       input_alt_angles=alt_torsion_angles[k][idx][:,0].detach().cpu().numpy(),
                                                       repacked_angles=predicted_real_values[k][idx][:,0].detach().cpu().numpy(),
                                                       chain_encoding=chain_encoding_all[k][idx][:,0].detach().cpu().numpy(),
                                                       index=idx,
                                                       model_mean=mean[k][idx][:,0].detach().cpu().numpy(),
                                                       model_concentration=concentration[k][idx][:,0].detach().cpu().numpy(),
                                                       model_mix=mix_logits[k][idx][:,0].detach().cpu().numpy())

               
                if not args.score_only:
                    with open(f'{base_folder}/{name_}_{k}.pdb', 'w') as fp:
                        fp.write(to_pdb(protein_pred))
            if args.score_only:
                with open(f'{base_folder}/{name_}_scored.pdb', 'w') as fp:
                    fp.write(to_pdb(protein_true))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    argparser.add_argument("--checkpoint_path", type=str, default="/projects/ml/struc2seq/data_for_complexes/training_scripts/2022/model_outputs/vm_3_002_xi1234_ind_LN_forcing/checkpoints/epoch500_step319788.pt", help="Path to model weights folder;") 
    argparser.add_argument("--batch_size", type=int, default=1, help="Number of examples")
    argparser.add_argument("--sample_size", type=int, default=1, help="Number of beam examples")
    argparser.add_argument("--output_npz", type=int, default=1, help="Output npz file")
    argparser.add_argument("--max_length", type=int, default=1000000, help="Max length")
    argparser.add_argument("--output_folder_path", type=str, default="./", help="Output folder path")
    argparser.add_argument("--pdb_path", type=str, default="", help="PDB path for the input")
    argparser.add_argument("--jsonl_path", type=str, default="", help="PDB path for the input")
    argparser.add_argument("--score_only", type=int, default=1, help="Scores only the input")
    
    args = argparser.parse_args()    
    main(args) 

