import argparse
import os.path

def main(args):
   
    import json, time, os, sys, glob
    import shutil
    import warnings
    import numpy as np
    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    from torch.utils.data.dataset import random_split, Subset
    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import subprocess
    import os.path
    from protein_mpnn_utils_HACK import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB, parse_fasta
    from protein_mpnn_utils_HACK import StructureDataset, StructureDatasetPDB, ProteinMPNN
    from sc_utils import pack_side_chains, build_sc_model, score_side_chains 
    
    if args.use_seed:
        seed=args.seed
    else:
        seed=int(np.random.randint(0, high=999, size=1, dtype=int)[0])
        

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
 
    hidden_dim = 128
    num_layers = 3 
  

    if args.path_to_model_weights:
        model_folder_path = args.path_to_model_weights
        if model_folder_path[-1] != '/':
            model_folder_path = model_folder_path + '/'
    else: 
        file_path = os.path.realpath(__file__)
        k = file_path.rfind("/")
        model_folder_path = file_path[:k] + '/vanilla_model_weights/'

    checkpoint_path = model_folder_path + f'{args.model_name}.pt'


    if args.transmembrane:
        checkpoint_path = "/databases/mpnn/tmd_weights/v_48_020.pt"
        if args.transmembrane == "yes":
            species_label = 1
        elif args.transmembrane == "no":
            species_label = 0
        else:
            print("WARNING: select correct transmembrane flag!!!")
    else:
        species_label = -1


    if args.species:
        checkpoint_path = "/databases/mpnn/species_class_weights/v_48_020.pt"
        if args.species == "homo_sapiens":
            species_label = 0
        elif args.species == "bacterial":
            species_label = 1
        elif args.species == "other":
            species_label = 2
        else:
            print("WARNING: select correct species!!!")
    elif args.transmembrane:
        pass
    else:
        species_label = -1


    #per residue label
    tmd_dict = None
    tmd_buried_list = None
    tmd_interface_list = None
    if args.transmembrane_chain_ids:
        tmd_chain_list = args.transmembrane_chain_ids.split(",")
        checkpoint_path = "/databases/mpnn/tmd_per_residue_weights/tmd_v_48_020.pt"
        if args.transmembrane_buried:
            tmd_buried_list = [[int(item) for item in one.split()] for one in args.transmembrane_buried.split(",")]
        else:
            tmd_buried_list = len(tmd_chain_list)*[[]] 
        if args.transmembrane_interface:
            tmd_interface_list = [[int(item) for item in one.split()] for one in args.transmembrane_interface.split(",")]
        else:
            tmd_interface_list = len(tmd_chain_list)*[[]]
        tmd_dict = {}
        for i, letter in enumerate(tmd_chain_list):
            tmd_dict[letter] = (tmd_buried_list[i], tmd_interface_list[i])


    folder_for_outputs = args.out_folder
    
    NUM_BATCHES = args.num_seq_per_target//args.batch_size
    BATCH_COPIES = args.batch_size
    temperatures = [float(item) for item in args.sampling_temp.split()]
    omit_AAs_list = args.omit_AAs
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    alphabet_dict = dict(zip(alphabet, range(21))) 
    omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    
    if isinstance(args.chain_id_jsonl, dict):
        chain_id_dict = args.chain_id_jsonl
    elif os.path.isfile(args.chain_id_jsonl):
        with open(args.chain_id_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            chain_id_dict = json.loads(json_str)
    else:
        chain_id_dict = None
        # print(40*'-')
        # print('chain_id_jsonl is NOT loaded')

    if isinstance(args.fixed_positions_jsonl, dict):
        fixed_positions_dict = args.fixed_positions_jsonl        
    elif os.path.isfile(args.fixed_positions_jsonl):
        with open(args.fixed_positions_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            fixed_positions_dict = json.loads(json_str)
    else:
        # print(40*'-')
        # print('fixed_positions_jsonl is NOT loaded')
        fixed_positions_dict = None
    
    if isinstance(args.pssm_jsonl, dict):
        pssm_dict = args.pssm_jsonl 
    elif os.path.isfile(args.pssm_jsonl):
        with open(args.pssm_jsonl, 'r') as json_file:
            json_list = list(json_file)
        pssm_dict = {}
        for json_str in json_list:
            pssm_dict.update(json.loads(json_str))
    else:
        # print(40*'-')
        # print('pssm_jsonl is NOT loaded')
        pssm_dict = None
    
    if isinstance(args.omit_AA_jsonl, dict):
        omit_AA_dict = args.omit_AA_jsonl 
    elif os.path.isfile(args.omit_AA_jsonl):
        with open(args.omit_AA_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            omit_AA_dict = json.loads(json_str)
    else:
        # print(40*'-')
        # print('omit_AA_jsonl is NOT loaded')
        omit_AA_dict = None
    
    
    if isinstance(args.bias_AA_jsonl, dict):
        bias_AA_dict = args.bias_AA_jsonl 
    elif os.path.isfile(args.bias_AA_jsonl):
        with open(args.bias_AA_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            bias_AA_dict = json.loads(json_str)
    else:
        # print(40*'-')
        # print('bias_AA_jsonl is NOT loaded')
        bias_AA_dict = None
    
    if isinstance(args.tied_positions_jsonl, dict):
        tied_positions_dict = args.tied_positions_jsonl 
    elif os.path.isfile(args.tied_positions_jsonl):
        with open(args.tied_positions_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            tied_positions_dict = json.loads(json_str)
    else:
        # print(40*'-')
        # print('tied_positions_jsonl is NOT loaded')
        tied_positions_dict = None

    if isinstance(args.bias_by_res_jsonl, dict):
        bias_by_res_dict = args.bias_by_res_jsonl
    elif os.path.isfile(args.bias_by_res_jsonl):
        with open(args.bias_by_res_jsonl, 'r') as json_file:
            json_list = list(json_file)
    
        for json_str in json_list:
            bias_by_res_dict = json.loads(json_str)
        # print('bias by residue dictionary is loaded')
    else:
        # print(40*'-')
        # print('bias by residue dictionary is not loaded, or not provided')
        bias_by_res_dict = None
   

 
    # print(40*'-')
    bias_AAs_np = np.zeros(len(alphabet))
    if bias_AA_dict:
            for n, AA in enumerate(alphabet):
                    if AA in list(bias_AA_dict.keys()):
                            bias_AAs_np[n] = bias_AA_dict[AA]
    
    if args.pdb_path:
        pdb_dict_list = parse_PDB(args.pdb_path, args.score_sc_only)
        dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=args.max_length)
        all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9]=='seq_chain'] #['A','B', 'C',...]
    else:
        dataset_valid = StructureDataset(args.jsonl_path, truncate=None, max_length=args.max_length)

 
    if args.pdb_bias_path:
        mpnn_alphabet_dict = {'A': 0,'C': 1,'D': 2,'E': 3,'F': 4,'G': 5,'H': 6,'I': 7,'K': 8,'L': 9,'M': 10,'N': 11,'P': 12,'Q': 13,'R': 14,'S': 15,'T': 16,'V': 17,'W': 18,'Y': 19,'X': 20}
        pdb_bias_dict_list = parse_PDB(args.pdb_bias_path)
        bias_by_res_dict = {}
        bias_by_res_dict[pdb_dict_list[0]['name']] = {}
        for chain in all_chain_list:
            bias_by_res_dict[pdb_dict_list[0]['name']][chain] = args.pdb_bias_level*np.eye(21)[np.array([mpnn_alphabet_dict[item] for item in list(pdb_bias_dict_list[0][f"seq_chain_{chain}"])])]

        
    # print(40*'-')
    if not args.pack_only:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        checkpoint_list = list(checkpoint)
        if "num_edges" not in checkpoint_list:
            checkpoint['num_edges'] = 48
        if "noise_level" not in checkpoint_list:
            checkpoint['noise_level'] = 0.2 
        # print('Number of edges:', checkpoint['num_edges'])
        noise_level_print = checkpoint['noise_level']
        # print(f'Training noise level: {noise_level_print}A')
        model = ProteinMPNN(num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=args.backbone_noise, k_neighbors=checkpoint['num_edges'], use_label=bool(args.species) or bool(args.transmembrane), label=species_label, load_tmd_model=bool(args.transmembrane_chain_ids))
        model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    
    # Build paths for experiment
    base_folder = folder_for_outputs
    if base_folder[-1] != '/':
        base_folder = base_folder + '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
 
    # Timing
    start_time = time.time()
    total_residues = 0
    protein_list = []
    total_step = 0
    if args.pack_side_chains or args.pack_only or args.score_sc_only:
        sc_model = build_sc_model()
    with torch.no_grad():
        test_sum, test_weights = 0., 0.
        #print('Generating sequences...')
        for ix, protein in enumerate(dataset_valid):
            batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta, tmd_labels = tied_featurize(batch_clones, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict, parse_all_atoms=args.score_sc_only, tmd_dict=tmd_dict)
            pssm_log_odds_mask = (pssm_log_odds_all > args.pssm_threshold).float() #1.0 for true, 0.0 for false
            
            randn_1 = torch.randn(chain_M.shape, device=X.device)
            log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1, tmd_labels=tmd_labels)
            mask_for_loss = mask*chain_M*chain_M_pos
            scores = _scores(S, log_probs, mask_for_loss)
            native_score = scores.cpu().data.numpy()
            # Generate some sequences
            
            if args.conditional_probs_only:
                log_conditional_probs_list = []
                for j in range(NUM_BATCHES):
                    randn_1 = torch.randn(chain_M.shape, device=X.device)
                    if args.conditional_probs_use_pseudo:
                        log_conditional_probs = model.conditional_probs_pseudo(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1, args.conditional_probs_only_backbone, tmd_labels=tmd_labels)
                    else:
                        log_conditional_probs = model.conditional_probs(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1, args.conditional_probs_only_backbone, tmd_labels=tmd_labels)
                    log_conditional_probs_list.append(log_conditional_probs.cpu().numpy())
                concat_log_p = np.concatenate(log_conditional_probs_list, 0) #[B, L, 21]
                probs = np.exp(concat_log_p)
                probs = probs[:,:,0:20]
                probs_sum = probs.sum(axis=2, keepdims=True)
                normalized_probs = probs / probs_sum
                    
            return normalized_probs
   
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    argparser.add_argument("--path_to_model_weights", type=str, default="/databases/mpnn/vanilla_model_weights/", help="Path to model weights folder;") 
    argparser.add_argument("--model_name", type=str, default="v_48_020", help="ProteinMPNN model name: v_48_002, v_48_010, v_48_020, v_48_030, v_32_002, v_32_010; v_32_020, v_32_030")

    argparser.add_argument("--use_seed", type=int, default=0, help="0 for False, 1 for True; To set global seed.")
    argparser.add_argument("--seed", type=int, default=0, help="Set seed for torch, numpy, and python random.")
 
    argparser.add_argument("--save_score", type=int, default=0, help="0 for False, 1 for True; save score=-log_prob to npy files")
    argparser.add_argument("--save_probs", type=int, default=0, help="0 for False, 1 for True; save MPNN predicted probabilites per position")
    argparser.add_argument("--assume_symmetry", type=int, default=0, help="0 for False, 1 for True; Skips decoding over tied residues")
    argparser.add_argument("--compute_input_sequence_score", type=int, default=1, help="0 for False, 1 for True")
   
    argparser.add_argument("--score_only", type=int, default=0, help="0 for False, 1 for True; score input backbone-sequence pairs")
    argparser.add_argument("--path_to_fasta", type=str, default="", help="path to fasta file with sequences to be scored")

    argparser.add_argument("--conditional_probs_only", type=int, default=0, help="0 for False, 1 for True; output conditional probabilities p(s_i given the rest of the sequence and backbone)")    
    argparser.add_argument("--conditional_probs_only_backbone", type=int, default=0, help="0 for False, 1 for True; if true output conditional probabilities p(s_i given backbone)") 
    argparser.add_argument("--conditional_probs_use_pseudo", type=int, default=0, help="0 for False, 1 for True; output conditional probabilities using ones-eye mask p(s_i given the rest of the sequence and backbone)")

    argparser.add_argument("--unconditional_probs_only", type=int, default=0, help="0 for False, 1 for True; output unconditional probabilities p(s_i given backbone)")
 
    argparser.add_argument("--backbone_noise", type=float, default=0.00, help="Standard deviation of Gaussian noise to add to backbone atoms")
    argparser.add_argument("--num_seq_per_target", type=int, default=1, help="Number of sequences to generate per target")
    argparser.add_argument("--batch_size", type=int, default=1, help="Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory")
    argparser.add_argument("--max_length", type=int, default=20000, help="Max sequence length")
    argparser.add_argument("--sampling_temp", type=str, default="0.1", help="A string of temperatures, 0.2 0.25 0.5. Sampling temperature for amino acids, T=0.0 means taking argmax, T>>1.0 means sample randomly. Suggested values 0.1, 0.15, 0.2, 0.25, 0.3. Higher values will lead to more diversity.")
    
    argparser.add_argument("--out_folder", type=str, help="Path to a folder to output sequences, e.g. /home/out/")
    argparser.add_argument("--pdb_path", type=str, default='', help="Path to a single PDB to be designed")
    argparser.add_argument("--pdb_path_chains", type=str, default='', help="Define which chains need to be designed for a single PDB ")
    argparser.add_argument("--jsonl_path", type=str, help="Path to a folder with parsed pdb into jsonl")
    argparser.add_argument("--chain_id_jsonl",type=str, default='', help="Path to a dictionary specifying which chains need to be designed and which ones are fixed, if not specied all chains will be designed.")
    argparser.add_argument("--fixed_positions_jsonl", type=str, default='', help="Path to a dictionary with fixed positions")
    argparser.add_argument("--omit_AAs", type=list, default='X', help="Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.")
    argparser.add_argument("--bias_AA_jsonl", type=str, default='', help="Path to a dictionary which specifies AA composion bias if neededi, e.g. {A: -1.1, F: 0.7} would make A less likely and F more likely.")
   
    argparser.add_argument("--bias_by_res_jsonl", default='', help="Path to dictionary with per position bias.") 
    argparser.add_argument("--omit_AA_jsonl", type=str, default='', help="Path to a dictionary which specifies which amino acids need to be omited from design at specific chain indices")
    argparser.add_argument("--pssm_jsonl", type=str, default='', help="Path to a dictionary with pssm")
    argparser.add_argument("--pssm_multi", type=float, default=0.0, help="A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions")
    argparser.add_argument("--pssm_threshold", type=float, default=0.0, help="A value between -inf + inf to restric per position AAs")
    argparser.add_argument("--pssm_log_odds_flag", type=int, default=0, help="0 for False, 1 for True")
    argparser.add_argument("--pssm_bias_flag", type=int, default=0, help="0 for False, 1 for True")
    
    argparser.add_argument("--tied_positions_jsonl", type=str, default='', help="Path to a dictionary with tied positions")

    argparser.add_argument("--score_sc_only", type=int, default=0, help="0 for False, 1 for True; score side chains")
    argparser.add_argument("--pack_only", type=int, default=0, help="0 for False, 1 for True; pack side chains for the input sequence only")
    argparser.add_argument("--pack_side_chains", type=int, default=0, help="0 for False, 1 for True; pack side chains")
    argparser.add_argument("--num_packs", type=int, default=1, help="Number of packing samples to output") 


    argparser.add_argument("--pdb_bias_path", type=str, default='', help="Path to a single PDB to be sequence biased by")
    argparser.add_argument("--pdb_bias_level", type=float, default=0.0, help="Higher number means more biased toward the pdb bias sequence")

    argparser.add_argument("--species", type=str, default="", help="Empty string will use vanilla MPNN, otherwise choose from 3 classes: 'homo_sapiens', 'bacterial', 'other' to bias sequences.")
    argparser.add_argument("--transmembrane", type=str, default="", help="Global label. Empty string will use vanilla MPNN, otherwise choose from 2 classes: 'yes', 'no'")  
    argparser.add_argument("--transmembrane_buried", type=str, default="", help="Indicate buried residue numbers.")
    argparser.add_argument("--transmembrane_interface", type=str, default="", help="Indicate interface residue numbers.")
    argparser.add_argument("--transmembrane_chain_ids", type=str, default="", help="Chain ids for the buried/interface residues; e.g. 'A,B,C,F'.")

    args = argparser.parse_args()    
    main(args)   
