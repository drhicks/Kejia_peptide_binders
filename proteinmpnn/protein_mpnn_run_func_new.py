import argparse
import os.path

def bb_mpnn(args):

    seqs_scores = []
    
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
    from protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB, parse_fasta
    from protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN
    from protein_mpnn_utils import mcmc_step
    from sc_utils import pack_side_chains, build_sc_model, score_side_chains 
    from run_af2_for_mcmc import compile_model
    from mcmc_predictors import CountPredictor, AF2Geom, ChargePredictor
    
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
    else:
        species_label = -1


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


    folder_for_outputs = args.out_folder
    
    NUM_BATCHES = args.num_seq_per_target//args.batch_size
    BATCH_COPIES = args.batch_size
    temperatures = [float(item) for item in args.sampling_temp.split()]
    omit_AAs_list = args.omit_AAs
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    alphabet_dict = dict(zip(alphabet, range(21))) 
    omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    
    dataset_valid = [args.jsonl_path]

    chain_id_dict = args.chain_id_jsonl
    
    fixed_positions_dict = args.fixed_positions_jsonl

    tied_positions_dict = args.tied_positions_jsonl
    
    omit_AA_dict = args.omit_AA_jsonl
    
    if os.path.isfile(args.pssm_jsonl):
        with open(args.pssm_jsonl, 'r') as json_file:
            json_list = list(json_file)
        pssm_dict = {}
        for json_str in json_list:
            pssm_dict.update(json.loads(json_str))
    else:
        print(40*'-')
        print('pssm_jsonl is NOT loaded')
        pssm_dict = None
    
    
    if os.path.isfile(args.bias_AA_jsonl):
        with open(args.bias_AA_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            bias_AA_dict = json.loads(json_str)
    else:
        print(40*'-')
        print('bias_AA_jsonl is NOT loaded')
        bias_AA_dict = None
    
    
    if os.path.isfile(args.bias_by_res_jsonl):
        with open(args.bias_by_res_jsonl, 'r') as json_file:
            json_list = list(json_file)
    
        for json_str in json_list:
            bias_by_res_dict = json.loads(json_str)
        print('bias by residue dictionary is loaded')
    else:
        print(40*'-')
        print('bias by residue dictionary is not loaded, or not provided')
        bias_by_res_dict = None
   
 
    print(40*'-')
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

        
    print(40*'-')
    if not args.pack_only:
        checkpoint = torch.load(checkpoint_path, map_location=device) 
        print('Number of edges:', checkpoint['num_edges'])
        noise_level_print = checkpoint['noise_level']
        print(f'Training noise level: {noise_level_print}A')
        model = ProteinMPNN(num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=args.backbone_noise, k_neighbors=checkpoint['num_edges'], use_label=bool(args.species) or bool(args.transmembrane), label=species_label)
        model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    # Timing
    start_time = time.time()
    total_residues = 0
    protein_list = []
    total_step = 0
    if args.pack_side_chains or args.pack_only or args.score_sc_only:
        sc_model = build_sc_model()
    # Validation epoch
    if args.do_mcmc == 1 and args.mcmc_predictor=='af2_geometry':
        t0 = time.time()
        crop_size = max([len(p['seq']) for p in dataset_valid])
        model_runner, runner_vmaped, use_model, model_params = compile_model(
            args.num_af2_models, args.af2_params_path, crop_size, args.num_recycle)
    with torch.no_grad():
        test_sum, test_weights = 0., 0.
        #print('Generating sequences...')
        for ix, protein in enumerate(dataset_valid):
            score_list = []
            all_probs_list = []
            all_log_probs_list = []
            S_sample_list = []
            batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(batch_clones, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict, parse_all_atoms=args.score_sc_only)
            pssm_log_odds_mask = (pssm_log_odds_all > args.pssm_threshold).float() #1.0 for true, 0.0 for false
            name_ = batch_clones[0]['name']
            if args.score_sc_only:
                score_side_chains(X[0][None], args.num_packs, sc_model, S[0][None], X[0][None], mask[0][None], batch_clones[0]['name']+f'_seq_input_score_packed', base_folder + 'packed', residue_idx[0][None], chain_encoding_all[0][None])
            elif args.pack_only:
                pack_side_chains(args.num_packs, sc_model, S[0][None], X[0][None], mask[0][None], batch_clones[0]['name']+f'_seq_input_packed', base_folder + 'packed', residue_idx[0][None], chain_encoding_all[0][None])
            else:
                if args.score_only:
                    loop_c = 0
                    if args.path_to_fasta:
                        fasta_names, fasta_seqs = parse_fasta(args.path_to_fasta)
                        loop_c = len(fasta_seqs)
                    for fc in range(1+loop_c):
                        if fc == 0:
                            structure_sequence_score_file = base_folder + '/score_only/' + batch_clones[0]['name'] + f'_from_pdb' + '.npy'
                        else:
                            structure_sequence_score_file = base_folder + '/score_only/' + batch_clones[0]['name'] + f'_from_fasta_{fc}' + '.npy'
                        native_score_list = []
                        if fc > 0:
                            S = torch.tensor([alphabet_dict[AA] for AA in fasta_seqs[fc-1] if AA in list(alphabet_dict.keys())], device=device)[None,:].repeat(X.shape[0], 1)
                        for j in range(NUM_BATCHES):
                            randn_1 = torch.randn(chain_M.shape, device=X.device)
                            log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
                            mask_for_loss = mask*chain_M*chain_M_pos
                            scores = _scores(S, log_probs, mask_for_loss)
                            native_score = scores.cpu().data.numpy()
                            native_score_list.append(native_score)
                        native_score = np.concatenate(native_score_list, 0)
                        ns_mean = native_score.mean()
                        ns_mean_print = np.format_float_positional(np.float32(ns_mean), unique=False, precision=4)
                        ns_std = native_score.std()
                        ns_std_print = np.format_float_positional(np.float32(ns_std), unique=False, precision=4)
                        ns_sample_size = native_score.shape[0]
                        np.save(structure_sequence_score_file, native_score)
                        if fc == 0:
                            print(f'From .pdb; score for {name_}, mean: {ns_mean_print}, std: {ns_std_print}, sample size: {ns_sample_size}')
                        else:
                            print(f'From .fasta; score for {name_}, mean: {ns_mean_print}, std: {ns_std_print}, sample size: {ns_sample_size}')
                elif args.conditional_probs_only:
                    print(f'Calculating sequence conditional probabilities for {name_}')
                    conditional_probs_only_file = base_folder + '/conditional_probs_only/' + batch_clones[0]['name']
                    log_conditional_probs_list = []
                    for j in range(NUM_BATCHES):
                        randn_1 = torch.randn(chain_M.shape, device=X.device)
                        if args.conditional_probs_use_pseudo:
                            log_conditional_probs = model.conditional_probs_pseudo(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1, args.conditional_probs_only_backbone)
                        else:
                            log_conditional_probs = model.conditional_probs(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1, args.conditional_probs_only_backbone)
                        log_conditional_probs_list.append(log_conditional_probs.cpu().numpy())
                    concat_log_p = np.concatenate(log_conditional_probs_list, 0) #[B, L, 21]
                    mask_out = (chain_M*chain_M_pos*mask)[0,].cpu().numpy()
                    S_np = S[0,].cpu().numpy()
                    log_p_of_input = np.take_along_axis(concat_log_p, S_np[None,:,None], -1)
                    log_odds = concat_log_p - log_p_of_input
                    np.savez(conditional_probs_only_file, log_p=concat_log_p, S=S_np, mask=mask[0,].cpu().numpy(), design_mask=mask_out, log_odds=log_odds)
                elif args.unconditional_probs_only:
                    print(f'Calculating sequence unconditional probabilities for {name_}')
                    unconditional_probs_only_file = base_folder + '/unconditional_probs_only/' + batch_clones[0]['name']
                    log_unconditional_probs_list = []
                    for j in range(NUM_BATCHES):
                        log_unconditional_probs = model.unconditional_probs(X, mask, residue_idx, chain_encoding_all)
                        log_unconditional_probs_list.append(log_unconditional_probs.cpu().numpy())
                    concat_log_p = np.concatenate(log_unconditional_probs_list, 0) #[B, L, 21]
                    mask_out = (chain_M*chain_M_pos*mask)[0,].cpu().numpy()
                    S_np = S[0,].cpu().numpy()
                    log_p_of_input = np.take_along_axis(concat_log_p, S_np[None,:,None], -1)
                    log_odds = concat_log_p - log_p_of_input
                    np.savez(unconditional_probs_only_file, log_p=concat_log_p, S=S_np, mask=mask[0,].cpu().numpy(), design_mask=mask_out, log_odds=log_odds)
                else:
                    if args.compute_input_sequence_score:
                        randn_1 = torch.randn(chain_M.shape, device=X.device)
                        log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
                        mask_for_loss = mask*chain_M*chain_M_pos
                        scores = _scores(S, log_probs, mask_for_loss)
                        native_score = scores.cpu().data.numpy()
                    else:
                        native_score = np.array([0.000000, 0.000000])
                    # Generate some sequences
                    print(f'Generating sequences for: {name_}')
                    t0 = time.time()
                    # Init Predictor for MCMC
                    if args.do_mcmc == 1:
                        if args.mcmc_predictor =='af2_geometry':
                            predictor = AF2Geom(
                                runner_vmaped, 
                                model_runner, 
                                use_model, 
                                name_, 
                                args.af2_random_seed,
                                args.af2_params_path,
                                args.out_folder,
                                args.num_af2_models,
                                model_params,
                            )
                        elif args.mcmc_predictor == "count":
                            predictor = CountPredictor()
                        elif args.mcmc_predictor == "charge":
                            charges = [int(item) for item in args.mcmc_charge_list.split()]
                            predictor = ChargePredictor(charges)

                    for temp_c, temp in enumerate(temperatures):
                        e_scores = np.zeros([NUM_BATCHES*BATCH_COPIES, args.mcmc_steps+1])
                        seqs = np.zeros([NUM_BATCHES*BATCH_COPIES, args.mcmc_steps+1, X.shape[1]])
                        acceptance_prob = np.zeros([NUM_BATCHES*BATCH_COPIES, args.mcmc_steps])
                        for j in range(NUM_BATCHES):
                            if args.do_mcmc == 1:
                                randn_2 = torch.randn(chain_M.shape, device=X.device)
                                sample_kwargs = dict(X=X, randn=randn_2, S_true=S, chain_mask= chain_M, chain_encoding_all=chain_encoding_all, residue_idx=residue_idx, 
                                        mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, 
                                        bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, 
                                        pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=args.pssm_multi, 
                                        pssm_log_odds_flag=bool(args.pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask,
                                        pssm_bias_flag=bool(args.pssm_bias_flag), 
                                        ### biasing
                                        bias_by_res=bias_by_res_all)
                                    
                                # Start by sampling a sequence from pMPNN and evaluating its score
                                S_curr = model.sample(**sample_kwargs)['S']
                                E_curr = predictor.predict(S_curr, -1, j, BATCH_COPIES, args.pdb_path, masked_chain_length_list_list[0])
                                idx1 = j*BATCH_COPIES
                                idx2 = (j+1)*BATCH_COPIES
                                e_scores[idx1:idx2,0] = E_curr.detach().cpu().numpy()
                                seqs[idx1:idx2,0,:] = S_curr.detach().cpu().numpy()
                                sample_dict = None
                                for step in range(args.mcmc_steps):
                                    # create the residue bias for the sequence ( logits are -10 to 10) 
                                    bias_by_res_all = torch.nn.functional.one_hot(S_curr, 21).float() 
                                    sample_kwargs['bias_by_res'] = bias_by_res_all * args.mcmc_bias_weight # multiply to weight bias
                                    sample_kwargs['randn'] = torch.randn(chain_M.shape, device=X.device)
                                    S_curr, E_curr, sample_dict, profile = mcmc_step(
                                        S_curr, E_curr, model, sample_kwargs, sample_dict, args.mcmc_temperature, step, j, BATCH_COPIES, predictor, args.pdb_path, masked_chain_length_list_list[0])
                                    print("MCMC step:", step, "energies:", E_curr, "seq. mut.:", profile[2])
                                    e_scores[idx1:idx2,step+1] = E_curr.detach().cpu().numpy()
                                    seqs[idx1:idx2,step+1,:] = S_curr.detach().cpu().numpy()
                                    acceptance_prob[idx1:idx2, step] = profile[1].detach().cpu().numpy()

                                S_sample = S_curr
                            else:
                                randn_2 = torch.randn(chain_M.shape, device=X.device)
                                if tied_positions_dict == None:
                                    sample_dict = model.sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=args.pssm_multi, pssm_log_odds_flag=bool(args.pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(args.pssm_bias_flag), bias_by_res=bias_by_res_all)
                                    S_sample = sample_dict["S"] 
                                else:
                                    sample_dict = model.tied_sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=args.pssm_multi, pssm_log_odds_flag=bool(args.pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(args.pssm_bias_flag), tied_pos=tied_pos_list_of_lists_list[0], tied_beta=tied_beta, bias_by_res=bias_by_res_all, assume_symmetry=bool(args.assume_symmetry))
                                # Compute scores
                                    S_sample = sample_dict["S"]
                            mask_for_loss = mask*chain_M*chain_M_pos
                            scores = _scores(S_sample, sample_dict["log_probs"], mask_for_loss)
                            scores = scores.cpu().data.numpy()
                            all_probs_list.append(sample_dict["probs"].cpu().data.numpy())
                            all_log_probs_list.append(sample_dict["log_probs"].cpu().data.numpy())
                            S_sample_list.append(S_sample.cpu().data.numpy())

                            seqs_1h = np.eye(21)[seqs.astype(np.int32)].astype(np.int32)
                            seq_sims = np.mean(np.sum(seqs_1h[:,:,None,None,:,:]*seqs_1h[None,None,:,:,:],-1),-1)
                            if args.do_mcmc:
                                np.savez(base_folder+'mcmc_run_profile', e_scores=e_scores, seqs=seqs, seq_similarities=seq_sims)

                            for b_ix in range(BATCH_COPIES):
                                masked_chain_length_list = masked_chain_length_list_list[b_ix]
                                masked_list = masked_list_list[b_ix]
                                seq_recovery_rate = torch.sum(torch.sum(torch.nn.functional.one_hot(S[b_ix], 21)*torch.nn.functional.one_hot(S_sample[b_ix], 21),axis=-1)*mask_for_loss[b_ix])/torch.sum(mask_for_loss[b_ix])
                                seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
                                score = scores[b_ix]
                                score_list.append(score)
                                native_seq = _S_to_seq(S[b_ix], chain_M[b_ix])
                                if b_ix == 0 and j==0 and temp==temperatures[0]:
                                    start = 0
                                    end = 0
                                    list_of_AAs = []
                                    for mask_l in masked_chain_length_list:
                                        end += mask_l
                                        list_of_AAs.append(native_seq[start:end])
                                        start = end
                                    native_seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                                    l0 = 0
                                    for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                                        l0 += mc_length
                                        native_seq = native_seq[:l0] + '/' + native_seq[l0:]
                                        l0 += 1
                                    sorted_masked_chain_letters = np.argsort(masked_list_list[0])
                                    print_masked_chains = [masked_list_list[0][i] for i in sorted_masked_chain_letters]
                                    sorted_visible_chain_letters = np.argsort(visible_list_list[0])
                                    print_visible_chains = [visible_list_list[0][i] for i in sorted_visible_chain_letters]
                                    native_score_print = np.format_float_positional(np.float32(native_score.mean()), unique=False, precision=4)
                                    script_dir = os.path.dirname(os.path.realpath(__file__))                                    
                                start = 0
                                end = 0
                                list_of_AAs = []
                                for mask_l in masked_chain_length_list:
                                    end += mask_l
                                    list_of_AAs.append(seq[start:end])
                                    start = end

                                seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                                l0 = 0
                                for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                                    l0 += mc_length
                                    seq = seq[:l0] + '/' + seq[l0:]
                                    l0 += 1
                                score_print = np.format_float_positional(np.float32(score), unique=False, precision=4)
                                seqs_scores.append([seq, score_print])

                    t1 = time.time()
                    dt = round(float(t1-t0), 4)
                    num_seqs = len(temperatures)*NUM_BATCHES*BATCH_COPIES
                    total_length = X.shape[1]
                    print(f'{num_seqs} sequences of length {total_length} generated in {dt} seconds')

                    return seqs_scores
   
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

    argparser.add_argument("--mcmc_steps", type=int, default=5, help="How many MCMC steps to do [5]")
    argparser.add_argument("--do_mcmc", type=int, default=0, help="if doing MCMC")
    argparser.add_argument("--num_af2_models", type=int, default=1, help="define how many AF2 models to use during MCMC")
    argparser.add_argument("--af2_batch_size", type=int, default=1, help="for GPUs")
    argparser.add_argument("--num_recycle", type=int, default=3, help="Recycling steps for AF2")
    argparser.add_argument("--af2_random_seed", type=int, default=3, help="AF2's random seed")
    argparser.add_argument("--af2_params_path", type=str, default="/projects/ml/alphafold", help="Path to AF2 parameters")
    #argparser.add_argument("--base_mcmc_pdb_folder", type=str, default="/mnt/home/dnan/projects/rotationdb/mcmc_intermediate", help="Path to intermediate outputs of AF2 pdbs")
    argparser.add_argument('--mcmc_predictor', type=str, choices=['charge', 'count', 'af2_geometry'])
    argparser.add_argument("--mcmc_bias_weight", type=float, default=0.5, help="Higher bias weight will lead to less number of mutations in the mcmc step.")
    argparser.add_argument("--mcmc_temperature", type=float, default=0.05, help="mcmc temperature prob = exp([E1-E0]/temperature); low temperature means greedy mcmc.")   

    argparser.add_argument("--mcmc_charge_list", type=str, default="0", help="A string of charges for chains A, B, C,..., 0 -1 4")

    argparser.add_argument("--species", type=str, default="", help="Empty string will use vanilla MPNN, otherwise choose from 3 classes: 'homo_sapiens', 'bacterial', 'other' to bias sequences.")
    argparser.add_argument("--transmembrane", type=str, default="", help="Empty string will use vanilla MPNN, otherwise choose from 2 classes: 'yes', 'no'")  
  
    args = argparser.parse_args()    
    main(args)   
