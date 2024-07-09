#!/usr/bin/env python

import os
import sys
import time
import argparse
import numpy as np
import pyrosetta
from pyrosetta.rosetta.std import ostringstream

# Initialize PyRosetta
PYROSETTA_OPTIONS = "-mute all -beta_nov16 -in:file:silent_struct_type binary" \
    " -holes:dalphaball /software/rosetta/DAlphaBall.gcc" \
    " -use_terminal_residues true -precompute_ig" \
    " -dunbrack_prob_buried 0.8 -dunbrack_prob_nonburied 0.8" \
    " -dunbrack_prob_buried_semi 0.8 -dunbrack_prob_nonburied_semi 0.8" \
    " -optimization:default_max_cycles 200"

pyrosetta.init(PYROSETTA_OPTIONS)

# Add paths
sys.path.insert(0, '/home/drhicks1/scripts/mpnn_git_repo/proteinMPNN/')
import protein_mpnn_run_HACK as mpnn_util
from drh_utils import (xml,
    get_sap,
    passes_quality_checks,
    renumber_pose,
    thread_mpnn_seq,
    pymol_align,
    add2scorefile,
    add2silent,
    optimize_sap,
    record_checkpoint,
    determine_finished_structs,
    generate_seqopt_features,
    chain_mask,
    tied_positions,
    fixed_positions)

sys.path.append('/home/drhicks1/silent_tools')
import silent_tools


argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

argparser.add_argument("--checkpoint_path", type=str, default="", help="Path to the model checkpoint")
argparser.add_argument("--path_to_model_weights", type=str, default="/databases/mpnn/vanilla_model_weights/", help="Path to model weights folder;")
argparser.add_argument("--model_name", type=str, default="v_48_020", help="ProteinMPNN model name: v_48_002, v_48_010, v_48_020, v_48_030, v_32_002, v_32_010; v_32_020, v_32_030")

argparser.add_argument("--use_seed", type=int, default=0, help="0 for False, 1 for True; To set global seed.")
argparser.add_argument("--seed", type=int, default=0, help="Set seed for torch, numpy, and python random.")

argparser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension for the model")
argparser.add_argument("--num_layers", type=int, default=3, help="Number of layers for the model")
argparser.add_argument("--num_connections", type=int, default=48, help="Default 48")

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

argparser.add_argument("--backbone_noise", type=float, default=0.01, help="Standard deviation of Gaussian noise to add to backbone atoms")
argparser.add_argument("--num_seq_per_target", type=int, default=5, help="Number of sequences to generate per target")
argparser.add_argument("--batch_size", type=int, default=1, help="Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory")
argparser.add_argument("--max_length", type=int, default=20000, help="Max sequence length")
argparser.add_argument("--sampling_temp", type=str, default="0.1", help="A string of temperatures, 0.2 0.25 0.5. Sampling temperature for amino acids, T=0.0 means taking argmax, T>>1.0 means sample randomly. Suggested values 0.1, 0.15, 0.2, 0.25, 0.3. Higher values will lead to more diversity.")

argparser.add_argument("--out_folder", type=str, default='./', help="Path to a folder to output sequences, e.g. /home/out/")
argparser.add_argument("--pdb_path", type=str, default='', help="Path to a single PDB to be designed")
argparser.add_argument("--pdb_path_chains", type=str, default='', help="Define which chains need to be designed for a single PDB ")
argparser.add_argument("--jsonl_path", type=str, help="Path to a folder with parsed pdb into jsonl")
argparser.add_argument("--chain_id_jsonl",type=str, default='', help="Path to a dictionary specifying which chains need to be designed and which ones are fixed, if not specied all chains will be designed.")
argparser.add_argument("--fixed_positions_jsonl", type=str, default='', help="Path to a dictionary with fixed positions")
argparser.add_argument("--omit_AAs", type=list, default='CX', help="Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.")
argparser.add_argument("--bias_AA_jsonl", type=str, default='/home/drhicks1/scripts/mpnn_git_repo/design_scripts/bias_AA_sap.jsonl', help="Path to a dictionary which specifies AA composion bias if needed, e.g. {A: -1.1, F: 0.7} would make A less likely and F more likely.")

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

############# DRH #############
argparser.add_argument( "--target", type=str, default="", help='optional path to a target that will replace existing target' )
argparser.add_argument( "--uniprot_id", type=str, default="", help='uniprot ID for glycan check' )

argparser.add_argument( "--max_out", type=int, default=5, help='max outputs per target' )

# Rosetta optimize and/or relax
argparser.add_argument("--optimize_sap_and_relax", action="store_true", help="Enable rosetta sap optimization and relax for binder.")
argparser.add_argument("--relax", action="store_true", help="Enable rosetta relax for binder.")

# Chains and positions to design
argparser.add_argument("--design_these_chains", type=str, default="A", help="List of chains to design such as : A,B,C")
argparser.add_argument("--keep_these_chains", type=str, default="B", help="List of chains to keep as context but not design. If none, use NA")
argparser.add_argument("--tie_chainsa", type=str, default="NA", help="Chains to enforce having same sequence, use NA if not using")
argparser.add_argument("--tie_chainsb", type=str, default="NA", help="For 2 comp stuff. ANOTHER SET of chains to enforce having same sequence, use NA if not using")
argparser.add_argument("--tie_repeats", type=int, default=0, help="For DHR/toroid/repeat design. Enter the number of times sequence should be repeated in your repetitive structure. Use 0 if not using")
argparser.add_argument("--fix_a", type=str, default="NA", help="List of resnums in chain A to lock to native identity. 1-indexed. Use NA if not using")
argparser.add_argument("--fix_b", type=str, default="NA", help="List of resnums in chain B to lock to native identity. Assumes something is being locked in A also. 1-indexed. Use NA if not using")

# Filters and thresholds
argparser.add_argument("--ddg_cutoff", type=float, default=-0.0, help="Threshold that predicted ddg must pass for a structure to be written to disk.")
argparser.add_argument("--ddg_filter", action="store_true", help="Enable rosetta relax for binder.")
argparser.add_argument("--ddg_soft", action="store_true", help="Use soft ddg")

# Debugging and output
argparser.add_argument("--debug", action="store_true", help="Enable debug mode.")
argparser.add_argument("--out_name", default="out", help="Name of the output silent file.")

# required silent file
argparser.add_argument( "-silent", type=str, default="", help='The name of a silent file to load pdbs for mpnn' )

args = argparser.parse_args()

silent = args.silent
silent_out = f"{args.out_name}.silent"

# Load objects from XML
objs = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(xml)

# Helper function to get filter and handle stochastic filter cases
def get_filter_and_check_stochastic(filter_name):
    """Retrieve filter from XML objects and convert if it's a stochastic filter."""
    filter_obj = objs.get_filter(filter_name)
    if isinstance(filter_obj, pyrosetta.rosetta.protocols.filters.StochasticFilter):
        filter_obj = filter_obj.subfilter()
    return filter_obj

# Load movers
FastRelax = objs.get_mover('FastRelax')
remove_massive_clashes = objs.get_mover('remove_massive_clashes')
remove_helix_gly = objs.get_mover('remove_helix_gly')
softish_min = objs.get_mover('softish_min')
hard_min = objs.get_mover('hard_min')
my_per_res_sap = objs.get_simple_metric("my_per_res_sap")
to_fa_standard = pyrosetta.rosetta.protocols.simple_moves.SwitchResidueTypeSetMover('fa_standard')

# Load filters
if args.ddg_soft:
    ddg_filter = get_filter_and_check_stochastic('ddg_soft_norepack')
else:
    ddg_filter = get_filter_and_check_stochastic('ddg')

cms_filter = get_filter_and_check_stochastic('contact_molecular_surface')

def optimize_and_relax(pose):
    remove_helix_gly.apply(pose)
    pose = optimize_sap(pose, xml, my_per_res_sap)
    FastRelax.apply(pose)
    return pose

def sequence_optimize(pdbfile, args):
    """Optimize sequence for a given PDB file."""
    
    t0 = time.time()

    feature_dict = generate_seqopt_features(pdbfile)
    chain_mask_dict = chain_mask(feature_dict, args.design_these_chains, args.keep_these_chains)

    tied_positions_out = tied_positions(feature_dict, args.tie_chainsb, args.tie_chainsa, args.tie_repeats)
    fixed_positions_out = fixed_positions(feature_dict, args.fix_a, args.fix_b)
    
    args.tied_positions_jsonl = tied_positions_out
    args.fixed_positions_jsonl = fixed_positions_out
    args.chain_id_jsonl = chain_mask_dict
    args.jsonl_path = feature_dict

    args.pdb_path = pdbfile
    sequences = mpnn_util.main(args)

    # Step 1: Remove redundant sequences while keeping the best (lowest) score
    unique_sequences = {}
    for seq, score in sequences:
        if seq not in unique_sequences or float(score) < float(unique_sequences[seq]):
            unique_sequences[seq] = score

    # Step 2: Convert the dictionary back to a list of tuples
    unique_sequence_list = [(seq, score) for seq, score in unique_sequences.items()]

    # Step 3: Sort the list by score
    unique_sequence_list.sort(key=lambda x: float(x[1]))

    sequences = unique_sequence_list

    print(f"MPNN generated {len(sequences)} sequences in {int(time.time() - t0)} seconds")
    print("seq:score")
    for seq in sequences:
        print(f"{seq[0]}:{seq[1]}")

    sequences = [seq[0] for seq in sequences]

    return split_sequence_by_chain(sequences, feature_dict, chain_mask_dict, args.design_these_chains)

def split_sequence_by_chain(sequences, feature_dict, chain_mask_dict, design_these_chains):
    """Split sequences by chain based on mask and feature dictionary."""
    
    masked_list = [letter for letter in chain_mask_dict[feature_dict['name']][0]]
    masked_chain_length_list = [len(feature_dict[f'seq_chain_{letter}']) for letter in masked_list]
    
    split_sequences = []
    if len(design_these_chains.split(",")) > 1:
        for seq in sequences:
            l0 = 0
            for mc_length in sorted(masked_chain_length_list)[:-1]:
                l0 += mc_length
                seq = seq[:l0] + '/' + seq[l0:]
                l0 += 1
            split_sequences.append(seq)
    else:
        split_sequences = sequences
        
    return split_sequences


def dl_design( pose, tag, silent_out, sfd_out , args):

    design_counter = 0

    prefix = f"{tag}_dldesign"

    pdb_stream = ostringstream()
    pose.dump_pdb(pdb_stream)
    pdbfile = pdb_stream.str()

    seqs = sequence_optimize( pdbfile, args )

    for idx,seq in enumerate( seqs ):
        tag = f"{prefix}_{idx}"

        pose = thread_mpnn_seq(pose, seq)

        remove_massive_clashes.apply(pose)

        if args.optimize_sap_and_relax:
            pose = optimize_and_relax(pose)

        if args.relax:
            FastRelax.apply(pose)

        if args.ddg_filter:
            try:
                ddg = ddg_filter.compute(pose)
                if ddg > args.ddg_cutoff: continue
            except: 
                ddg = 999
        else:
            ddg = 999
        
        try:
            contact_molecular_surface = cms_filter.compute(pose)
        except:
            contact_molecular_surface = 0

        try:
            sap_score = get_sap(pose.split_by_chain()[1])
        except:
            sap_score = 999

        add2silent( pose, tag, sfd_out , silent_out)

        add2scorefile( tag, f"{args.out_name}.sc", {'ddg':ddg, 'contact_molecular_surface':contact_molecular_surface, 'sap':sap_score} )

        design_counter += 1
    
        if design_counter >= args.max_out: return design_counter

    return design_counter

def main( pdb, silent_out, sfd_in, sfd_out ):

    t0 = time.time()
    print( "Attempting pose: %s"%pdb )
    
    # Load pose from PDB
    pose = pyrosetta.Pose()
    sfd_in.get_structure( pdb ).fill_pose( pose )
    if args.target != "":
        target_pose = args.target
        split_pose = pose.split_by_chain()
        pose = split_pose[1]

        _, _, target_pose = pymol_align(target_pose, split_pose[2])

        pose.append_pose_by_jump(target_pose, 1)

        renumber_pose(pose)

        to_fa_standard.apply( pose )

    #if passes_quality_checks(pdb, pose, args.uniprot_id):
    if 1:
        
        good_designs = dl_design( pose, pdb, silent_out, sfd_out, args)

        seconds = int(time.time() - t0)

        print( f"protocols.jd2.JobDistributor: {pdb} reported success. {good_designs} designs generated in {seconds} seconds" )
    
    else:
        print(f"{pdb} failed quality checks")

#################################
# Begin Main Loop
#################################
checkpoint_filename = "check.point"
debug = args.debug

silent_index = silent_tools.get_silent_index(silent)

if not os.path.isfile(silent_out):
    with open(silent_out, 'w') as f: f.write(silent_tools.silent_header(silent_index))

sfd_out = pyrosetta.rosetta.core.io.silent.SilentFileData(silent_out, False, False, "binary", pyrosetta.rosetta.core.io.silent.SilentFileOptions())

sfd_in = pyrosetta.rosetta.core.io.silent.SilentFileData(pyrosetta.rosetta.core.io.silent.SilentFileOptions())
sfd_in.read_file(silent)

finished_structs = determine_finished_structs( checkpoint_filename )

if args.target != "":
    target_pose = pyrosetta.pose_from_pdb(args.target)

for pdb in silent_index['tags']:
    if pdb in finished_structs: continue

    if args.target != "":
        args.target = target_pose.clone()

    if debug: main( pdb, silent_out, sfd_in, sfd_out )

    else:
        t0 = time.time()

        try: main( pdb, silent_out, sfd_in, sfd_out )

        except KeyboardInterrupt: sys.exit( "Script killed by Control+C, exiting" )

        except:
            seconds = int(time.time() - t0)
            print( "protocols.jd2.JobDistributor: %s failed in %i seconds with error: %s"%( pdb, seconds, sys.exc_info()[0] ) )
            continue

    # We are done with one pdb, record that we finished
    record_checkpoint( pdb, checkpoint_filename )
