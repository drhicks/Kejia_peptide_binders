#!/usr/bin/env python

import os, sys
import math

sys.path.insert( 0, '/home/nrbennet/rosetta_builds/master_branch/pyrosetta_builds/py39_builds/build1' )
sys.path.insert(0, '/home/nrbennet/protocols/dl/dl_design/justas_seq_op/single_chain/')

from pyrosetta import *
from pyrosetta.rosetta import *

init( "-mute all -beta_nov16 -in:file:silent_struct_type binary" +
    " -holes:dalphaball /software/rosetta/DAlphaBall.gcc" +
    " -use_terminal_residues true -mute basic.io.database core.scoring" +
    " -dunbrack_prob_buried 0.8 -dunbrack_prob_nonburied 0.8" +
    " -dunbrack_prob_buried_semi 0.8 -dunbrack_prob_nonburied_semi 0.8")
#    " -optimization:default_max_cycles 200")

import numpy as np
from collections import defaultdict
import time
import argparse
import itertools
import subprocess
import time
import pandas as pd
import glob
from decimal import Decimal
from collections import OrderedDict

import torch

sys.path.insert(0, '/home/drhicks1/scripts/proteinmpnn/')
import protein_mpnn_run_func_new as mpnn_util

sys.path.append( '/home/nrbennet/software/silent_tools' )
import silent_tools

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
argparser.add_argument("--num_seq_per_target", type=int, default=5, help="Number of sequences to generate per target")
argparser.add_argument("--batch_size", type=int, default=1, help="Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory")
argparser.add_argument("--max_length", type=int, default=20000, help="Max sequence length")
argparser.add_argument("--sampling_temp", type=str, default="0.001 0.01 0.1 0.15 0.2", help="A string of temperatures, 0.2 0.25 0.5. Sampling temperature for amino acids, T=0.0 means taking argmax, T>>1.0 means sample randomly. Suggested values 0.1, 0.15, 0.2, 0.25, 0.3. Higher values will lead to more diversity.")

argparser.add_argument("--out_folder", type=str, help="Path to a folder to output sequences, e.g. /home/out/")
argparser.add_argument("--pdb_path", type=str, default='', help="Path to a single PDB to be designed")
argparser.add_argument("--pdb_path_chains", type=str, default='', help="Define which chains need to be designed for a single PDB ")
argparser.add_argument("--jsonl_path", type=str, help="Path to a folder with parsed pdb into jsonl")
argparser.add_argument("--chain_id_jsonl",type=str, default='', help="Path to a dictionary specifying which chains need to be designed and which ones are fixed, if not specied all chains will be designed.")
argparser.add_argument("--fixed_positions_jsonl", type=str, default='', help="Path to a dictionary with fixed positions")
argparser.add_argument("--omit_AAs", type=list, default='CX', help="Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.")
argparser.add_argument("--bias_AA_jsonl", type=str, default='/home/drhicks1/scripts/proteinmpnn/design_scripts/bias_AA_kejia.jsonl', help="Path to a dictionary which specifies AA composion bias if neededi, e.g. {A: -1.1, F: 0.7} would make A less likely and F more likely.")

argparser.add_argument("--bias_by_res_jsonl", default='', help="Path to dictionary with per position bias.") 
argparser.add_argument("--omit_AA_jsonl", type=str, default=None, help="Path to a dictionary which specifies which amino acids need to be omited from design at specific chain indices")
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

#from TimH
argparser.add_argument("--design_these_chains", type=str, default="A", help="List of chains to design such as : A,B,C")
argparser.add_argument("--keep_these_chains", type=str, default="B", help="List of chains to keep as context but not design. If none, use NA")
argparser.add_argument("--tie_chainsa", type=str, default="NA", help="Chains to enforce having same sequence, use NA if not using")
argparser.add_argument("--tie_chainsb", type=str, default="NA", help="For 2 comp stuff. ANOTHER SET of chains to enforce having same sequence, use NA if not using")
argparser.add_argument("--tie_repeats", type=int, default=0, help="For DHR/toroid/repeat design. Enter the number of times sequence should be repeated in your repetitive structure. Use 0 if not using")
argparser.add_argument("--fix_a", type=str, default="NA", help="List of resnums in chain A to lock to native identity. 1-indexed. Use NA if not using")
argparser.add_argument("--fix_b", type=str, default="NA", help="List of resnums in chain B to lock to native identity. Assumes something is being locked in A also. 1-indexed. Use NA if not using")

argparser.add_argument( "-silent", type=str, default="", help='The name of a silent file to run this metric on. pdbs are not accepted at this point in time' )
argparser.add_argument( "--out_name", type=str, default="out", help='The name of your out silent file' )

argparser.add_argument( "--ddg_cutoff", type=float, default=100, help='The threshold that predicted ddg must pass for a structure to be written to disk (default 100)' )
argparser.add_argument( "--patchdock_res", type=str, default="1", help='what aa should we fix your xlink motif as' )
argparser.add_argument( "--motif_aa", type=str, default="Y", help='what aa should we fix your xlink motif as' )
argparser.add_argument( "--max_out", type=int, default=5, help='max designs to output' )
argparser.add_argument( "--loop_aas", type=str, default="DGNPSTV", help='what aa can your loops be' )
argparser.add_argument( "--idealize_A", type=int, default=0, help='idealize chain A? set option to 1' )

args = argparser.parse_args()    

silent = args.silent

#load movers
#xml = "/home/nrbennet/protocols/xml/relax/dlpredictor/FastRelax/predictor_nodesign.xml"
#objs = protocols.rosetta_scripts.XmlObjects.create_from_file( xml )
#switch to own xml to add interface relax script to FastRelax to prevent over packing
xml = """<ROSETTASCRIPTS>
    <SCOREFXNS>
        <ScoreFunction name="sfxn" weights="beta_nov16" >
        </ScoreFunction>
        <ScoreFunction name="sfxn_relax" weights="beta_nov16" >
            <Reweight scoretype="arg_cation_pi" weight="3" />
            <Reweight scoretype="approximate_buried_unsat_penalty" weight="5" />
            <Set approximate_buried_unsat_penalty_burial_atomic_depth="3.5" />
            <Set approximate_buried_unsat_penalty_hbond_energy_threshold="-0.5" />
            <Set approximate_buried_unsat_penalty_natural_corrections1="true" />
        </ScoreFunction>
        <ScoreFunction name="sfxn_softish" weights="beta_nov16" >
            <Reweight scoretype="fa_rep" weight="0.15" />
        </ScoreFunction>
        
    </SCOREFXNS>
    <RESIDUE_SELECTORS>
        <Chain name="chainA" chains="A"/>
        <Chain name="chainB" chains="B"/>
        <Neighborhood name="interface_chA" selector="chainB" distance="8.0" />
        <Neighborhood name="interface_chB" selector="chainA" distance="8.0" />
        <And name="AB_interface" selectors="interface_chA,interface_chB" />
        <Not name="Not_interface" selector="AB_interface" />
        <And name="actual_interface_chB" selectors="AB_interface,chainB" />
        <And name="not_interface_chB" selectors="Not_interface,chainB" />

        <True name="all" />
    
        <Slice name="patchdock_res" indices="%%patchdock_res%%" selector="chainB" />

    </RESIDUE_SELECTORS>

    <RESIDUE_SELECTORS>
        <!-- Layer Design -->
        <Layer name="surface" select_core="false" select_boundary="false" select_surface="true" use_sidechain_neighbors="true"/>
        <Layer name="boundary" select_core="false" select_boundary="true" select_surface="false" use_sidechain_neighbors="true"/>
        <Layer name="core" select_core="true" select_boundary="false" select_surface="false" use_sidechain_neighbors="true"/>
        <SecondaryStructure name="sheet" overlap="0" minH="3" minE="2" include_terminal_loops="false" use_dssp="true" ss="E"/>
        <SecondaryStructure name="entire_loop" overlap="0" minH="3" minE="2" include_terminal_loops="true" use_dssp="true" ss="L"/>
        <SecondaryStructure name="entire_helix" overlap="0" minH="3" minE="2" include_terminal_loops="false" use_dssp="true" ss="H"/>
        <And name="helix_cap" selectors="entire_loop">
            <PrimarySequenceNeighborhood lower="1" upper="0" selector="entire_helix"/>
        </And>
        <And name="helix_start" selectors="entire_helix">
            <PrimarySequenceNeighborhood lower="0" upper="1" selector="helix_cap"/>
        </And>
        <And name="helix" selectors="entire_helix">
            <Not selector="helix_start"/>
        </And>
        <And name="loop" selectors="entire_loop">
            <Not selector="helix_cap"/>
        </And>

        <ResidueName name="gly_positions" residue_name3="GLY" />

        <And name="helix_gly" selectors="entire_helix,gly_positions"/>
        <Not name="not_helix_gly" selector="helix_gly"/>

    </RESIDUE_SELECTORS>

    <TASKOPERATIONS>
        <DesignRestrictions name="layer_design_no_core_polars">
            <Action selector_logic="surface AND helix_start"  aas="DEHKPQR"/>
            <Action selector_logic="surface AND helix"        aas="EHKQR"/>
            <Action selector_logic="surface AND sheet"        aas="EHKNQRST"/>
            <Action selector_logic="surface AND loop"         aas="DEGHKNPQRST"/>
            <Action selector_logic="boundary AND helix_start" aas="ADEHIKLNPQRSTVWY"/>
            <Action selector_logic="boundary AND helix"       aas="ADEHIKLNQRSTVWY"/>
            <Action selector_logic="boundary AND sheet"       aas="DEFHIKLNQRSTVWY"/>
            <Action selector_logic="boundary AND loop"        aas="ADEFGHIKLNPQRSTVWY"/>
            <Action selector_logic="core AND helix_start"     aas="AFILMPVWY"/>
            <Action selector_logic="core AND helix"           aas="AFILVWY"/>
            <Action selector_logic="core AND sheet"           aas="FILVWY"/>
            <Action selector_logic="core AND loop"            aas="AFGILPVWY"/>
            <Action selector_logic="helix_cap"                aas="DNST"/>
        </DesignRestrictions>
    </TASKOPERATIONS>

    <TASKOPERATIONS>
        <ProteinInterfaceDesign name="pack_long" design_chain1="0" design_chain2="0" jump="1" interface_distance_cutoff="15"/>
        <IncludeCurrent name="current" />
        <LimitAromaChi2 name="limitchi2" chi2max="110" chi2min="70" include_trp="True" />
        <ExtraRotamersGeneric name="ex1_ex2" ex1="1" ex2="1" />


        <OperateOnResidueSubset name="only_helix_gly" selector="not_helix_gly">
            <PreventRepackingRLT/>
        </OperateOnResidueSubset>
        <OperateOnResidueSubset name="restrict_target_not_interface" selector="not_interface_chB">
            <PreventRepackingRLT/>
        </OperateOnResidueSubset>
        <OperateOnResidueSubset name="restrict2repacking" selector="all">
            <RestrictToRepackingRLT/>
        </OperateOnResidueSubset>
        <OperateOnResidueSubset name="restrict_to_interface" selector="Not_interface">
            <PreventRepackingRLT/>
        </OperateOnResidueSubset>
        <OperateOnResidueSubset name="restrict_target2repacking" selector="chainB">
            <RestrictToRepackingRLT/>
        </OperateOnResidueSubset>
        <OperateOnResidueSubset name="restrict_target" selector="chainB">
            <PreventRepackingRLT/>
        </OperateOnResidueSubset>
        
    </TASKOPERATIONS>
    <MOVERS>

        <SwitchChainOrder name="chain1onlypre" chain_order="1" />
        <ScoreMover name="scorepose" scorefxn="sfxn" verbose="false" />
        <ParsedProtocol name="chain1only">
            <Add mover="chain1onlypre" />
            <Add mover="scorepose" />
        </ParsedProtocol>
        <TaskAwareMinMover name="min" scorefxn="sfxn" bb="0" chi="1" task_operations="pack_long" />

    </MOVERS>
    <FILTERS>

        <Ddg name="ddg" threshold="0" jump="1" repeats="3" repack="1" relax_mover="min" confidence="0" scorefxn="sfxn" extreme_value_removal="1" />
        <ShapeComplementarity name="interface_sc" verbose="0" min_sc="0.55" write_int_area="1" write_median_dist="1" jump="1" confidence="0"/>

        ### score function monomer terms
        <ScoreType name="total_score_MBF" scorefxn="sfxn" score_type="total_score" threshold="0" confidence="0" />
        <MoveBeforeFilter name="total_score_monomer" mover="chain1only" filter="total_score_MBF" confidence="0" />
        <ResidueCount name="res_count_MBF" max_residue_count="9999" confidence="0"/>
        <MoveBeforeFilter name="res_count_monomer" mover="chain1only" filter="res_count_MBF" confidence="0" />

        <CalculatorFilter name="score_per_res" equation="total_score_monomer / res" threshold="-3.5" confidence="0">
            <Var name="total_score_monomer" filter="total_score_monomer"/>
            <Var name="res" filter="res_count_monomer"/>
        </CalculatorFilter>

        <ContactMolecularSurface name="contact_molecular_surface" distance_weight="0.5" target_selector="chainA" binder_selector="chainB" confidence="0" />
        <ContactMolecularSurface name="contact_patch" distance_weight="0.5" target_selector="patchdock_res" binder_selector="chainA" confidence="0" />

    </FILTERS>


    <MOVERS>

        <LoadPoseFromPDBLite name="load_from_lite" />

        <PackRotamersMover name="remove_helix_gly" scorefxn="sfxn" task_operations="current,ex1_ex2,limitchi2,restrict_target_not_interface,restrict_target2repacking,layer_design_no_core_polars,only_helix_gly"/>

        <PackRotamersMover name="pack_no_design" scorefxn="sfxn" task_operations="current,ex1_ex2,limitchi2,restrict_to_interface,restrict2repacking"/>
        <PackRotamersMover name="remove_massive_clashes" scorefxn="sfxn" task_operations="current,restrict2repacking,restrict_target_not_interface"/>
        <PackRotamersMover name="pack_binder" scorefxn="sfxn" task_operations="current,restrict2repacking,restrict_target"/>

       <TaskAwareMinMover name="softish_min" scorefxn="sfxn_softish" chi="1" bb="1" jump="1" tolerance="0.01" cartesian="false" task_operations="restrict_target_not_interface,restrict_target2repacking" />

       <TaskAwareMinMover name="hard_min" scorefxn="sfxn" chi="1" bb="1" jump="1" tolerance="0.01" cartesian="false" task_operations="restrict_target_not_interface,restrict_target2repacking" />

        <FastRelax name="FastRelax" scorefxn="sfxn_relax" repeats="1" batch="false" ramp_down_constraints="false" cartesian="false" bondangle="false" bondlength="false" min_type="dfpmin_armijo_nonmonotone" task_operations="current,ex1_ex2,restrict_target_not_interface,limitchi2" relaxscript="InterfaceRelax2019">
            <MoveMap name="MM" >
                <Chain number="1" chi="true" bb="true" />
                <Chain number="2" chi="true" bb="false" />
                <Jump number="1" setting="true" />
            </MoveMap>
        </FastRelax>

        <Idealize name="idealize" atom_pair_constraint_weight="0.05" coordinate_constraint_weight="0.01" fast="false" report_CA_rmsd="true" impose_constraints="true" constraints_only="false"/>


    </MOVERS>
    <APPLY_TO_POSE>
    </APPLY_TO_POSE>
    <PROTOCOLS>
    </PROTOCOLS>
    <OUTPUT />
</ROSETTASCRIPTS>
"""

xml = xml.replace('<Slice name="patchdock_res" indices="%%patchdock_res%%" selector="chainB" />', f'<Slice name="patchdock_res" indices="{args.patchdock_res}" selector="chainB" />')
objs = protocols.rosetta_scripts.XmlObjects.create_from_string(xml)

# Load the movers we will need

FastRelax = objs.get_mover( 'FastRelax' )
remove_massive_clashes = objs.get_mover( 'remove_massive_clashes' )
pack_binder = objs.get_mover( 'pack_binder' )
remove_helix_gly = objs.get_mover( 'remove_helix_gly' )
pack_no_design = objs.get_mover( 'pack_no_design' )
softish_min = objs.get_mover( 'softish_min' )
hard_min = objs.get_mover( 'hard_min' )
idealize = objs.get_mover('idealize')
load_from_lite = objs.get_mover( 'load_from_lite' )
to_fa_standard = protocols.simple_moves.SwitchResidueTypeSetMover( 'fa_standard' )

ddg_filter = objs.get_filter( 'ddg' )
if ( isinstance(ddg_filter, pyrosetta.rosetta.protocols.filters.StochasticFilter) ):
    ddg_filter = ddg_filter.subfilter()

contact_patch_filter = objs.get_filter( 'contact_patch' )
if ( isinstance(contact_patch_filter, pyrosetta.rosetta.protocols.filters.StochasticFilter) ):
    contact_patch_filter = contact_patch_filter.subfilter()

cms_filter = objs.get_filter( 'contact_molecular_surface' )
if ( isinstance(cms_filter, pyrosetta.rosetta.protocols.filters.StochasticFilter) ):
    cms_filter = cms_filter.subfilter()

silent_out = f"{args.out_name}.silent"

sfxn = core.scoring.ScoreFunctionFactory.create_score_function("beta_nov16")


xml2= """<ROSETTASCRIPTS>
    <SCOREFXNS>
        <ScoreFunction name="sfxn_design" weights="beta_nov16" symmetric="1">
            <Reweight scoretype="netcharge" weight="1.0" />
            <Reweight scoretype="res_type_constraint" weight="0.5" />
            <Reweight scoretype="approximate_buried_unsat_penalty" weight="5.0" />
            <Set approximate_buried_unsat_penalty_natural_corrections1="true" />
            <Reweight scoretype="sap_constraint" weight="1.0" />
        </ScoreFunction>
    </SCOREFXNS>
    <RESIDUE_SELECTORS>
        <Chain name="chainA" chains="A"/>
        <Chain name="chainB" chains="B"/>
        <Neighborhood name="interface_chA" selector="chainB" distance="8.0" />
        <Neighborhood name="interface_chB" selector="chainA" distance="8.0" />
        <And name="AB_interface" selectors="interface_chA,interface_chB" />
        <Not name="Not_interface" selector="AB_interface" />
        <And name="actual_interface_chB" selectors="AB_interface,chainB" />
        <And name="not_interface_chB" selectors="Not_interface,chainB" />

        <Index name="bad_sap" resnums="9999999"/>
        <Layer name="surface" select_core="false" select_boundary="false" select_surface="true" use_sidechain_neighbors="true"/>
        <Layer name="boundary" select_core="false" select_boundary="true" select_surface="false" use_sidechain_neighbors="true"/>
        <Layer name="core" select_core="true" select_boundary="false" select_surface="false" use_sidechain_neighbors="true"/>
        <Layer name="surface_sasa" select_core="false" select_boundary="false" select_surface="true"
            ball_radius="2.0" use_sidechain_neighbors="false" core_cutoff="20.0" surface_cutoff="40.0" />
        <Layer name="core_sasa" select_core="true" select_boundary="false" select_surface="false"
            ball_radius="2.0" use_sidechain_neighbors="false" core_cutoff="20.0" surface_cutoff="40.0" />

        <Or name="surface_all" selectors="surface,surface_sasa"/>
        <Not name="not_surface" selector="surface_all"/>

        <ResidueName name="ala_positions" residue_name3="ALA" />
        <And name="surface_ala" selectors="surface_all,ala_positions"/>
        <Or name="core_all" selectors="core,core_sasa"/>

        <ResidueName name="pro_and_gly_positions" residue_name3="PRO,GLY" />

        <Or name="stored_bad_sap_plus_ala" selectors="surface_ala,bad_sap"/>
        <Not name="not_stored_bad_sap" selector="stored_bad_sap_plus_ala"/>
        <True name="true_sel" />
    </RESIDUE_SELECTORS>
    <RESIDUE_SELECTORS>
        <!-- Layer Design -->
        <SecondaryStructure name="sheet" overlap="0" minH="3" minE="2" include_terminal_loops="false" use_dssp="true" ss="E"/>
        <SecondaryStructure name="entire_loop" overlap="0" minH="3" minE="2" include_terminal_loops="true" use_dssp="true" ss="L"/>
        <SecondaryStructure name="entire_helix" overlap="0" minH="3" minE="2" include_terminal_loops="false" use_dssp="true" ss="H"/>
        <And name="helix_cap" selectors="entire_loop">
            <PrimarySequenceNeighborhood lower="1" upper="0" selector="entire_helix"/>
        </And>
        <And name="helix_start" selectors="entire_helix">
            <PrimarySequenceNeighborhood lower="0" upper="1" selector="helix_cap"/>
        </And>
        <And name="helix" selectors="entire_helix">
            <Not selector="helix_start"/>
        </And>
        <And name="loop" selectors="entire_loop">
            <Not selector="helix_cap"/>
        </And>
    </RESIDUE_SELECTORS>

    <TASKOPERATIONS>
        <DesignRestrictions name="layer_design">
        <Action selector_logic="surface AND helix_start"  aas="DEHKPQR"/>
        <Action selector_logic="surface AND helix"        aas="EHKQR"/>
        <Action selector_logic="surface AND sheet"        aas="EHKNQRST"/>
        <Action selector_logic="surface AND loop"         aas="DEGHKNPQRST"/>
        <Action selector_logic="boundary AND helix_start" aas="ADEFHIKLNPQRSTVWY"/>
        <Action selector_logic="boundary AND helix"       aas="ADEFHIKLNQRSTVWY"/>
        <Action selector_logic="boundary AND sheet"       aas="ADEFHIKLNQRSTVWY"/>
        <Action selector_logic="boundary AND loop"        aas="ADEFGHIKLNPQRSTVWY"/>
        <Action selector_logic="core AND helix_start"     aas="AFILPVWYDNSTH"/>
        <Action selector_logic="core AND helix"           aas="AFILVWYDNSTH"/>
        <Action selector_logic="core AND sheet"           aas="AFILVWYDNSTH"/>
        <Action selector_logic="core AND loop"            aas="AFGILPVWYDNSTH"/>
        <Action selector_logic="helix_cap"                aas="DGNPST"/>
        </DesignRestrictions>
    </TASKOPERATIONS>

    <TASKOPERATIONS>
        <LimitAromaChi2 name="limitchi2" chi2max="110" chi2min="70" include_trp="True" />
        <ExtraRotamersGeneric name="ex1_ex2" ex1="1" ex2aro="1" />

        <DisallowIfNonnative name="disallow_GLY" resnum="0" disallow_aas="G" />
        <DisallowIfNonnative name="disallow_PRO" resnum="0" disallow_aas="P" />

        <OperateOnResidueSubset name="restrict_target_not_interface" selector="not_interface_chB">
            <PreventRepackingRLT/>
        </OperateOnResidueSubset>
        <OperateOnResidueSubset name="restrict_target2repacking" selector="chainB">
            <RestrictToRepackingRLT/>
        </OperateOnResidueSubset>

        <OperateOnResidueSubset name="restrict_PRO_GLY" selector="pro_and_gly_positions">
            <RestrictToRepackingRLT/>
        </OperateOnResidueSubset>
        <OperateOnResidueSubset name="only_surface" selector="not_surface">
            <RestrictToRepackingRLT/>
        </OperateOnResidueSubset>
        <OperateOnResidueSubset name="not_core" selector="core_all">
            <PreventRepackingRLT/>
        </OperateOnResidueSubset>

        <OperateOnResidueSubset name="restrict_not_bad_sap" selector="not_stored_bad_sap">
            <RestrictToRepackingRLT/>
        </OperateOnResidueSubset>
    </TASKOPERATIONS>
    <SIMPLE_METRICS>
        <PerResidueSapScoreMetric name="my_per_res_sap" />
    </SIMPLE_METRICS>
    <MOVERS>
        <AddSapConstraintMover name="add_sap" speed="lightning" sap_goal="0" penalty_per_sap="0.5" score_selector="true_sel" sap_calculate_selector="true_sel" /> #sasa_selector="true_sel"
        <PackRotamersMover name="redesign_bad_sap" scorefxn="sfxn_design" 
            task_operations="limitchi2,ex1_ex2,only_surface,disallow_GLY,disallow_PRO,not_core,restrict_PRO_GLY,layer_design,restrict_not_bad_sap,restrict_target_not_interface,restrict_target2repacking"/> 
        <AddNetChargeConstraintMover name="netcharge_cst" filename="/home/drhicks1/scripts/mpnn/mpnn_design_scripts/netcharge" selector="chainA" />
    </MOVERS>
    <PROTOCOLS>
        <Add mover="add_sap" />
        <Add mover="netcharge_cst" />
        <Add mover="redesign_bad_sap" />
    </PROTOCOLS>
    <OUTPUT />
</ROSETTASCRIPTS>
"""

objs2 = protocols.rosetta_scripts.XmlObjects.create_from_string(xml2)

my_per_res_sap = objs2.get_simple_metric("my_per_res_sap")

#load movers

# PDB Parse Util Functions

alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
states = len(alpha_1)
alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
           'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']

aa_1_N = {a:n for n,a in enumerate(alpha_1)}
aa_3_N = {a:n for n,a in enumerate(alpha_3)}
aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}


def AA_to_N(x):
  # ["ARND"] -> [[0,1,2,3]]
  x = np.array(x);
  if x.ndim == 0: x = x[None]
  return [[aa_1_N.get(a, states-1) for a in y] for y in x]

def N_to_AA(x):
  # [[0,1,2,3]] -> ["ARND"]
  x = np.array(x);
  if x.ndim == 1: x = x[None]
  return ["".join([aa_N_1.get(a,"-") for a in y]) for y in x]

# End PDB Parse Util Functions

def thread_mpnn_seq( pose, binder_seq ):
    rsd_set = pose.residue_type_set_for_pose( core.chemical.FULL_ATOM_t )

    for resi, mut_to in enumerate( binder_seq ):
        resi += 1 # 1 indexing
        name3 = aa_1_3[ mut_to ]
        new_res = core.conformation.ResidueFactory.create_residue( rsd_set.name_map( name3 ) )
        pose.replace_residue( resi, new_res, True )
    
    return pose

def get_final_dict(score_dict, string_dict):
    print(score_dict)
    final_dict = OrderedDict()
    keys_score = [] if score_dict is None else list(score_dict)
    keys_string = [] if string_dict is None else list(string_dict)

    all_keys = keys_score + keys_string

    argsort = sorted(range(len(all_keys)), key=lambda x: all_keys[x])

    for idx in argsort:
        key = all_keys[idx]

        if ( idx < len(keys_score) ):
            final_dict[key] = "%8.3f"%(score_dict[key])
        else:
            final_dict[key] = string_dict[key]

    return final_dict
    
def add2scorefile(tag, scorefilename, score_dict=None):
  
    write_header = not os.path.isfile(scorefilename)
    with open(scorefilename, "a") as f:
        add_to_score_file_open(tag, f, write_header, score_dict)

def add_to_score_file_open(tag, f, write_header=False, score_dict=None, string_dict=None):
    final_dict = get_final_dict( score_dict, string_dict )
    if ( write_header ):
        f.write("SCORE:     %s description\n"%(" ".join(final_dict.keys())))
    scores_string = " ".join(final_dict.values())
    f.write("SCORE:     %s        %s\n"%(scores_string, tag))

def generate_mut_string(seq):
    return 'MUT:' + '_'.join( [ f"{idx+1}.{aa_1_3[aa1]}" for idx, aa1 in enumerate(seq) ] )

def swap_mut_string(tag, mut_string, og_struct):
    outlines = []
    for line in og_struct:
        line = line.strip()
        if not 'PDBinfo-LABEL:' in line:
            # Swap out tags on all lines except for the remark line
            splits = line.split()
            if len(splits) == 0:
                outlines.append( '' )
                continue
            outline = splits[:-1]
            outline.append(tag)
            outlines.append( ' '.join(outline) )
            continue

        splits = line.split(' ')
        outsplits = []
        mut_found = False

        for split in splits:
            if not split.startswith('MUT:'):
                outsplits.append(split)
                continue
            mut_found = True
            outsplits.append(mut_string)

        if not mut_found: outsplits.append(mut_string)

        outlines.append( ' '.join(outsplits) )

    return '\n'.join(outlines)

def add2silent( pose, tag, sfd_out ):
    struct = sfd_out.create_SilentStructOP()
    struct.fill_struct( pose, tag )
    sfd_out.add_structure( struct )
    sfd_out.write_silent_struct( struct, silent_out )

def add2lite( seq, tag, og_struct ):
    
    mut_string = generate_mut_string(seq)
    new_struct = swap_mut_string(tag, mut_string, og_struct)
    with open( silent_out, 'a' ) as f: f.write( f"{new_struct}\n" )

def optimize_sap(pose):
    sap_dict = my_per_res_sap.calculate(pose)
    bad_sap = [str(x) for x in sap_dict if sap_dict[x] > 1.4]

    if len(bad_sap) == 0 : return pose

    newxml = xml2.replace("9999999", ",".join(bad_sap))

    objs3 = protocols.rosetta_scripts.XmlObjects.create_from_string(newxml)

    add_sap = objs3.get_mover( 'add_sap' )
    netcharge_cst = objs3.get_mover( 'netcharge_cst' )
    redesign_bad_sap = objs3.get_mover( 'redesign_bad_sap' )

    add_sap.apply(pose)
    netcharge_cst.apply(pose)
    redesign_bad_sap.apply(pose)

    return pose

def select_surface(pose):
    surface_residues = []
    shallow_atoms = pyrosetta.rosetta.core.scoring.atomic_depth.atoms_deeper_than(pose, 2.0, True, 3.5, False, 0.25)
    for i in range(1, pose.size()+1):
        resi = pose.residue(i)
        shallow_bools = shallow_atoms(i)
        for atom_i in range(1, resi.natoms()+1):
            if resi.atom_is_backbone(atom_i):
                continue
            if resi.atom_is_hydrogen(atom_i):
                continue
            if shallow_bools[atom_i]:
                surface_residues.append(i)
                break
    return surface_residues

def parse_PDB_biounits(x, atoms=['N','CA','C'], chain=None):
  '''
  input:  x = PDB filename
          atoms = atoms to extract (optional)
  output: (length, atoms, coords=(x,y,z)), sequence
  '''
  xyz,seq,min_resn,max_resn = {},{},1e6,-1e6
  for line in open(x,"rb"):
    line = line.decode("utf-8","ignore").rstrip()

    if line[:6] == "HETATM" and line[17:17+3] == "MSE":
      line = line.replace("HETATM","ATOM  ")
      line = line.replace("MSE","MET")

    if line[:4] == "ATOM":
      ch = line[21:22]
      if ch == chain or chain is None:
        atom = line[12:12+4].strip()
        resi = line[17:17+3]
        resn = line[22:22+5].strip()
        x,y,z = [float(line[i:(i+8)]) for i in [30,38,46]]

        if resn[-1].isalpha(): 
            resa,resn = resn[-1],int(resn[:-1])-1
        else: 
            resa,resn = "",int(resn)-1
#         resn = int(resn)
        if resn < min_resn: 
            min_resn = resn
        if resn > max_resn: 
            max_resn = resn
        if resn not in xyz: 
            xyz[resn] = {}
        if resa not in xyz[resn]: 
            xyz[resn][resa] = {}
        if resn not in seq: 
            seq[resn] = {}
        if resa not in seq[resn]: 
            seq[resn][resa] = resi

        if atom not in xyz[resn][resa]:
          xyz[resn][resa][atom] = np.array([x,y,z])

  # convert to numpy arrays, fill in missing values
  seq_,xyz_ = [],[]
  try:
      for resn in range(min_resn,max_resn+1):
        if resn in seq:
          for k in sorted(seq[resn]): seq_.append(aa_3_N.get(seq[resn][k],20))
        else: seq_.append(20)
        if resn in xyz:
          for k in sorted(xyz[resn]):
            for atom in atoms:
              if atom in xyz[resn][k]: xyz_.append(xyz[resn][k][atom])
              else: xyz_.append(np.full(3,np.nan))
        else:
          for atom in atoms: xyz_.append(np.full(3,np.nan))
      return np.array(xyz_).reshape(-1,len(atoms),3), N_to_AA(np.array(seq_))
  except TypeError:
      return 'no_chain', 'no_chain'

def generate_seqopt_features( pdbfile ):
    atoms = ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
        'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
        'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
        'CZ3', 'NZ']
    my_dict = {}
    s = 0
    concat_seq = ''
    concat_N = []
    concat_CA = []
    concat_C = []
    concat_O = []
    concat_mask = []
    coords_dict = {}
    for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']:
        xyz, seq = parse_PDB_biounits(pdbfile, atoms=atoms, chain=letter)
        if type(xyz) != str:
            concat_seq += seq[0]
            my_dict['seq_chain_'+letter]=seq[0]
            coords_dict_chain = {}
            coords_dict_chain['all_atoms_chain_'+letter]=xyz.tolist()
            my_dict['coords_chain_'+letter]=coords_dict_chain
            s += 1
    fi = pdbfile.rfind("/")
    my_dict['name']=pdbfile[(fi+1):-4]
    my_dict['num_of_chains'] = s
    my_dict['seq'] = concat_seq

    return my_dict

def chain_mask(in_dict):
    pre_chain_list = args.design_these_chains
    a_list = pre_chain_list.split(",")
    b_list = [str(i) for i in a_list]
    masked_chain_list = b_list
    visible_chain_list_pre = args.keep_these_chains
    if args.keep_these_chains == 'NA':
        visible_chain_list = []
    else:
        c_list = visible_chain_list_pre.split(",")
        d_list = [str(i) for i in c_list]
        visible_chain_list = d_list

    masked_dict = {}
    all_chain_list = [item[-1:] for item in list(in_dict) if item[:9]=='seq_chain'] #['A','B', 'C',...]
    masked_dict[in_dict['name']]= (masked_chain_list, visible_chain_list)
    
    return masked_dict

def tied_positions(in_dict):
    tie_chainsb_list=[]
    if args.tie_chainsb not in ['NA']:
        tie_chains_pre = args.tie_chainsb
        g_list = tie_chains_pre.split(",")
        tie_chainsb_list = [int(i) for i in g_list]
        

    if args.tie_chainsa not in ['NA']:
        tie_chains_prea = args.tie_chainsa
        t_list = tie_chains_prea.split(",")
        tie_chains_list = [int(i) for i in t_list]

        tied_dict = {}
        all_chain_list = sorted([item[-1:] for item in list(in_dict) if item[:9]=='seq_chain'])
        tied_positions_list = []
        first_chain_num = tie_chains_list[0]
        first_letter = all_chain_list[first_chain_num]
        chain_len_to_grab = f"seq_chain_{first_letter}"
        for i in range(1,len(in_dict[chain_len_to_grab])+1):
            temp_dict = {}
            for abc in tie_chains_list:
                temp_dict[all_chain_list[abc]] = [i]
            tied_positions_list.append(temp_dict)

        if tie_chainsb_list != []:
            first_chain_num_b = tie_chainsb_list[0]
            first_letter_b = all_chain_list[first_chain_num_b]
            chainb_len_to_grab = f"seq_chain_{first_letter_b}"
            for i in range(1,len(in_dict[chainb_len_to_grab])+1):
                temp_dict = {}
                for bc in tie_chainsb_list:
                    temp_dict[all_chain_list[bc]] = [i]
                tied_positions_list.append(temp_dict) 
        tied_dict[in_dict['name']] = tied_positions_list
        #print(tied_dict)

        return tied_dict

    elif int(args.tie_repeats) != 0:
        print('This is a repeat protein/toroid')
        num_reps = int(args.tie_repeats)
        tied_dict = {}
        all_chain_list = sorted([item[-1:] for item in list(in_dict) if item[:9]=='seq_chain'])
        tied_positions_list = []
        input_size = len(in_dict['seq'])
        repeat_size = int(input_size / num_reps)
        for i in range(1,repeat_size+1):
            temp_dict = {}
            temp_dict[all_chain_list[0]] = [i, i+repeat_size, i+repeat_size+repeat_size,i+repeat_size+repeat_size+repeat_size,i+repeat_size+repeat_size+repeat_size+repeat_size,i+repeat_size+repeat_size+repeat_size+repeat_size+repeat_size,i+repeat_size+repeat_size+repeat_size+repeat_size+repeat_size+repeat_size,i+repeat_size+repeat_size+repeat_size+repeat_size+repeat_size+repeat_size+repeat_size]
            res = dict()
            for sub in temp_dict:
                res[sub] = temp_dict[sub][:num_reps]
            tied_positions_list.append(res) 
        tied_dict[in_dict['name']] = tied_positions_list
        
        return tied_dict
    else:
        return None

def fixed_positions(in_dict):
    fixed_dict = {}
    if args.fix_a not in ['NA']:
        fixa = args.fix_a
        fixlista = fixa.split(",")
        a_list = [int(i) for i in fixlista]
        all_chain_list = [item[-1:] for item in list(in_dict) if item[:9]=='seq_chain']
        fixed_position_dict = {}
        for chain in all_chain_list:
            if chain == 'A':
                fixed_position_dict[chain] = [a_list]
            elif chain == 'B' and args.fix_b not in ['NA']:
                fixb = args.fix_b
                fixlistb = fixb.split(",")
                b_list = [int(i) for i in fixlistb]
                fixed_position_dict[chain] = [b_list]
            else:
                fixed_position_dict[chain] = []
        
        fixed_dict[in_dict['name']] = fixed_position_dict

        return fixed_dict
    else:
        return None


def sequence_optimize( pdbfile):
    
    t0 = time.time()

    feature_dict = generate_seqopt_features( pdbfile )
    tmp_pose = pose_from_pdb(pdbfile)
    
    chain_A_seq = "X" + tmp_pose.split_by_chain()[1].sequence()
    chain_A_dssp = "X" + pyrosetta.rosetta.core.scoring.dssp.Dssp(tmp_pose.split_by_chain()[1]).get_dssp_secstruct()

    print(f"feature_dict['name'] {feature_dict['name']}")

    chain_mask_dict = chain_mask(feature_dict)
    fixed_positions_dict=fixed_positions(feature_dict)
    tied_positions_dict=tied_positions(feature_dict)

    #add on fly
    args.tied_positions_jsonl = tied_positions_dict
    args.fixed_positions_jsonl = fixed_positions_dict
    args.chain_id_jsonl = chain_mask_dict
    args.jsonl_path = feature_dict
    
    #Add code to only allow polars at non loop surface res
    #Add code to only allow loop res at loop positions
    all_aas = list("ARNDCQEGHILKMFPSTWYV")

    polars = "DEHKNQRSTP" #allow P for N-helix Pro
    hydrophobics = "ACFGILMVWY"
    loop_aa = [x for x in args.loop_aas]
    loops_omit = [x for x in all_aas if x not in loop_aa]
    hydrophobics = [x for x in hydrophobics]

    loops = [i for i in list(range(1, len(chain_A_dssp))) if chain_A_dssp[i] == "L"]

    surface_res = list(set(select_surface(tmp_pose)))
    surface_res = [x for x in surface_res if x in list(range(1, len(chain_A_seq)))]
    surface_res = [x for x in surface_res if x not in loops]

    omit_res_list = [[[x], ["C"]] for x in list(range(tmp_pose.chain_begin(1), tmp_pose.chain_end(1)+1))]

    for surface_resi in surface_res:
        omit_res_list[surface_resi-1] = [omit_res_list[surface_resi-1][0], omit_res_list[surface_resi-1][1] + hydrophobics]
    for loop_i in loops:
        omit_res_list[loop_i-1] = [omit_res_list[loop_i-1][0], omit_res_list[loop_i-1][1] + loops_omit]
    for i in range(len(omit_res_list)):
        omit_res_list[i] = [omit_res_list[i][0], list(set(omit_res_list[i][1]))]

    print(omit_res_list)
    omit_AA_jsonl = {}
    omit_AA_jsonl[feature_dict['name']] = {}
    omit_AA_jsonl[feature_dict['name']]["A"] = omit_res_list

    #print(omit_AA_jsonl[feature_dict['name']]["A"])

    args.omit_AA_jsonl = omit_AA_jsonl

    args.pdb_path = pdbfile
    sequences = mpnn_util.bb_mpnn(args)
    
    os.remove( pdbfile )

    print( f"MPNN generated {len(sequences)} sequences in {int( time.time() - t0 )} seconds" ) 
    
    sequences.sort(key=lambda x: float(x[1]))

    print("seq:score")
    for seq in sequences:
        print(f"{seq[0]}:{seq[1]}")

    sequences = [sequences[0][0]]

    # unique_seqs = []
    # for sequence in sequences:
    #     if sequence[0] in unique_seqs: 
    #         continue
    #     else:
    #         unique_seqs.append(sequence[0])
    
    # sequences = unique_seqs

    masked_list = []
    masked_chain_length_list = []
    for letter in chain_mask_dict[feature_dict['name']][0]:
        masked_list.append(letter)
        chain_seq = feature_dict[f'seq_chain_{letter}']
        chain_length = len(chain_seq)
        masked_chain_length_list.append(chain_length)
    
    split_sequences = []
    if len(args.design_these_chains.split(",")) > 1:
        for seq in sequences:
            l0 = 0
            for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                l0 += mc_length
                seq = seq[:l0] + '/' + seq[l0:]
                l0 += 1
            split_sequences.append(seq)

    else:
        split_sequences = sequences


    # for seq in split_sequences:
    #    print(seq)
    
    return split_sequences


def relax_pose( pose ):
    remove_massive_clashes.apply( pose )
    remove_helix_gly.apply(pose)
    pose = optimize_sap(pose)
    FastRelax.apply( pose )

    return pose

def fa_pose_from_str(pdb_str):
    pose = pyrosetta.Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose, pdb_str)
    return pose

def backbone_rmsd( move_pose, to_pose, atoms=["N", "CA", "C"] ):

    move_res = np.array(list(range(1, move_pose.size()+1)))

    to_res = np.array(list(range(1, to_pose.size()+1)))

    move_to_pairs = []
    coords_move = utility.vector1_numeric_xyzVector_double_t()
    coords_to = utility.vector1_numeric_xyzVector_double_t()

    for i in range(len(move_res)):
        seqpos_move = move_res[i]
        seqpos_to = to_res[i]

        move_to_pairs.append((seqpos_move, seqpos_to))

        for atom in atoms:
            coords_move.append(move_pose.residue(seqpos_move).xyz(atom))
            coords_to.append(to_pose.residue(seqpos_to).xyz(atom))


    move_pose_copy = move_pose.clone()


    backbone_rmsd = 0

    backbone_distances = []

    if ( len(move_to_pairs) > 0 ):

        rotation_matrix = numeric.xyzMatrix_double_t()
        move_com = numeric.xyzVector_double_t()
        ref_com = numeric.xyzVector_double_t()

        protocols.toolbox.superposition_transform( coords_move, coords_to, rotation_matrix, move_com, ref_com )

        protocols.toolbox.apply_superposition_transform( move_pose, rotation_matrix, move_com, ref_com )

        for seqpos_move, seqpos_to in move_to_pairs:
            for atom in atoms:
                backbone_distance = move_pose.residue(seqpos_move).xyz(atom).distance_squared(to_pose.residue(seqpos_to).xyz(atom))
                backbone_rmsd += backbone_distance
                backbone_distances.append(backbone_distance)

        backbone_rmsd /= len(backbone_distances)
        backbone_rmsd = np.sqrt(backbone_rmsd)
    
    print(f"backbone rmsd for residues {move_res}: {backbone_rmsd}")
    return backbone_rmsd

def pair_RMSD( pose, other_pose):

    if 1:
    #try:
        bb_rmsd = backbone_rmsd( pose, other_pose)
        return bb_rmsd
    if not 1:
    #except:
        print("rmsd problem; likley colinear atoms")
        return 999


def dl_design( pose, tag, og_struct, sfd_out ):

    design_counter = 0

    prefix = f"{tag}_dldesign"
    pdbfile = f"{prefix}.pdb"
    
    try:
        pose.dump_pdb( pdbfile )
    except:
        remove_massive_clashes(pose)
        pose.dump_pdb( pdbfile )

    seqs = sequence_optimize( pdbfile )

    for idx,seq in enumerate( seqs ):
        pose = thread_mpnn_seq( pose, seq )

        pose = relax_pose(pose)

        pose.dump_pdb( pdbfile )
        seqs2 = sequence_optimize( pdbfile )
        
        for idx2,seq2 in enumerate( seqs2 ):
            tag = f"{prefix}_{idx}_{idx2}"
            pose = thread_mpnn_seq( pose, seq2 )
            
            pose = relax_pose(pose)

            ddg = ddg_filter.compute(pose)

            if ddg > args.ddg_cutoff: continue
            
            try:
                contact_patch = contact_patch_filter.compute(pose)
            except:
                contact_patch = 0

            contact_molecular_surface = cms_filter.compute(pose)
            
            # if lite: add2lite( seq, tag, og_struct )
            # else: add2silent( pose, tag, sfd_out )

            add2silent( pose, tag, sfd_out )

            add2scorefile( tag, f"{args.out_name}.sc", {'ddg':ddg, 'contact_patch':contact_patch, 'contact_molecular_surface':contact_molecular_surface} )
            design_counter += 1

    return design_counter

def main( pdb, silent_structure, sfd_in, sfd_out ):

    t0 = time.time()
    print( "Attempting pose: %s"%pdb )
    
    # Load pose from PDBLite
    pose = Pose()
    sfd_in.get_structure( pdb ).fill_pose( pose )
    if lite:
        load_from_lite.apply( pose )
        to_fa_standard.apply( pose )

    if args.idealize_A:
        poseA = pose.split_by_chain()[1]
        poseB = pose.split_by_chain()[2]
        idealize.apply(poseA)
        poseA.append_pose_by_jump(poseB, 1)
        pose = poseA

    dl_design( pose, pdb, silent_structure, sfd_out )

    seconds = int(time.time() - t0)

    print( f"protocols.jd2.JobDistributor: {pdb} reported success. designs generated in {seconds} seconds" )

def detect_lite(silentfile, silent_index):
    struct = silent_tools.get_silent_structure( silentfile, silent_index, silent_index['tags'][0] )
    for line in struct:
        if 'SCAFFOLD_PDB' in line: return True
    return False

# Checkpointing Functions

def record_checkpoint( pdb, checkpoint_filename ):
    with open( checkpoint_filename, 'a' ) as f:
        f.write( pdb )
        f.write( '\n' )

def determine_finished_structs( checkpoint_filename ):
    done_set = set()
    if not os.path.isfile( checkpoint_filename ): return done_set

    with open( checkpoint_filename, 'r' ) as f:
        for line in f:
            done_set.add( line.strip() )

    return done_set

# End Checkpointing Functions

#################################
# Begin Main Loop
#################################

silent_index = silent_tools.get_silent_index(silent)
lite = detect_lite(silent, silent_index) # using as a global variable

if not os.path.isfile(silent_out):
    with open(silent_out, 'w') as f: f.write(silent_tools.silent_header(silent_index))

#sfd_out = None
#if not lite: sfd_out = core.io.silent.SilentFileData(silent_out, False, False, "binary", core.io.silent.SilentFileOptions())
sfd_out = core.io.silent.SilentFileData(silent_out, False, False, "binary", core.io.silent.SilentFileOptions())

sfd_in = rosetta.core.io.silent.SilentFileData(rosetta.core.io.silent.SilentFileOptions())
sfd_in.read_file(silent)

checkpoint_filename = "check.point"
#debug = False
debug = True

finished_structs = determine_finished_structs( checkpoint_filename )

for pdb in silent_index['tags']:

    if pdb in finished_structs: continue

    silent_structure = silent_tools.get_silent_structure( silent, silent_index, pdb )

    if debug: main( pdb, silent_structure, sfd_in, sfd_out )

    else: # When not in debug mode the script will continue to run even when some poses fail
        t0 = time.time()

        try: main( pdb, silent_structure, sfd_in, sfd_out )

        except KeyboardInterrupt: sys.exit( "Script killed by Control+C, exiting" )

        except:
            seconds = int(time.time() - t0)
            print( "protocols.jd2.JobDistributor: %s failed in %i seconds with error: %s"%( pdb, seconds, sys.exc_info()[0] ) )
            continue

    # We are done with one pdb, record that we finished
    record_checkpoint( pdb, checkpoint_filename )
