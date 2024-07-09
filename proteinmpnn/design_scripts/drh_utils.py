import sys
import os
import json
from itertools import groupby
from collections import OrderedDict

import numpy as np
import pyrosetta
from pyrosetta import pose_from_pdb, get_fa_scorefxn, rosetta
import random
import requests
from Bio import pairwise2

alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
states = len(alpha_1)
alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
           'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']

aa_1_N = {a:n for n,a in enumerate(alpha_1)}
aa_3_N = {a:n for n,a in enumerate(alpha_3)}
aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}

def N_to_AA(x):
  # [[0,1,2,3]] -> ["ARND"]
  x = np.array(x);x
  if x.ndim == 1: x = x[None]
  return ["".join([aa_N_1.get(a,"-") for a in y]) for y in x]


xml = """<ROSETTASCRIPTS>
    <SCOREFXNS>
        <ScoreFunction name="sfxn" weights="beta_nov16" >
        </ScoreFunction>
        <ScoreFunction name="sfxn_soft" weights="beta_nov16_soft" >
        </ScoreFunction>
        <ScoreFunction name="sfxn_softish" weights="beta_nov16" >
            <Reweight scoretype="fa_rep" weight="0.15" />
        </ScoreFunction>
        <ScoreFunction name="sfxn_design" weights="beta_nov16">
            <Reweight scoretype="netcharge" weight="1.0" />
            <Reweight scoretype="res_type_constraint" weight="0.5" />
            <Reweight scoretype="approximate_buried_unsat_penalty" weight="5.0" />
            <Set approximate_buried_unsat_penalty_natural_corrections1="true" />
            <Reweight scoretype="sap_constraint" weight="1.0" />
        </ScoreFunction>

    </SCOREFXNS>
    <RESIDUE_SELECTORS>
        <Chain name="chainA" chains="1"/>
        <Chain name="chainB" chains="2"/>
        <Neighborhood name="interface_chA" selector="chainB" distance="10.0" />
        <Neighborhood name="interface_chB" selector="chainA" distance="10.0" />
        <And name="AB_interface" selectors="interface_chA,interface_chB" />
        <Not name="Not_interface" selector="AB_interface" />
        <And name="actual_interface_chB" selectors="AB_interface,chainB" />
        <And name="not_interface_chB" selectors="Not_interface,chainB" />
        <True name="all" />
        <Slice name="patchdock_res" indices="%%patchdock_res%%" selector="chainB" />
        <Index name="bad_sap" resnums="9999999"/>
    </RESIDUE_SELECTORS>

    <RESIDUE_SELECTORS>
        <!-- Layer Design -->
        <Layer name="surface" select_core="false" select_boundary="false" select_surface="true" use_sidechain_neighbors="true"/>
        <Layer name="boundary" select_core="false" select_boundary="true" select_surface="false" use_sidechain_neighbors="true"/>
        <Layer name="core" select_core="true" select_boundary="false" select_surface="false" use_sidechain_neighbors="true"/>
        <SecondaryStructure name="sheet" overlap="0" minH="3" minE="2" include_terminal_loops="false" use_dssp="true" ss="E"/>
        <SecondaryStructure name="entire_loop" overlap="0" minH="3" minE="2" include_terminal_loops="true" use_dssp="true" ss="L"/>
        <SecondaryStructure name="entire_helix" overlap="0" minH="3" minE="2" include_terminal_loops="false" use_dssp="true" ss="H"/>
        <Layer name="surface_sasa" select_core="false" select_boundary="false" select_surface="true"
            ball_radius="2.0" use_sidechain_neighbors="false" core_cutoff="20.0" surface_cutoff="40.0" />
        <Layer name="core_sasa" select_core="true" select_boundary="false" select_surface="false"
            ball_radius="2.0" use_sidechain_neighbors="false" core_cutoff="20.0" surface_cutoff="40.0" />
        <Or name="core_all" selectors="core,core_sasa"/>
        <Or name="surface_all" selectors="surface,surface_sasa"/>
        <Not name="not_surface" selector="surface_all"/>
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
        <ResidueName name="pro_and_gly_positions" residue_name3="PRO,GLY" />
        <ResidueName name="ala_positions" residue_name3="ALA" />
        <And name="surface_ala" selectors="surface_all,ala_positions"/>
        <Or name="stored_bad_sap_plus_ala" selectors="surface_ala,bad_sap"/>
        <Not name="not_stored_bad_sap" selector="stored_bad_sap_plus_ala"/>
        <True name="true_sel" />
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
        <SetIGType name="precompute_ig" lin_mem_ig="false" lazy_ig="false" double_lazy_ig="false" precompute_ig="true"/>
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
        <OperateOnResidueSubset name="only_surface" selector="not_surface">
            <RestrictToRepackingRLT/>
        </OperateOnResidueSubset>
        <DisallowIfNonnative name="disallow_GLY" resnum="0" disallow_aas="G" />
        <DisallowIfNonnative name="disallow_PRO" resnum="0" disallow_aas="P" />
        <OperateOnResidueSubset name="not_core" selector="core_all">
            <PreventRepackingRLT/>
        </OperateOnResidueSubset>
        <OperateOnResidueSubset name="restrict_PRO_GLY" selector="pro_and_gly_positions">
            <RestrictToRepackingRLT/>
        </OperateOnResidueSubset>
        <OperateOnResidueSubset name="restrict_not_bad_sap" selector="not_stored_bad_sap">
            <RestrictToRepackingRLT/>
        </OperateOnResidueSubset>
    </TASKOPERATIONS>
    <MOVERS>
        <TaskAwareMinMover name="min" scorefxn="sfxn" bb="0" chi="1" task_operations="pack_long" />
    </MOVERS>
    <FILTERS>
        <Ddg name="ddg" threshold="0" jump="1" repeats="3" repack="1" relax_mover="min" confidence="0" scorefxn="sfxn" extreme_value_removal="1" />
        <Ddg name="ddg_soft_norepack" threshold="0" jump="1" repeats="1" repack="0" confidence="0" scorefxn="sfxn_soft"/>
        <ContactMolecularSurface name="contact_molecular_surface" distance_weight="0.5" target_selector="chainA" binder_selector="chainB" confidence="0" />
        <ContactMolecularSurface name="contact_patch" distance_weight="0.5" target_selector="patchdock_res" binder_selector="chainA" confidence="0" />
    </FILTERS>
    <SIMPLE_METRICS>
        <PerResidueSapScoreMetric name="my_per_res_sap" />
    </SIMPLE_METRICS>
    <MOVERS>
        <PackRotamersMover name="remove_helix_gly" scorefxn="sfxn" task_operations="current,ex1_ex2,limitchi2,restrict_target_not_interface,restrict_target2repacking,layer_design_no_core_polars,only_helix_gly"/>
        <PackRotamersMover name="remove_massive_clashes" scorefxn="sfxn" task_operations="current,restrict2repacking,restrict_target_not_interface"/>
        <PackRotamersMover name="remove_massive_clashes_A" scorefxn="sfxn" task_operations="current,restrict2repacking"/>
        <TaskAwareMinMover name="softish_min" scorefxn="sfxn_softish" chi="1" bb="1" jump="1" tolerance="0.01" cartesian="false" task_operations="restrict_target_not_interface,restrict_target2repacking" />
        <TaskAwareMinMover name="hard_min" scorefxn="sfxn" chi="1" bb="1" jump="1" tolerance="0.01" cartesian="false" task_operations="restrict_target_not_interface,restrict_target2repacking" />
        <FastRelax name="FastRelax" scorefxn="sfxn" repeats="1" batch="false" ramp_down_constraints="false" cartesian="false" bondangle="false" bondlength="false" min_type="dfpmin_armijo_nonmonotone" task_operations="current,ex1_ex2,restrict_target_not_interface,limitchi2" relaxscript="InterfaceRelax2019">
            <MoveMap name="MM" >
                <Chain number="1" chi="true" bb="true" />
                <Chain number="2" chi="true" bb="false" />
                <Jump number="1" setting="true" />
            </MoveMap>
        </FastRelax>

        <AddSapConstraintMover name="add_sap" speed="lightning" sap_goal="0" penalty_per_sap="0.5" score_selector="true_sel" sap_calculate_selector="true_sel" />
        <PackRotamersMover name="redesign_bad_sap" scorefxn="sfxn_design"
            task_operations="precompute_ig,limitchi2,ex1_ex2,only_surface,disallow_GLY,disallow_PRO,not_core,restrict_PRO_GLY,layer_design_no_core_polars,restrict_not_bad_sap,restrict_target_not_interface,restrict_target2repacking"/> 
        <AddNetChargeConstraintMover name="netcharge_cst" filename="/home/drhicks1/scripts/proteinmpnn/design_scripts/netcharge" selector="true_sel" />
    </MOVERS>
    <APPLY_TO_POSE>
    </APPLY_TO_POSE>
    <PROTOCOLS>
    </PROTOCOLS>
    <OUTPUT />
</ROSETTASCRIPTS>
"""

def find_contacting_residues(pose, chain_A=1, chain_B=2, rep_distance_threshold=10.0, atom_distance_threshold=5.0):
    """
    Find all residues in chain A that are in atomic contact with chain B within given distance thresholds.
    """
    contacting_residues = set()
    
    def get_representative_atom(residue):
        if residue.has("CB"):
            return residue.atom_index("CB")
        else:
            return residue.atom_index("CA")

    for resA in range(pose.chain_begin(chain_A), pose.chain_end(chain_A) + 1):
        rep_atom_A = get_representative_atom(pose.residue(resA))
        for resB in range(pose.chain_begin(chain_B), pose.chain_end(chain_B) + 1):
            rep_atom_B = get_representative_atom(pose.residue(resB))
            dist = pose.residue(resA).xyz(rep_atom_A).distance(pose.residue(resB).xyz(rep_atom_B))
            if dist < rep_distance_threshold:
                for atomA in range(1, pose.residue(resA).natoms() + 1):
                    for atomB in range(1, pose.residue(resB).natoms() + 1):
                        atom_dist = pose.residue(resA).xyz(atomA).distance(pose.residue(resB).xyz(atomB))
                        if atom_dist < atom_distance_threshold:
                            contacting_residues.add(resA)
                            break
                    if resA in contacting_residues:
                        break

    return sorted(list(contacting_residues))

def calculate_ddg(pose, residues):
    """
    Calculate the ddg of mutation to alanine for each residue in the given list.
    """
    xml = """<ROSETTASCRIPTS>
        <SCOREFXNS>
            <ScoreFunction name="sfxn" weights="beta_nov16_soft" >
            </ScoreFunction>
        </SCOREFXNS>
        <FILTERS>
            <Ddg name="ddg" threshold="0" jump="1" repeats="1" repack="0" confidence="0" scorefxn="sfxn"/>
        </FILTERS>
        <PROTOCOLS>
        </PROTOCOLS>
    </ROSETTASCRIPTS>
    """
    objs = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(xml)
    ddg_filter = objs.get_filter("ddg")
    if isinstance(ddg_filter, pyrosetta.rosetta.protocols.filters.StochasticFilter):
        ddg_filter = ddg_filter.subfilter()

    original_ddg = ddg_filter.compute(pose)

    ddg_values = {}
    for res in residues:
        mut_pose = pose.clone()
        ala_pose = pyrosetta.pose_from_sequence("GAG")
        new_res = ala_pose.residue(2)
        mut_pose.replace_residue(res, new_res, True)
        mut_ddg = ddg_filter.compute(mut_pose)
        ddg =  original_ddg - mut_ddg
        ddg_values[res] = ddg

    print(f"ddg_values: {ddg_values}")
    return ddg_values

def biased_residue_sampling(ddg_values, temperature=4.0):
    """
    Randomly sample a number N from 30% to 70% of the number of residues,
    then randomly sample N residues from the set, biased by their ddg values.
    """
    residues = list(ddg_values.keys())
    num_residues = len(residues)
    N = random.randint(int(0.3 * num_residues), int(0.7 * num_residues))

    # Calculate Boltzmann probabilities
    energies = np.array([ddg_values[res] for res in residues])
    boltzmann_factors = np.exp(-energies / temperature)
    probabilities = boltzmann_factors / np.sum(boltzmann_factors)

    print(f"probabilities: {probabilities}")

    # Sample residues based on Boltzmann probabilities without replacement
    sampled_residues = np.random.choice(residues, size=N, replace=False, p=probabilities)
    return sorted(list(sampled_residues))

def get_random_interface_residues_by_ddg(pose, chain_A=1, chain_B=2, rep_distance_threshold=10.0, atom_distance_threshold=5.0):
    contacting_residues = find_contacting_residues(pose, chain_A, chain_B, rep_distance_threshold, atom_distance_threshold)
    ddg_values = calculate_ddg(pose, contacting_residues)
    sampled_residues = biased_residue_sampling(ddg_values)

    return sampled_residues


def get_sap(pose):
    sap = pyrosetta.rosetta.core.pack.guidance_scoreterms.sap.SapScoreMetric()
    return sap.calculate(pose)

def superposition_transform_with_weight(input_coords_move, input_coords_to, rotation_matrix, move_com, ref_com, atom_weight):

    coords_move = rosetta.utility.vector1_numeric_xyzVector_double_t()
    coords_to = rosetta.utility.vector1_numeric_xyzVector_double_t()

    assert(len(input_coords_move) == len(input_coords_to))
    assert(len(input_coords_move) == len(atom_weight))

    for i in range(len(input_coords_move)):
        for j in range(atom_weight[i]):
            coords_move.append(input_coords_move[i+1])
            coords_to.append(input_coords_to[i+1])

    rosetta.protocols.toolbox.superposition_transform( coords_move, coords_to, rotation_matrix, move_com, ref_com )

def pymol_align( move_pose, to_pose, sel_move=None, sel_to=None, atoms=["N", "CA", "C"], throw_away=0.3, rmsd_cut=0.25):

    if ( not sel_move is None ):
        move_res = np.array(list(rosetta.core.select.get_residues_from_subset(sel_move)))
    else:
        move_res = np.array(list(range(1, move_pose.size()+1)))

    if ( not sel_to is None ):
        to_res = np.array(list(rosetta.core.select.get_residues_from_subset(sel_to)))
    else:
        to_res = np.array(list(range(1, to_pose.size()+1)))

    seq_move = "x" + move_pose.sequence()
    seq_to = "x" + to_pose.sequence()

    seq_move = "".join(np.array(list(seq_move))[move_res])
    seq_to = "".join(np.array(list(seq_to))[to_res])

    align_move, align_to, idk1, idk2, idk3 = pairwise2.align.globalms(seq_move, seq_to, 2, -0.5, -0.1, -0.01)[0]
    print(align_move, align_to)

    all_move_to = []
    move_to_pairs = []
    all_coords_move = rosetta.utility.vector1_numeric_xyzVector_double_t()
    all_coords_to = rosetta.utility.vector1_numeric_xyzVector_double_t()
    all_atom_weight = []

    i_move = 0
    i_to = 0
    for i in range(len(align_move)):
        if ( align_move[i] == align_to[i] ):

            seqpos_move = move_res[i_move]
            seqpos_to = to_res[i_to]

            move_to_pairs.append((seqpos_move, seqpos_to))
            all_move_to.append((seqpos_move, seqpos_to))


            weight = 1

            for atom in atoms:
                all_coords_move.append(move_pose.residue(seqpos_move).xyz(atom))
                all_coords_to.append(to_pose.residue(seqpos_to).xyz(atom))
                all_atom_weight.append(weight)
        else:

            if ( align_move[i] != "-" and align_to[i] != "-" ):

                seqpos_move = move_res[i_move]
                seqpos_to = to_res[i_to]

                all_move_to.append((seqpos_move, seqpos_to))



        if ( align_move[i] != "-" ):
            i_move += 1
        if ( align_to[i] != "-" ):
            i_to += 1

    move_pose_copy = move_pose.clone()


    if ( len(move_to_pairs) > 0 ):

        all_coords_move = list(all_coords_move)
        all_coords_to = list(all_coords_to)
        all_atom_weight = list(all_atom_weight)

        #initialize empty mask and distances
        mask = [True for x in range(len(move_to_pairs)*3)]
        
        all_distances = [0 for x in range(len(move_to_pairs)*3)]
        aln_distances = [0 for x in range(len(move_to_pairs)*3)]

        for cycle in range(10):

            move_pose = move_pose_copy.clone()

            #use distances to make a mask
            aln_distances = np.array(aln_distances)
            all_distances = np.array(all_distances)
            cutoff = np.percentile(aln_distances, 100 - throw_away * 10)
            # print("Cutoff %.3f"%cutoff)
            mask = all_distances <= cutoff
            # print(f"mask: {mask.sum()} of {len(mask)}")

            #apply mask to get coords that we will superimpose
            coords_move = rosetta.utility.vector1_numeric_xyzVector_double_t()
            coords_to = rosetta.utility.vector1_numeric_xyzVector_double_t()
            atom_weight = []

            # print(f"len(all_coords_move) {len(all_coords_move)}")
            # print(f"len(mask) {len(mask)}")

            for i in range(len(all_coords_move)):
                if ( not mask[i] ):
                    continue
                coords_move.append(all_coords_move[i])
                coords_to.append(all_coords_to[i])
                atom_weight.append(all_atom_weight[i])

            #build rotation matrix and com      
            rotation_matrix = rosetta.numeric.xyzMatrix_double_t()
            move_com = rosetta.numeric.xyzVector_double_t()
            ref_com = rosetta.numeric.xyzVector_double_t()

            #fill rotation matrix and com in place
            superposition_transform_with_weight(coords_move, coords_to, rotation_matrix, move_com, ref_com, atom_weight)

            #apply rotation matrix and com to pose
            rosetta.protocols.toolbox.apply_superposition_transform(move_pose, rotation_matrix, move_com, ref_com)

            #calculate distances and rmsd for next cycle or to break cycle using mask
            rmsd = 0
            imask = -1
            irmsd = 0
            all_distances = []
            aln_distances = []
            for seqpos_move, seqpos_to in move_to_pairs:
                for atom in atoms:
                    distance = move_pose.residue(seqpos_move).xyz(atom).distance_squared(to_pose.residue(seqpos_to).xyz(atom))
                    all_distances.append(distance)
                    imask += 1
                    if ( not mask[imask] ):
                        continue
                    aln_distances.append(distance)
                    rmsd += distance
                    irmsd += 1

            rmsd /= irmsd
            rmsd = np.sqrt(rmsd)

            print(f"cycle: {cycle} RMSD: {rmsd:.3f}")

            if rmsd <= rmsd_cut:
                break

    return rmsd, move_to_pairs, move_pose

def calculate_radius_of_gyration(pose, chain_num=1):
    """
    Manually calculate the radius of gyration for a specified chain within a pose using only CA atoms.
    Args:
    - pose: A PyRosetta Pose object.
    - chain_num: The number of the chain for which to calculate the radius of gyration.
    
    Returns:
    - The radius of gyration value.
    """
    # Get the indices for the CA atoms of the desired chain
    start_res = pose.conformation().chain_begin(chain_num)
    end_res = pose.conformation().chain_end(chain_num)
    
    # Collect coordinates of CA atoms
    ca_coords = [np.array(pose.residue(i).xyz("CA")) for i in range(start_res, end_res + 1)]
    
    # Calculate the center of mass
    center_of_mass = np.mean(ca_coords, axis=0)
    
    # Calculate the radius of gyration
    rg_squared = np.mean([(np.linalg.norm(coord - center_of_mass))**2 for coord in ca_coords])
    radius_of_gyration = np.sqrt(rg_squared)
    
    return radius_of_gyration

def is_compact_protein(pose, chain_num=1, threshold=14.0):
    """
    Determines if a protein is compact based on the radius of gyration.
    """
    radius_of_gyration = calculate_radius_of_gyration(pose, chain_num)
    print(f"Radius of gyration for chain {chain_num}: {radius_of_gyration:.2f} Å")  # For debugging purposes
    return radius_of_gyration <= threshold

def getSecondaryStructure(pose, return_ss=False, chain_num=1):
    DSSP = pyrosetta.rosetta.protocols.moves.DsspMover()
    chain_of_interest_len = len(pose.chain_sequence(chain_num))
    DSSP.apply(pose)
    ss = pose.secstruct()[:chain_of_interest_len]
    ss_reduced = ''.join([i[0] for i in groupby(list(ss))])

    H_count = ss_reduced.count('H')
    E_count = ss_reduced.count('E')

    if return_ss:
        return ss
    elif H_count > 0 and E_count > 0:
        return f'{H_count}H{E_count}E'
    elif H_count > 0:
        return f'{H_count}H'
    else:
        return f'{E_count}E'

def is_valid_binder_structure(pose, chain_num=1):
    secondary_structure = getSecondaryStructure(pose, chain_num=chain_num)

    # Reject if it's only a single helix
    if secondary_structure == "1H":
        return False
    
    # Reject if it's only two helix
    if secondary_structure == "2H":
        return False

    # Reject if it has a single strand
    if "1E" in secondary_structure:
        return False
    
    # Add any other rejection criteria here

    return True

def is_binder_near_glycans(pose, uniprot_id, threshold=1.0):
    """
    Check if the binder (chain A) is near any glycan residues in the target (chain B).

    Args:
    - pose: Rosetta pose containing chain A and chain B.
    - uniprot_id: UniProt ID for chain B.
    - threshold: Distance threshold for considering residues as 'near'.

    Returns:
    - bool: True if binder is near glycans, False otherwise.
    """

    # Check if we already have the file locally
    filename = f"{uniprot_id}.json"
    
    if not os.path.exists(filename):
        # Download the UniProt JSON if not found locally
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        response = requests.get(url)
        
        # Store the JSON locally
        with open(filename, 'w') as f:
            f.write(response.text)
    
    # Load the data from the local file
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Extracting the glycan sites
    uniprot_seq = data['sequence']['value']
    js_features = data['features']
    glycan_seqpos = []

    for feat in js_features:
        tp = feat['type']
        if tp == "Glycosylation":
            loc = feat['location']
            start = loc['start']['value']
            end = loc['end']['value']
            assert start == end
            glycan_seqpos.append(int(start))
    
    # Align pose sequence with UniProt sequence
    pose_seq = pose.chain_sequence(2)  # Assuming chain B corresponds to the target
    alignment = pairwise2.align.globalms(uniprot_seq, pose_seq, 2, -0.5, -0.1, -0.01)[0]
    #alignment = pairwise2.align.globalms(uniprot_seq, pose_seq, 2, -1, -0.5, -0.1)[0]
    
    # Find the pose positions corresponding to the glycan sites
    pose_glycan_positions = []
    pose_pos = 0
    for uniprot_pos, (uniprot_residue, pose_residue) in enumerate(zip(alignment[0], alignment[1])):
        if pose_residue != "-":
            pose_pos += 1
        if uniprot_pos + 1 in glycan_seqpos:
            pose_glycan_positions.append(pose_pos)
    
    # Correct the pose glycan positions by adding the length of the binder (chain A)
    binder_length = pose.chain_end(1)  # Assuming chain A corresponds to the binder
    pose_glycan_positions = [pos + binder_length for pos in pose_glycan_positions]
    # print(pose_glycan_positions)
    
    # Check if any residue from chain A is near the identified glycan sites
    for glycan_pos in pose_glycan_positions:
        glycan_residue = pose.residue(glycan_pos)
        for res_pos in range(1, pose.chain_end(1) + 1):  # Assuming chain A corresponds to the binder
            if glycan_residue.xyz("CA").distance(pose.residue(res_pos).xyz("CA")) < threshold:
                return True
    
    return False

def extract_hotspots(filename):
    # Split the filename and get the part with hotspots
    hotspot_str = filename.split("_hotspots_")[1].split("_noise")[0]
    # Convert the hotspot string to list of integers
    return list(map(int, hotspot_str.split('x')))

def are_hotspots_in_contact(pose, hotspots, distance_threshold=10.0):
    """
    Check if any of the hotspot residues are in contact with the binder.
    
    Args:
    - pose: A PyRosetta Pose object.
    - hotspots: A list of residue numbers (integers) in the target that are considered hotspots.
    - distance_threshold: Distance threshold for considering two residues to be in contact.
    
    Returns:
    - True if any hotspot residue is in contact with any residue in the binder, False otherwise.
    """
    binder_start = pose.chain_begin(1)
    binder_end = pose.chain_end(1)

    for binder_res in range(binder_start, binder_end + 1):
        for hotspot_res in hotspots:
            distance = pose.residue(binder_res).xyz("CA").distance(pose.residue(hotspot_res).xyz("CA"))
            if distance <= distance_threshold:
                return True  # A hotspot is in contact with the binder
    
    return False  # No hotspots are in contact with the binder

def get_chain_length(pose, chain_num):
    """
    Calculate the length of a specific chain in a pose.
    
    Args:
    - pose: A PyRosetta Pose object.
    - chain_id: The chain identifier for which to calculate the length.
    
    Returns:
    - An integer representing the number of residues in the specified chain.
    """
    #number of residues in the specified chain
    chain_length = pose.chain_end(chain_num) - pose.chain_begin(chain_num) + 1
    
    return chain_length

def passes_quality_checks(pdb_file, pose, uniprot_id):
    print(f"Processing {pdb_file}")
        
    if not is_valid_binder_structure(pose):
        print("Invalid binder structure.")
        return  False
    
    if not is_compact_protein(pose):
        print("Protein is not compact.")
        return  False
    
    #glycan check fails for truncated target when it should not
    #need to debug

    # if is_binder_near_glycans(pose, uniprot_id):
    #     print("Binder is near glycans.")
    #     return  False

    #cannot use hotspots on trimmed target
    #improve logic by alignment like for glycans

    # hotspots = extract_hotspots(pdb_file)
    # if hotspots:
    #     chain_A_length = get_chain_length(pose, 1)
    #     adjusted_hotspots = [x + chain_A_length for x in hotspots]
    
    #     if not are_hotspots_in_contact(pose, adjusted_hotspots):
    #         print("Hotspots are not in contact with binder.")
    #         return False

    return True

def renumber_pose(pose):
    # Get the total number of residues
    n_residues = pose.total_residue()
    
    # Renumber the PDBInfo object
    pdb_info = pose.pdb_info()
    for i in range(1, n_residues + 1):
        pdb_info.number(i, i)

def thread_mpnn_seq( pose, binder_seq ):
    rsd_set = pose.residue_type_set_for_pose( pyrosetta.rosetta.core.chemical.FULL_ATOM_t )

    for resi, mut_to in enumerate( binder_seq ):
        resi += 1 # 1 indexing
        name3 = aa_1_3[ mut_to ]
        new_res = pyrosetta.rosetta.core.conformation.ResidueFactory.create_residue( rsd_set.name_map( name3 ) )
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

def add_to_score_file_open(tag, f, write_header=False, score_dict=None, string_dict=None):
    final_dict = get_final_dict( score_dict, string_dict )
    if ( write_header ):
        f.write("SCORE:     %s description\n"%(" ".join(final_dict.keys())))
    scores_string = " ".join(final_dict.values())
    f.write("SCORE:     %s        %s\n"%(scores_string, tag))

def add2scorefile(tag, scorefilename, score_dict=None):
  
    write_header = not os.path.isfile(scorefilename)
    with open(scorefilename, "a") as f:
        add_to_score_file_open(tag, f, write_header, score_dict)

def add2silent( pose, tag, sfd_out, silent_out):
    struct = sfd_out.create_SilentStructOP()
    struct.fill_struct( pose, tag )
    sfd_out.add_structure( struct )
    sfd_out.write_silent_struct( struct, silent_out )

def dlpredictor( pose ):

    remove_massive_clashes.apply( pose )
    softish_min.apply( pose )
    hard_min.apply( pose )

    ddg = ddg_filter.compute( pose )
    
    return ddg

def optimize_sap(pose, xml, my_per_res_sap):
    sap_dict = my_per_res_sap.calculate(pose)
    bad_sap = [str(x) for x in sap_dict if sap_dict[x] > 1.4]

    newxml = xml.replace("9999999", ",".join(bad_sap))

    objs2 = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(newxml)

    add_sap = objs2.get_mover( 'add_sap' )
    netcharge_cst = objs2.get_mover( 'netcharge_cst' )
    redesign_bad_sap = objs2.get_mover( 'redesign_bad_sap' )

    add_sap.apply(pose)
    netcharge_cst.apply(pose)
    redesign_bad_sap.apply(pose)

    return pose

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

def parse_PDB_biounits(x, atoms=['N','CA','C'], chain=None):
  '''
  input:  x = PDB filename or PDB content as string
          atoms = atoms to extract (optional)
  output: (length, atoms, coords=(x,y,z)), sequence
  '''
  xyz,seq,min_resn,max_resn = {},{},1e6,-1e6
  
  # Determine if x is a filename or PDB content string
  if os.path.exists(x):
      lines = open(x,"rb").readlines()
  else:
      lines = x.splitlines()

  for line in lines:
    if isinstance(line, bytes):  # Convert bytes to string if necessary
        line = line.decode("utf-8","ignore")
    line = line.rstrip()

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

def chain_mask(in_dict, design_these_chains, keep_these_chains):
    pre_chain_list = design_these_chains
    a_list = pre_chain_list.split(",")
    b_list = [str(i) for i in a_list]
    masked_chain_list = b_list
    visible_chain_list_pre = keep_these_chains
    if keep_these_chains == 'NA':
        visible_chain_list = []
    else:
        c_list = visible_chain_list_pre.split(",")
        d_list = [str(i) for i in c_list]
        visible_chain_list = d_list

    masked_dict = {}
    all_chain_list = [item[-1:] for item in list(in_dict) if item[:9]=='seq_chain'] #['A','B', 'C',...]
    masked_dict[in_dict['name']]= (masked_chain_list, visible_chain_list)
    
    return masked_dict

def tied_positions(in_dict, tie_chainsb, tie_chainsa, tie_repeats):
    tie_chainsb_list=[]
    if tie_chainsb not in ['NA']:
        tie_chains_pre = tie_chainsb
        g_list = tie_chains_pre.split(",")
        tie_chainsb_list = [int(i) for i in g_list]
        

    if tie_chainsa not in ['NA']:
        tie_chains_prea = tie_chainsa
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

    elif int(tie_repeats) != 0:
        print('This is a repeat protein/toroid')
        num_reps = int(tie_repeats)
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

def fixed_positions(in_dict, fix_a, fix_b, tied=None):
    if tied != None:
        all_chains = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        tied = tied.split(",")
        tied = [int(x) for x in tied]
        tied = [all_chains[x] for x in tied]
        print(f"HEY HEY tied: {tied}")

    fixed_dict = {}
    if fix_a not in ['NA']:
        fixa = fix_a
        fixlista = fixa.split(",")
        a_list = [int(i) for i in fixlista]
        all_chain_list = [item[-1:] for item in list(in_dict) if item[:9]=='seq_chain']
        fixed_position_dict = {}
        for chain in all_chain_list:
            if chain == 'A':
                if tied != None and 'A' in tied:
                    for tied_chain in tied:
                        fixed_position_dict[tied_chain] = [a_list]
                else:
                    fixed_position_dict[chain] = [a_list]
            elif chain == 'B' and fix_b not in ['NA']:
                fixb = fix_b
                fixlistb = fixb.split(",")
                b_list = [int(i) for i in fixlistb]
                fixed_position_dict[chain] = [b_list]
            else:
                if chain not in fixed_position_dict:
                    fixed_position_dict[chain] = []
        
        fixed_dict[in_dict['name']] = fixed_position_dict

        return fixed_dict
    else:
        return None
