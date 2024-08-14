import pyrosetta
import os
import sys
from collections import OrderedDict
import numpy as np

pyrosetta.init('-in:file:silent_struct_type binary -beta_nov16')

def read_tag_map(tag_map_file):
    """Reads the tag map file and returns a list of tuples (tag1, tag2)."""
    tag_map = []
    with open(tag_map_file, 'r') as f:
        for line in f:
            tag1, tag2 = line.strip().split()
            tag_map.append((tag1, tag2))
    return tag_map

def add2scorefile(tag, scorefilename, write_header=False, score_dict=None, string_dict=None):
    with open(scorefilename, "a") as f:
        add_to_score_file_open(tag, f, write_header, score_dict, string_dict)

def get_final_dict(score_dict, string_dict=None):
    '''
    Given dictionaries of numerical scores and string scores, return a sorted dictionary
    of the scores, ready to be written to the scorefile.
    '''
    final_dict = OrderedDict()
    keys_score = [] if score_dict is None else list(score_dict)
    keys_string = [] if string_dict is None else list(string_dict)

    all_keys = keys_score + keys_string

    argsort = sorted(range(len(all_keys)), key=lambda x: all_keys[x])

    for idx in argsort:
        key = all_keys[idx]
        if idx < len(keys_score):
            final_dict[key] = "%8.3f" % (score_dict[key])
        else:
            final_dict[key] = string_dict[key]

    return final_dict

def add_to_score_file_open(tag, f, write_header=False, score_dict=None, string_dict=None):
    final_dict = get_final_dict(score_dict, string_dict)
    if write_header:
        f.write("SCORE:  %s description\n" % (" ".join(final_dict.keys())))
    scores_string = " ".join(final_dict.values())
    f.write("SCORE:  %s    %s\n" % (scores_string, tag))

def pose_from_silent(sfd_in, tag):
    pose = pyrosetta.Pose()
    sfd_in.get_structure(tag).fill_pose(pose)
    return pose

def selector_CA_align(pose1, pose2, pose1_residue_selector, pose2_residue_selector):
    pose1_residue_selection = pyrosetta.rosetta.core.select.get_residues_from_subset(pose1_residue_selector.apply(pose1))
    pose2_residue_selection = pyrosetta.rosetta.core.select.get_residues_from_subset(pose2_residue_selector.apply(pose2))

    assert len(pose1_residue_selection) == len(pose2_residue_selection)

    pose1_coordinates = pyrosetta.rosetta.utility.vector1_numeric_xyzVector_double_t()
    pose2_coordinates = pyrosetta.rosetta.utility.vector1_numeric_xyzVector_double_t()

    for pose1_residue_index, pose2_residue_index in zip(pose1_residue_selection, pose2_residue_selection):
        pose1_coordinates.append(pose1.residue(pose1_residue_index).xyz('CA'))
        pose2_coordinates.append(pose2.residue(pose2_residue_index).xyz('CA'))

    rotation_matrix = pyrosetta.rosetta.numeric.xyzMatrix_double_t()
    pose1_center = pyrosetta.rosetta.numeric.xyzVector_double_t()
    pose2_center = pyrosetta.rosetta.numeric.xyzVector_double_t()

    pyrosetta.rosetta.protocols.toolbox.superposition_transform(pose1_coordinates,
                                                                pose2_coordinates,
                                                                rotation_matrix,
                                                                pose1_center,
                                                                pose2_center)

    pyrosetta.rosetta.protocols.toolbox.apply_superposition_transform(pose1,
                                                                      rotation_matrix,
                                                                      pose1_center,
                                                                      pose2_center)

import numpy as np

def CA_rmsd(pose1, pose2, chain_residues):
    """
    Calculate the RMSD between the CA atoms of the selected residues in pose1 and pose2 using NumPy.

    Args:
        pose1: PyRosetta Pose object for the first structure.
        pose2: PyRosetta Pose object for the second structure.
        chain_residues: List of residue indices to be included in the RMSD calculation.

    Returns:
        The RMSD value between the CA atoms of the specified residues.
    """
    # Extract the CA atom coordinates from both poses
    coords1 = []
    coords2 = []
    
    for res_id in chain_residues:
        # Convert xyzVector_double_t to a NumPy array
        coord1 = np.array([pose1.residue(res_id).xyz("CA").x,
                           pose1.residue(res_id).xyz("CA").y,
                           pose1.residue(res_id).xyz("CA").z])
        
        coord2 = np.array([pose2.residue(res_id).xyz("CA").x,
                           pose2.residue(res_id).xyz("CA").y,
                           pose2.residue(res_id).xyz("CA").z])
        
        coords1.append(coord1)
        coords2.append(coord2)

    coords1 = np.array(coords1)
    coords2 = np.array(coords2)
    
    # Calculate the difference between the coordinates
    diff = coords1 - coords2
    
    # Square the differences and sum them
    squared_diff = np.sum(diff ** 2, axis=1)
    
    # Calculate the RMSD
    rmsd = np.sqrt(np.mean(squared_diff))
    
    return rmsd

def CA_rmsd_optimized(pose1, pose2, chain_residues):
    """
    Optimized calculation of RMSD between the CA atoms of selected residues in pose1 and pose2 using NumPy.
    """
    # Extract coordinates into a single NumPy array operation
    coords1 = np.array([
        [pose1.residue(res_id).xyz("CA").x,
         pose1.residue(res_id).xyz("CA").y,
         pose1.residue(res_id).xyz("CA").z]
        for res_id in chain_residues
    ])

    coords2 = np.array([
        [pose2.residue(res_id).xyz("CA").x,
         pose2.residue(res_id).xyz("CA").y,
         pose2.residue(res_id).xyz("CA").z]
        for res_id in chain_residues
    ])

    # Calculate squared differences
    diff_squared = np.sum((coords1 - coords2) ** 2, axis=1)
    
    # Calculate RMSD
    rmsd = np.sqrt(np.mean(diff_squared))
    
    return rmsd

def superimpose_and_calculate_rmsd(pose1, pose2, chain_id):
    chainA = pyrosetta.rosetta.core.select.residue_selector.ChainSelector("A")
    chainB = pyrosetta.rosetta.core.select.residue_selector.ChainSelector("B")

    chain_sel = chainA if chain_id == "A" else chainB

    # Check the chain length
    chain1_length_pose1 = pose1.chain_end(1) - pose1.chain_begin(1) + 1
    chain1_length_pose2 = pose2.chain_end(1) - pose2.chain_begin(1) + 1
    chain2_length_pose1 = pose1.chain_end(2) - pose1.chain_begin(2) + 1
    chain2_length_pose2 = pose2.chain_end(2) - pose2.chain_begin(2) + 1

    if chain1_length_pose1 != chain1_length_pose2 or chain2_length_pose1 != chain2_length_pose2:
        print(f"Warning: Chain length mismatch between poses for chain {chain_id}. Skipping this pair.")
        return {}

    # Superimpose by CA of the selected chain
    selector_CA_align(pose1, pose2, pose1_residue_selector=chain_sel, pose2_residue_selector=chain_sel)

    # Get the residue ranges for chain A and B
    chain1_residues_A = list(range(pose1.chain_begin(1), pose1.chain_end(1) + 1))
    chain1_residues_B = list(range(pose1.chain_begin(2), pose1.chain_end(2) + 1))

    # Calculate RMSD for chain A to chain A and chain B to chain B
    rmsd_chain_a = CA_rmsd_optimized(pose1, pose2, chain1_residues_A)
    rmsd_chain_b = CA_rmsd_optimized(pose1, pose2, chain1_residues_B)

    score_dict = {
        f"super_{chain_id}_rmsd_A": rmsd_chain_a,
        f"super_{chain_id}_rmsd_B": rmsd_chain_b
    }

    return score_dict

if __name__ == "__main__":
    import sys
    silent_file1 = sys.argv[1]
    silent_file2 = sys.argv[2]
    tag_map_file = sys.argv[3]

    scorefilename = "rmsd_out.sc"
    write_header = not os.path.exists(scorefilename)

    sfd_in1 = pyrosetta.rosetta.core.io.silent.SilentFileData(pyrosetta.rosetta.core.io.silent.SilentFileOptions())
    sfd_in1.read_file(silent_file1)

    sfd_in2 = pyrosetta.rosetta.core.io.silent.SilentFileData(pyrosetta.rosetta.core.io.silent.SilentFileOptions())
    sfd_in2.read_file(silent_file2)

    tag_map = read_tag_map(tag_map_file)

    for tag1, tag2 in tag_map:
        pose1 = pose_from_silent(sfd_in1, tag1)
        pose2 = pose_from_silent(sfd_in2, tag2)

        name_dict = {"description2": tag2}
        score_dict = {}
        score_dict.update(superimpose_and_calculate_rmsd(pose1, pose2, "A"))
        score_dict.update(superimpose_and_calculate_rmsd(pose1, pose2, "B"))

        if len(score_dict) < 1:
            continue

        if score_dict:
            add2scorefile(f"{tag1}", scorefilename, write_header=write_header, score_dict=score_dict, string_dict=name_dict)
            write_header = False
