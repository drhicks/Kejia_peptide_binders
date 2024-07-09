import numpy as np
from typing import Tuple
import collections
from collections import OrderedDict
import os, sys

from alphafold.common import residue_constants

import jax.numpy as jnp

import sys

try:
    from silent_tools import silent_tools
except ImportError:
    print("silent_tools not in path; adding directory from drhicks1")
    # The module isn't in the path, add the required directory and try again
    sys.path.append("/home/drhicks1/")
    try:
        from silent_tools import silent_tools
    except ImportError:
        # Handle the case where the module still can't be imported
        print("Failed to import silent_tools even after modifying sys.path")
        sys.exit(1)

from pyrosetta import *
from pyrosetta.rosetta import *
from pyrosetta.rosetta.std import ostringstream

from io import StringIO

def add2scorefile(tag, scorefilename, write_header=False, score_dict=None):
        with open(scorefilename, "a") as f:
                add_to_score_file_open(tag, f, write_header, score_dict)

def add_to_score_file_open(tag, f, write_header=False, score_dict=None, string_dict=None):
        final_dict = get_final_dict( score_dict, string_dict )
        if ( write_header ):
                f.write("SCORE:  %s description\n"%(" ".join(final_dict.keys())))
        scores_string = " ".join(final_dict.values())
        f.write("SCORE:  %s    %s\n"%(scores_string, tag))

def add2silent( tag, pose, score_dict, sfd_out ):
        # pose = pose_from_file( pdb )

        # pose = insert_chainbreaks( pose, binderlen )

        struct = sfd_out.create_SilentStructOP()
        struct.fill_struct( pose, tag )

        for scorename, value in score_dict.items():
            if ( isinstance(value, str) ):
                struct.add_string_value(scorename, value)
            else:
                struct.add_energy(scorename, value)

        sfd_out.add_structure( struct )
        sfd_out.write_silent_struct( struct, "out.silent" )

def record_checkpoint( tag_buffer, checkpoint_filename ):
        with open( checkpoint_filename, 'a' ) as f:
                for tag in tag_buffer:
                        f.write( tag )
                        f.write( '\n' )

def determine_finished_structs( checkpoint_filename ):
        done_set = set()
        if not os.path.isfile( checkpoint_filename ): return done_set

        with open( checkpoint_filename, 'r' ) as f:
                for line in f:
                        done_set.add( line.strip() )

        return done_set

def get_queries_from_silent(silent_file):
    silent_index = silent_tools.get_silent_index( silent_file )

    with open(silent_file, errors='ignore') as sf:
        queries = []
        for tag in silent_index['tags']:

            structure = silent_tools.get_silent_structure_file_open( sf, silent_index, tag )

            sequence_chunks = silent_tools.get_sequence_chunks( structure )

            if (sequence_chunks is None):
                continue

            queries.append([tag, sequence_chunks, None])

    return queries

def pose_from_silent(sfd_in, tag):
    pose = pyrosetta.Pose()
    sfd_in.get_structure(tag).fill_pose(pose)
    return pose

def generate_template_features(
                                seq: str,
                                all_atom_positions: np.ndarray,
                                all_atom_masks: np.ndarray,
                                residue_mask: list
                                ) -> dict:
    '''
    Given the sequence and all atom positions and masks, generate the template features.
    Residues which are False in the residue mask are not included in the template features,
    this means they will be free to be predicted by the model.
    '''

    # Split the all atom positions and masks into a list of arrays for easier manipulation
    all_atom_positions = np.split(all_atom_positions, all_atom_positions.shape[0])
    all_atom_masks = np.split(all_atom_masks, all_atom_masks.shape[0])

    output_templates_sequence = []
    output_confidence_scores = []
    templates_all_atom_positions = []
    templates_all_atom_masks = []

    # Initially fill will all zero values
    for _ in seq:
        templates_all_atom_positions.append(
            np.zeros((residue_constants.atom_type_num, 3)))
        templates_all_atom_masks.append(np.zeros(residue_constants.atom_type_num))
        output_templates_sequence.append('-')
        output_confidence_scores.append(-1)

    confidence_scores = []
    for _ in seq: confidence_scores.append( 9 )

    for idx, i in enumerate(seq):

        if not residue_mask[ idx ]: continue

        templates_all_atom_positions[ idx ] = all_atom_positions[ idx ][0] # assign target indices to template coordinates
        templates_all_atom_masks[ idx ] = all_atom_masks[ idx ][0]
        output_templates_sequence[ idx ] = seq[ idx ]
        output_confidence_scores[ idx ] = confidence_scores[ idx ] # 0-9 where higher is more confident

    output_templates_sequence = ''.join(output_templates_sequence)

    templates_aatype = residue_constants.sequence_to_onehot(
        output_templates_sequence, residue_constants.HHBLITS_AA_TO_ID)

    template_feat_dict = {'template_all_atom_positions': np.array(templates_all_atom_positions)[None],
        'template_all_atom_masks': np.array(templates_all_atom_masks)[None],
        'template_sequence': [output_templates_sequence.encode()],
        'template_aatype': np.array(templates_aatype)[None],
        'template_confidence_scores': np.array(output_confidence_scores)[None],
        'template_domain_names': ['none'.encode()],
        'template_release_date': ["none".encode()]}

    return template_feat_dict   

def parse_initial_guess(all_atom_positions) -> jnp.ndarray:
    '''
    Given a numpy array of all atom positions, return a jax array of the initial guess
    '''

    list_all_atom_positions = np.split(all_atom_positions, all_atom_positions.shape[0])

    templates_all_atom_positions = []

    # Initially fill with zeros
    for _ in list_all_atom_positions:
        templates_all_atom_positions.append(jnp.zeros((residue_constants.atom_type_num, 3)))

    for idx in range(len(list_all_atom_positions)):
        templates_all_atom_positions[idx] = list_all_atom_positions[idx][0] 

    return jnp.array(templates_all_atom_positions)

def af2_get_atom_positions(pose) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Given a pose, return the AF2 atom positions array and atom mask array for the protein.
    '''

    # Create a string stream
    pdb_stream = ostringstream()

    # Dump the pose to the string stream instead of a file
    pose.dump_pdb(pdb_stream)

    # Get the string from the stream
    pdb_str = pdb_stream.str()

    # Process the string to get lines
    lines = pdb_str.split('\n')

    # indices of residues observed in the structure (C-alpha atoms)
    idx_s = [int(l[22:26]) for l in lines if l.startswith("ATOM") and l[12:16].strip() == "CA"]

    # Make a set from idx_s
    idx_set = set(idx_s)

    # Check for duplicate residue numbers
    if len(idx_s) != len(idx_set):
        print("Duplicate residue numbers found.")
        return None, None

    num_res = len(idx_s)

    all_positions = np.zeros([num_res, residue_constants.atom_type_num, 3])
    all_positions_mask = np.zeros([num_res, residue_constants.atom_type_num], dtype=np.int64)

    residues = collections.defaultdict(list)
    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(idx_s), 14, 3), np.nan, dtype=np.float32)
    for l in lines:
        if not l.startswith("ATOM"):
            continue
        resNo, atom, aa = int(l[22:26]), l[12:16], l[17:20]

        residues[resNo].append((atom.strip(), aa, [float(l[30:38]), float(l[38:46]), float(l[46:54])]))

    for resNo in residues:

        pos = np.zeros([residue_constants.atom_type_num, 3], dtype=np.float32)
        mask = np.zeros([residue_constants.atom_type_num], dtype=np.float32)

        for atom in residues[resNo]:
            atom_name = atom[0]
            x, y, z = atom[2]
            if atom_name in residue_constants.atom_order.keys():
                pos[residue_constants.atom_order[atom_name]] = [x, y, z]
                mask[residue_constants.atom_order[atom_name]] = 1.0
            elif atom_name.upper() == 'SE' and res.get_resname() == 'MSE':
                # Put the coordinates of the selenium atom in the sulphur column.
                pos[residue_constants.atom_order['SD']] = [x, y, z]
                mask[residue_constants.atom_order['SD']] = 1.0

        idx = idx_s.index(resNo) # This is the order they show up in the pdb
        all_positions[idx] = pos
        all_positions_mask[idx] = mask

    return all_positions, all_positions_mask

def insert_truncations(residue_index, Ls) -> np.ndarray:
    '''
    Given the residue index feature and the absolute indices of the truncations,
    insert the truncations into the residue index feature.

    Args:
        residue_index (np.ndarray) : [L] The residue index feature.

        Ls (list)              : The absolute indices of the chainbreaks.
                                     Chainbreaks will be inserted after these zero-indexed indices.
    '''

    idx_res = residue_index
    for break_i in Ls:
        idx_res[break_i:] += 200
    
    residue_index = idx_res

    return residue_index

def get_final_dict(score_dict, string_dict) -> OrderedDict:
    '''
    Given dictionaries of numerical scores and a string scores, return a sorted dictionary
    of the scores, ready to be written to the scorefile.
    '''

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

def insert_Rosetta_chainbreaks( pose, binderlen ) -> core.pose.Pose:
    '''
    Given a pose and a list of indices to insert chainbreaks after,
    insert the chainbreaks into the pose.

    Args:
        pose (Pose)   : The pose to insert chainbreaks into.

        binderlen (list) : The length of the binder chain
    '''

    conf = pose.conformation()
    conf.insert_chain_ending( binderlen )

    pose.set_new_conformation( conf )

    splits = pose.split_by_chain()
    newpose = splits[1]
    for i in range( 2, len( splits )+1 ):
        newpose.append_pose_by_jump( splits[i], newpose.size() )
 
    info = core.pose.PDBInfo( newpose, True )
    newpose.pdb_info( info )

    return newpose

def check_residue_distances(all_positions, all_positions_mask, max_amide_distance) -> list:
    '''
    Given a list of residue positions and a maximum amide distance, determine which residues
    are too far apart and should have a chainbreak inserted between them.

    This is mostly taken from the AF2 source code and modified for our purposes.
    '''

    breaks = []
    
    c_position = residue_constants.atom_order['C']
    n_position = residue_constants.atom_order['N']
    prev_is_unmasked = False
    this_c = None
    for i, (coords, mask) in enumerate(zip(all_positions, all_positions_mask)):

        # These coordinates only should be considered if both the C and N atoms are present.
        this_is_unmasked = bool(mask[c_position]) and bool(mask[n_position])
        if this_is_unmasked:
            this_n = coords[n_position]
            # Check whether the previous residue had both C and N atoms present.
            if prev_is_unmasked:

                distance = np.linalg.norm(this_n - prev_c)
                if distance > max_amide_distance:
                    # If the distance between the C and N atoms is too large, insert a chainbreak.
                    # This chainbreak is listed as being at residue i in zero-indexed numbering.
                    breaks.append(i)
                    print( f'The distance between residues {i} and {i+1} is {distance:.2f} A' +
                        f' > limit {max_amide_distance} A.' )
                    print( f"I'm going to insert a chainbreak after residue {i}" )

            prev_c = coords[c_position]

        prev_is_unmasked = this_is_unmasked

    return breaks
