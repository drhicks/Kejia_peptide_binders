import pyrosetta
import argparse
import blosum62
import pandas as pd
import logging
import sys
from pyrosetta.rosetta.protocols.flexpep_docking import FlexPepDockingProtocol

# Constants
DEFAULT_PYROSETTA_FLAGS = '-pep_refine -ex1 -ex2aro -restore_talaris_behavior -default_max_cycles 100 ' \
                          '-dunbrack_prob_buried 0.8 -dunbrack_prob_nonburied 0.8 -dunbrack_prob_buried_semi 0.8 ' \
                          '-dunbrack_prob_nonburied_semi 0.8 -boost_fa_atr False -rep_ramp_cycles 5 -mcm_cycles 5'
DEFAULT_INTERFACE_DIST_CUTOFF = 8.0
DEFAULT_MIN_SEQ_IDENTITY = 0.0
DEFAULT_MIN_BLOSUM = -10
DEFAULT_MAX_FAREP = 300.0
DEFAULT_FLEX_DOCKING_REPEATS = 10
DEFAULT_OUTPUT_CSV = 'thread_peptide_sequence_stats.csv'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

codemap = {
    'R': 'ARG', 'H': 'HIS', 'K': 'LYS', 'D': 'ASP', 'E': 'GLU',
    'S': 'SER', 'T': 'THR', 'N': 'ASN', 'Q': 'GLN', 'C': 'CYS',
    'G': 'GLY', 'P': 'PRO', 'A': 'ALA', 'V': 'VAL', 'I': 'ILE',
    'L': 'LEU', 'M': 'MET', 'F': 'PHE', 'Y': 'TYR', 'W': 'TRP'
}

def pep_flex_dock(in_pose, sf):
    """Perform flexible peptide docking."""
    pose = in_pose.clone()
    fpdock = FlexPepDockingProtocol()
    fpdock.apply(pose)
    return pose

def find_hbonds(pose, search_reslist, target_reslist, skip=[]):
    """Find hydrogen bonds in the pose."""
    my_hbond_set = {'bb-bb': [], 'bb-sc': [], 'sc-bb': [], 'sc-sc': []}
    my_hbond_list = []
    hbond_set = pyrosetta.rosetta.core.scoring.hbonds.HBondSet()
    pose.update_residue_neighbors()
    pyrosetta.rosetta.core.scoring.hbonds.fill_hbond_set(pose, False, hbond_set)
    for i in search_reslist:
        for hb_i in hbond_set.residue_hbonds(i):
            if hb_i.acc_res() in target_reslist or hb_i.don_res() in target_reslist:
                if len(skip) > 0:
                    skip_check = any(hb_i.acc_res() == hb_i.don_res() + k or hb_i.acc_res() == hb_i.don_res() - k for k in skip)
                    if skip_check:
                        continue

                if (hb_i.don_res(), hb_i.acc_res(), hb_i) in my_hbond_list:
                    continue
                my_hbond_list.append((hb_i.don_res(), hb_i.acc_res(), hb_i))

                if hb_i.acc_atm_is_backbone() and hb_i.don_hatm_is_backbone():
                    my_hbond_set['bb-bb'].append((hb_i.don_res(), hb_i.acc_res(), hb_i))
                elif (hb_i.acc_atm_is_backbone() and hb_i.acc_res() in search_reslist) \
                        or (hb_i.don_hatm_is_backbone() and hb_i.don_res() in search_reslist):
                    my_hbond_set['bb-sc'].append((hb_i.don_res(), hb_i.acc_res(), hb_i))
                elif (hb_i.acc_atm_is_backbone() and hb_i.acc_res() in target_reslist) \
                        or (hb_i.don_hatm_is_backbone() and hb_i.don_res() in target_reslist):
                    my_hbond_set['sc-bb'].append((hb_i.don_res(), hb_i.acc_res(), hb_i))
                elif hb_i.acc_atm_is_backbone() or hb_i.don_hatm_is_backbone():
                    my_hbond_set['bb-sc'].append((hb_i.don_res(), hb_i.acc_res(), hb_i))
                else:
                    my_hbond_set['sc-sc'].append((hb_i.don_res(), hb_i.acc_res(), hb_i))
    return my_hbond_list, my_hbond_set

def find_bidentate_hbond(pose, hbond_list):
    """Find bidentate hydrogen bonds in the pose."""
    hbonds_by_sc_res = {}
    for hb in hbond_list:
        hb_i = hb[-1]
        if hb_i.don_res() not in hbonds_by_sc_res:
            hbonds_by_sc_res[hb_i.don_res()] = []
        if hb_i.acc_res() not in hbonds_by_sc_res:
            hbonds_by_sc_res[hb_i.acc_res()] = []

        hb_info = (hb_i.don_res(), pose.residue(hb_i.don_res()).atom_name(hb_i.don_hatm()).strip(),
                   hb_i.acc_res(), pose.residue(hb_i.acc_res()).atom_name(hb_i.acc_atm()).strip())
        if not hb_i.don_hatm_is_backbone():
            hbonds_by_sc_res[hb_i.don_res()].append(hb_info)
        if not hb_i.acc_atm_is_backbone():
            hbonds_by_sc_res[hb_i.acc_res()].append(hb_info)

    bidentate_hbonds_by_sc_res = {res: hbonds_by_sc_res[res] for res in hbonds_by_sc_res if len(hbonds_by_sc_res[res]) > 1}
    return bidentate_hbonds_by_sc_res

def _config_my_task_factory_for_repack(added_residues):
    """Configure the task factory for repacking."""
    print("DEBUG 1")
    repack_residues = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
    for res_num in added_residues:
        repack_residues.append_index(res_num)
    restrict_to_repack = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        pyrosetta.rosetta.core.pack.task.operation.RestrictToRepackingRLT(), repack_residues, False)
    freeze_rest = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT(), repack_residues, True)
    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
    tf.push_back(restrict_to_repack)
    tf.push_back(freeze_rest)
    print("DEBUG 2")
    return tf

def repack_pose(pose, sf, reslist):
    """Repack the pose."""
    pack_rotamers = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(sf)
    pack_rotamers.task_factory(_config_my_task_factory_for_repack(reslist))
    pack_rotamers.apply(pose)
    return pose

def setup_movemap(pose, bblist=[], chilist=[]):
    """Set up the move map."""
    mm = pyrosetta.rosetta.core.kinematics.MoveMap()
    if len(bblist) == 0 and len(chilist) == 0:
        mm.set_chi(True)
        mm.set_bb(True)
    else:
        mm.set_chi(False)
        mm.set_bb(False)
        for resid in range(1, pose.size() + 1):
            if resid in bblist:
                mm.set_bb(resid, True)
                mm.set_chi(resid, True)
            elif resid in chilist:
                mm.set_chi(resid, True)
    mm.set_jump(True)
    return mm

def minimize_pose(pose, sf, reslist, cartesian_=False, coordcst_=False):
    """Minimize the pose."""
    movemap = setup_movemap(pose, bblist=reslist, chilist=reslist)
    if coordcst_:
        sf.set_weight(pyrosetta.rosetta.core.scoring.coordinate_constraint, 1.0)
        pyrosetta.rosetta.core.scoring.constraints.add_coordinate_constraints(pose, 0.1, False)

    min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover(movemap, sf, 'lbfgs_armijo_nonmonotone', 0.01, True, True, False)
    min_mover.max_iter(100)
    if cartesian_:
        min_mover.cartesian(True)
    min_mover.apply(pose)
    return pose

def compute_seq_identity(seq1, seq2):
    """Compute sequence identity between two sequences."""
    assert len(seq1) == len(seq2)
    ic = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return ic / len(seq1)

def process_peptides(args, pose, input_pose, template_name, bidentates, peptide_reslist, bb_farep, sf, sf_farep):
    """Process the peptides and generate data."""
    peptides = {args.peptide_header: args.peptide_seq}
    data = {x: [] for x in ['pdb', 'template', 'template_bidentates', 'target_peptide', 'template_peptide_start',
                            'template_peptide_end', 'target_peptide_start', 'target_peptide_end', 'fa_rep',
                            'sequence_identity', 'blosum_score']}

    for pep in peptides:
        logging.info(f'Threading {pep} onto {template_name}')
        pose_thread_list = []
        this_peptide_reslist = peptide_reslist.copy()
        this_ref_bidentates = bidentates.copy()

        if len(peptides[pep]) < len(peptide_reslist):
            pose_thread_list_tmp, this_peptide_reslist = handle_shorter_peptides(peptides, pep, input_pose, pose, bidentates, this_ref_bidentates,
                                                            bb_farep, sf, sf_farep, this_peptide_reslist, template_name,
                                                            args.min_seq_identity, args.min_blosum)
            pose_thread_list.extend(pose_thread_list_tmp)

        else:
            pose_thread_list_tmp, this_peptide_reslist = handle_longer_peptides(peptides, pep, input_pose, pose, bidentates, this_ref_bidentates,
                                                           bb_farep, sf, sf_farep, this_peptide_reslist, template_name,
                                                           args.min_seq_identity, args.min_blosum)
            pose_thread_list.extend(pose_thread_list_tmp)

        pose_packmin_list = []
        for pt_id, pt in enumerate(pose_thread_list):
            pt[0] = repack_pose(pt[0], sf, this_peptide_reslist)
            pt[0] = minimize_pose(pt[0], sf.clone(), this_peptide_reslist, coordcst_=True)

            for pert in range(0, args.flex_docking_repeats):
                pt[0] = pep_flex_dock(pt[0], sf)
                if sf_farep(pt[0]) - bb_farep <= args.max_farep:
                    if args.keep_bidentates:
                        this_hbond_list, this_hbond_set = find_hbonds(pt[0], list(range(1, pt[0].split_by_chain(1).size() + 1)),
                                                                      list(range(pt[0].split_by_chain(1).size() + 1, pt[0].size() + 1)))
                        this_bidentates = find_bidentate_hbond(pt[0], this_hbond_set['sc-bb'])
                        bidentate_check = True
                        for bidentate_resi in this_ref_bidentates:
                            if bidentate_resi not in this_bidentates:
                                bidentate_check = False
                                break
                        if not bidentate_check:
                            continue
                    pose_packmin_list.append(pt + [sf_farep(pt[0]) - bb_farep])

        for ipt, pt in enumerate(pose_packmin_list):
            pt[0].dump_pdb(f'{template_name}_{pep}_{pt[1][0]}_{pt[1][1]}_{pt[2][0]}_{pt[2][1]}_{ipt}.pdb')
            data['pdb'].append(f'{template_name}_{pt[1][0]}_{pt[1][1]}_{pt[2][0]}_{pt[2][1]}_{ipt}.pdb')
            data['template'].append(template_name)
            data['template_bidentates'].append('_'.join([str(x) for x in sorted(list(this_ref_bidentates.keys()))]))
            data['target_peptide'].append(pep)
            data['template_peptide_start'].append(pt[1][0])
            data['template_peptide_end'].append(pt[1][1])
            data['target_peptide_start'].append(pt[2][0])
            data['target_peptide_end'].append(pt[2][1])
            data['fa_rep'].append(pt[-1])
            data['sequence_identity'].append(pt[-3])
            data['blosum_score'].append(pt[-2])

    return data

def handle_shorter_peptides(peptides, pep, input_pose, pose, bidentates, this_ref_bidentates, bb_farep, sf, sf_farep,
                            this_peptide_reslist, template_name, min_seq_identity, min_blosum):
    """Handle threading for shorter peptides."""
    pose_thread_list = []
    for start_pos in range(len(this_peptide_reslist) - len(peptides[pep]) + 1):
        template_seq = input_pose.sequence(input_pose.split_by_chain(1).size() + start_pos + 1,
                                           input_pose.split_by_chain(1).size() + start_pos + len(peptides[pep]))
        thread_seq = peptides[pep]
        seq_id = compute_seq_identity(template_seq, thread_seq)
        blosum_score = blosum62.compute_blosum62_score(template_seq, thread_seq)
        if seq_id >= min_seq_identity and blosum_score >= min_blosum:
            pose_thread = pose.clone()
            for resid in range(len(peptides[pep])):
                mt = pyrosetta.rosetta.protocols.simple_moves.MutateResidue(
                    input_pose.split_by_chain(1).size() + start_pos + resid + 1, codemap[peptides[pep][resid]])
                mt.apply(pose_thread)
            deleted_reslist = []
            if input_pose.split_by_chain(1).size() + start_pos + len(peptides[pep]) < pose_thread.size():
                pose_thread.delete_residue_range_slow(input_pose.split_by_chain(1).size() + start_pos + len(peptides[pep]) + 1,
                                                      pose_thread.size())
                deleted_reslist += list(range(input_pose.split_by_chain(1).size() + start_pos + len(peptides[pep]) + 1,
                                              pose_thread.size() + 1))
            if start_pos > 0:
                pose_thread.delete_residue_range_slow(input_pose.split_by_chain(1).size() + 1,
                                                      input_pose.split_by_chain(1).size() + start_pos)
                deleted_reslist += list(range(input_pose.split_by_chain(1).size() + 1,
                                              input_pose.split_by_chain(1).size() + start_pos + 1))

            this_peptide_reslist = list(range(input_pose.split_by_chain(1).size() + 1, pose_thread.size() + 1))
            for sc in bidentates:
                for bbresi in [bidentates[sc][0][0], bidentates[sc][0][2], bidentates[sc][1][0], bidentates[sc][1][2]]:
                    if bbresi in deleted_reslist and sc in this_ref_bidentates:
                        this_ref_bidentates.pop(sc)
            pose_thread_list.append([pose_thread, [start_pos + 1, start_pos + len(peptides[pep])], [1, len(peptides[pep])],
                                     seq_id, blosum_score])
    return pose_thread_list, this_peptide_reslist

def handle_longer_peptides(peptides, pep, input_pose, pose, bidentates, this_ref_bidentates, bb_farep, sf, sf_farep,
                           this_peptide_reslist, template_name, min_seq_identity, min_blosum):
    """Handle threading for longer peptides."""
    pose_thread_list = []
    for start_pos in range(len(peptides[pep]) - len(this_peptide_reslist) + 1):
        template_seq = input_pose.sequence(input_pose.split_by_chain(1).size() + 1,
                                           input_pose.split_by_chain(1).size() + len(this_peptide_reslist))
        thread_seq = peptides[pep][start_pos:start_pos + len(this_peptide_reslist)]
        seq_id = compute_seq_identity(template_seq, thread_seq)
        blosum_score = blosum62.compute_blosum62_score(template_seq, thread_seq)
        if seq_id >= min_seq_identity and blosum_score >= min_blosum:
            pose_thread = pose.clone()
            for resid in range(len(this_peptide_reslist)):
                mt = pyrosetta.rosetta.protocols.simple_moves.MutateResidue(
                    input_pose.split_by_chain(1).size() + resid + 1, codemap[peptides[pep][start_pos + resid]])
                mt.apply(pose_thread)
            pose_thread_list.append([pose_thread, [1, len(this_peptide_reslist)],
                                     [start_pos + 1, start_pos + len(this_peptide_reslist)], seq_id, blosum_score])
    return pose_thread_list, this_peptide_reslist

def main(args):
    """Main function to execute the peptide threading and docking."""
    try:
        pyrosetta.init(args.pyrosetta_flags)
        sf = pyrosetta.get_score_function()
        # sf_cart = sf.clone()
        # sf_cart.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.pro_close, 0.0)
        # sf_cart.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded, 0.5)
        sf_farep = sf.clone()
        for st in sf_farep.get_nonzero_weighted_scoretypes():
            sf_farep.set_weight(st, 0)
        sf_farep.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.fa_rep, 1)

        template_file = args.template.strip()
        template_name = template_file.split('/')[-1].replace('.pdb', '')

        try:
            input_pose = pyrosetta.pose_from_file(template_file)
        except Exception as e:
            logging.error(f"Failed to load template file: {e}")
            sys.exit(1)

        binder_size = input_pose.split_by_chain(1).size()
        pose = input_pose.clone()

        hbond_list, hbond_set = find_hbonds(input_pose, list(range(1, input_pose.split_by_chain(1).size() + 1)),
                                            list(range(input_pose.split_by_chain(1).size() + 1, input_pose.size() + 1)))
        bidentates = find_bidentate_hbond(input_pose, hbond_set['sc-bb'])

        peptide_reslist = list(range(binder_size + 1, pose.size() + 1))
        print("DEBUG 3")
        peptide_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
        for res_num in peptide_reslist:
            peptide_selector.append_index(res_num)
        interface_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(
            peptide_selector, args.interface_dist_cutoff, True)
        interface_vec = interface_selector.apply(pose)
        interface_reslist = [i + 1 for i, j in enumerate(interface_vec) if j]

        for resi in interface_reslist:
            if args.keep_bidentates and resi in bidentates:
                continue

        bb_farep = sf_farep(pose)
        print("DEBUG 4")
        data = process_peptides(args, pose, input_pose, template_name, bidentates, peptide_reslist, bb_farep, sf, sf_farep)
        print("DEBUG 5")
        try:
            df = pd.DataFrame(data)
            df.to_csv(args.output_csv, index=False)
            logging.info(f"Results successfully saved to {args.output_csv}")
        except Exception as e:
            logging.error(f"Failed to save results to CSV: {e}")
            sys.exit(1)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', type=str, required=True, help='A single template')
    parser.add_argument('--peptide_header', type=str, required=True, help='Peptide header')
    parser.add_argument('--peptide_seq', type=str, required=True, help='Peptide sequence')
    parser.add_argument('--interface_dist_cutoff', type=float, default=DEFAULT_INTERFACE_DIST_CUTOFF,
                        help=f'Distance cutoff for neighborhood selector for interface residues. Default={DEFAULT_INTERFACE_DIST_CUTOFF}')
    parser.add_argument('--min_seq_identity', type=float, default=DEFAULT_MIN_SEQ_IDENTITY,
                        help=f'Min sequence identity required between threaded sequence and original peptide sequence in the template. Default={DEFAULT_MIN_SEQ_IDENTITY}')
    parser.add_argument('--min_blosum', type=float, default=DEFAULT_MIN_BLOSUM,
                        help=f'Min blosum score required between threaded sequence and original peptide sequence in the template. Default={DEFAULT_MIN_BLOSUM}')
    parser.add_argument('--max_farep', type=float, default=DEFAULT_MAX_FAREP,
                        help=f'Max allowed value for fa_rep(protein+peptide_seq)-fa_rep(protein+peptide_backbone). Default={DEFAULT_MAX_FAREP}')
    parser.add_argument('--keep_bidentates', action='store_true',
                        help='If enabled, keep the bidentate hbonds residues; otherwise, these residues are mutated to ala')
    parser.add_argument('--pyrosetta_flags', type=str, default=DEFAULT_PYROSETTA_FLAGS,
                        help='Flags for initializing PyRosetta')
    parser.add_argument('--flex_docking_repeats', type=int, default=DEFAULT_FLEX_DOCKING_REPEATS,
                        help=f'Number of repeats for flexible docking protocol. Default={DEFAULT_FLEX_DOCKING_REPEATS}')
    parser.add_argument('--output_csv', type=str, default=DEFAULT_OUTPUT_CSV,
                        help=f'Output CSV file name. Default={DEFAULT_OUTPUT_CSV}')
    args = parser.parse_args()

    main(args)
