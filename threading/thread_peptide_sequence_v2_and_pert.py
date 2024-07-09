import pyrosetta
import sys
import argparse
import blosum62
import pandas as pd

#DRH HACK
from pyrosetta.rosetta.protocols.flexpep_docking import FlexPepDockingProtocol
#DRH HACK

codemap = {'R':'ARG','H':'HIS','K':'LYS','D':'ASP','E':'GLU', \
'S':'SER','T':'THR','N':'ASN','Q':'GLN','C':'CYS', \
'G':'GLY','P':'PRO','A':'ALA','V':'VAL','I':'ILE', \
'L':'LEU','M':'MET','F':'PHE','Y':'TYR','W':'TRP'}


# ===================== bidentate hydrogen bonds ===========================>

#DRH HACK
def pep_flex_dock(in_pose, sf):
    #binder = A
    #peptide target = B
    pose = in_pose.clone()
    fpdock = FlexPepDockingProtocol()
    fpdock.apply(pose)

    return pose

def find_hbonds(pose, search_reslist, target_reslist, skip=[]):
    '''
        skip = [] for each element k:
            hbonds between residue i and residue i+k are skipped (e.g. when k = 2, gamma turns are skipped)
        my_hbond_set = {'bb-bb':[(don_resid, acc_resid, hb_i object)] ...}
    '''
    my_hbond_set = {'bb-bb':[],'bb-sc':[],'sc-bb':[], 'sc-sc':[]}
    my_hbond_list = []
    #sc_pack_list = [] # residues in this list have sc hbonding to bb, and therfore should be kept
    hbond_set = pyrosetta.rosetta.core.scoring.hbonds.HBondSet()
    pose.update_residue_neighbors()  # no need to update as I just scored the pose 
    pyrosetta.rosetta.core.scoring.hbonds.fill_hbond_set(pose, False, hbond_set)
    for i in search_reslist:
        for hb_i in hbond_set.residue_hbonds(i):
            if hb_i.acc_res() in target_reslist or hb_i.don_res() in target_reslist:

                # e.g. skip gamma turn hbond i-i+2
                if len(skip) > 0:
                    skip_check = False
                    for k in skip:
                        if hb_i.acc_res() == hb_i.don_res() + k or hb_i.acc_res() == hb_i.don_res() - k:
                            skip_check = True
                            break
                    if skip_check:
                        continue

                # skip repeat of hbonds (happens when there are internal hb pairs)
                if (hb_i.don_res(), hb_i.acc_res(), hb_i) in my_hbond_list:
                    continue
                my_hbond_list.append( (hb_i.don_res(), hb_i.acc_res(), hb_i) )

                if hb_i.acc_atm_is_backbone() and hb_i.don_hatm_is_backbone():
                    my_hbond_set['bb-bb'].append( (hb_i.don_res(), hb_i.acc_res(), hb_i) )
                elif (hb_i.acc_atm_is_backbone() and hb_i.acc_res() in search_reslist) \
                     or (hb_i.don_hatm_is_backbone() and hb_i.don_res() in search_reslist):
                    #print(hb_i.acc_res(), hb_i.don_res())
                    my_hbond_set['bb-sc'].append( (hb_i.don_res(), hb_i.acc_res(), hb_i) )
                elif (hb_i.acc_atm_is_backbone() and hb_i.acc_res() in target_reslist) \
                     or (hb_i.don_hatm_is_backbone() and hb_i.don_res() in target_reslist):
                    #print(hb_i.acc_res(), hb_i.don_res())
                    my_hbond_set['sc-bb'].append( (hb_i.don_res(), hb_i.acc_res(), hb_i) )
                elif hb_i.acc_atm_is_backbone() or hb_i.don_hatm_is_backbone():
                    #print(hb_i.acc_res(), hb_i.don_res())
                    my_hbond_set['bb-sc'].append( (hb_i.don_res(), hb_i.acc_res(), hb_i) )
                else:
                    my_hbond_set['sc-sc'].append( (hb_i.don_res(), hb_i.acc_res(), hb_i) )

    return my_hbond_list, my_hbond_set

def find_bidentate_hbond(pose, hbond_list):
    '''
        Definition of bidentate hbond: 
            at least two different HEAVY atoms from the same residue participate in hbonds within two different atoms

        hbond_list = [(don_resid, acc_resid, hb_i object), ...]
    '''

    hbonds_by_sc_res = {} # {sidechain_res:[(don_res,don_atom,acc_res,acc_atom)]}
    for hb in hbond_list:

        hb_i = hb[-1] # hb object

        if hb_i.don_res() not in hbonds_by_sc_res:
            hbonds_by_sc_res[hb_i.don_res()] = []
        if hb_i.acc_res() not in hbonds_by_sc_res:
            hbonds_by_sc_res[hb_i.acc_res()] = []

        hb_info = ( hb_i.don_res(), pose.residue(hb_i.don_res()).atom_name(hb_i.don_hatm()).strip(), \
                    hb_i.acc_res(), pose.residue(hb_i.acc_res()).atom_name(hb_i.acc_atm()).strip() ) # don_res, don_atm_name, acc_res, acc_atm_name
        if not hb_i.don_hatm_is_backbone():
            hbonds_by_sc_res[hb_i.don_res()].append( hb_info )
        if not hb_i.acc_atm_is_backbone():
            hbonds_by_sc_res[hb_i.acc_res()].append( hb_info )

    bidentate_hbonds_by_sc_res = {}
    for res in hbonds_by_sc_res:
        if len(hbonds_by_sc_res[res]) > 1:
            bidentate_hbonds_by_sc_res[res] = hbonds_by_sc_res[res]
    return bidentate_hbonds_by_sc_res


def simple_hbond_finder(pose, reslist1, reslist2, delta_HA=3., delta_theta=30., 
                        reslist1_atom_type=['bb','sc'], reslist2_atom_type=['bb','sc'],verbose=False):
    '''
    input:
        pose
        reslist1
        reslist2 (can overlap with reslist1)
        delta_HA: max distance cutoff
        delta_theta: max angle deviation cutoff
        reslist1/reslist2_type: type of atom to be considered
    output:
        list of hbond atom pairs: [(don_res1,don_atm1,accpt_res2,accpt_atm2,delta_HA,theta),(),()...]
    ''' 
    
    def _get_don_accpt_residue_atom_indices(pose, reslist, atom_type=['bb','sc']):
        '''
        don_list = [(resid,Hpol_atomid,base_atomid),()..]
        accpt_list = [(resid,accpt_pos,base_atomid),()..]
        '''
        don_list, accpt_list = [], []
        for resid in reslist:
            bb_list = [x for x in range(1,pose.residue(resid).natoms()+1) if pose.residue(resid).atom_is_backbone(x)]
            for atomid in pose.residue(resid).Hpol_index():
                if ('bb' not in atom_type and atomid in bb_list) or ('sc' not in atom_type and atomid not in bb_list):
                    continue
                don_list.append( (resid, atomid, pose.residue(resid).atom_base(atomid)) )
            for atomid in pose.residue(resid).accpt_pos():
                if ('bb' not in atom_type and atomid in bb_list) or ('sc' not in atom_type and atomid not in bb_list):
                    continue
                accpt_list.append( (resid, atomid, pose.residue(resid).atom_base(atomid)) )
        return don_list, accpt_list
    
    reslist1_don, reslist1_accpt = _get_don_accpt_residue_atom_indices(pose, reslist1, reslist1_atom_type)
    reslist2_don, reslist2_accpt = _get_don_accpt_residue_atom_indices(pose, reslist2, reslist2_atom_type)
    #print(reslist1_don)
    #print(reslist1_accpt)
    #print(reslist2_don)
    #print(reslist2_accpt)
    
    hbonds = []
    
    
    for don in reslist1_don:
        for accpt in reslist2_accpt:
            dist = pyrosetta.rosetta.numeric.xyzVector_double_t.distance(pose.residue(don[0]).xyz(don[1]), pose.residue(accpt[0]).xyz(accpt[1]))
            # angle defined by 3 atom xyzs
            angle = pyrosetta.rosetta.numeric.angle_degrees_double( pose.residue(don[0]).xyz(don[2]), pose.residue(don[0]).xyz(don[1]), pose.residue(accpt[0]).xyz(accpt[1]))
            # angle defined by vectors (not working as no reference angle for this)
            #don_vec = np.array( pose.residue(don[0]).xyz(don[1]) - pose.residue(don[0]).xyz(don[2]) )
            #accpt_vec = np.array( pose.residue(accpt[0]).xyz(accpt[1]) - pose.residue(accpt[0]).xyz(accpt[2]) )
            #angle = np.arccos(np.dot(don_vec, accpt_vec) / (np.linalg.norm(don_vec) * np.linalg.norm(accpt_vec)))
            #angle = angle*180/np.pi
            if verbose:
                print(don, pose.residue(don[0]).atom_name(don[1]).strip(), accpt, pose.residue(accpt[0]).atom_name(accpt[1]).strip(), dist,angle)
            if dist <= delta_HA and angle >= 180-delta_theta:
                # always write out reslist1 first
                hbonds.append( (don[0],don[1],accpt[0],accpt[1],dist,angle,'don-accpt') )
    for don in reslist2_don:
        for accpt in reslist1_accpt:
            dist = pyrosetta.rosetta.numeric.xyzVector_double_t.distance(pose.residue(don[0]).xyz(don[1]), pose.residue(accpt[0]).xyz(accpt[1]))
            # angle defined by 3 atom xyzs
            angle = pyrosetta.rosetta.numeric.angle_degrees_double( pose.residue(don[0]).xyz(don[2]), pose.residue(don[0]).xyz(don[1]), pose.residue(accpt[0]).xyz(accpt[1]))
            # angle defined by vectors (not working as no reference angle for this)
            #don_vec = np.array( pose.residue(don[0]).xyz(don[1]) - pose.residue(don[0]).xyz(don[2]) )
            #accpt_vec = np.array( pose.residue(accpt[0]).xyz(accpt[1]) - pose.residue(accpt[0]).xyz(accpt[2]) )
            #angle = np.arccos(np.dot(don_vec, accpt_vec) / (np.linalg.norm(don_vec) * np.linalg.norm(accpt_vec)))
            #angle = angle*180/np.pi            
            if verbose:
                print(don, pose.residue(don[0]).atom_name(don[1]).strip(), accpt, pose.residue(accpt[0]).atom_name(accpt[1]).strip(), dist,angle)
            if dist <= delta_HA and angle >= 180-delta_theta:
                # always write out reslist1 first
                hbonds.append( (accpt[0],accpt[1],don[0],don[1],dist,angle,'accpt-don') )               

                
    return hbonds

def find_potential_bidentate_hbond(pose, reslist1, reslist2, delta_HA=3., delta_theta=30.,
                                   reslist1_atom_type=['sc'], reslist2_atom_type=['bb']):
    hbonds = simple_hbond_finder(pose, reslist1, reslist2, delta_HA=delta_HA, delta_theta=delta_theta,
                                 reslist1_atom_type=reslist1_atom_type, reslist2_atom_type=reslist2_atom_type)
    hbonds_by_sc_res = {} # {sidechain_res:[(don_res,don_atom,acc_res,acc_atom,distance,angle)]}
    for hb in hbonds:
        if hb[0] not in hbonds_by_sc_res:
            hbonds_by_sc_res[hb[0]] = []
        if hb[6] == 'don-accpt':
            hbonds_by_sc_res[hb[0]].append( (hb[0], pose.residue(hb[0]).atom_name(hb[1]).strip(), hb[2], pose.residue(hb[2]).atom_name(hb[3]).strip(), hb[4], hb[5]) ) 
        elif hb[6] == 'accpt-don':
            hbonds_by_sc_res[hb[0]].append( (hb[2], pose.residue(hb[2]).atom_name(hb[3]).strip(), hb[0], pose.residue(hb[0]).atom_name(hb[1]).strip(), hb[4], hb[5]) )
        else:
            print('Error: incorrect donor acceptor info: {}'.fomrat(hb[6]))
        
    bidentate_hbonds_by_sc_res = {}
    for res in hbonds_by_sc_res:
        if len(hbonds_by_sc_res[res]) > 1:
            bidentate_hbonds_by_sc_res[res] = hbonds_by_sc_res[res]
    return bidentate_hbonds_by_sc_res

def check_fully_satisfied_bidenates(pose, bidentates, bidentate_resid):

    this_residue = pose.residue(bidentate_resid)
    sc_heavy_atms = {}

    hpol_map = {}
    for hpol in this_residue.Hpol_index():
        if not this_residue.atom_is_backbone(hpol):
            #print(hpol, this_residue.atom_name(hpol))
            hpol_map[hpol] = False

    for atom_id in range(1, len(this_residue.atoms())+1):
        atm_name = this_residue.atom_name(atom_id).strip()
        if 'H' not in atm_name:
            if 'N' in atm_name or 'O' in atm_name: # heavy atoms
                if not this_residue.atom_is_backbone(atom_id): # avoid backbone 
                    hb_group = [atm_name]
                    for hpol in hpol_map:
                        if not hpol_map[hpol] and hpol in this_residue.bonded_neighbor(atom_id):
                            hb_group.append(this_residue.atom_name(hpol).strip())
                            hpol_map[hpol] = True
                    sc_heavy_atms[tuple(hb_group)] = False
    #print(sc_heavy_atms)

    for hb in bidentates[bidentate_resid]:
        for hb_i in [0,2]: # don, acc        
            if hb[hb_i] == bidentate_resid:
                for hb_group in sc_heavy_atms:
                    if hb[hb_i+1] in hb_group:
                        sc_heavy_atms[hb_group] = True

    #print(bidentate_resid, sc_heavy_atms)

    for hb_group in sc_heavy_atms:
        if sc_heavy_atms[hb_group] == False:
            return False

    return True


# <===================== bidentate hydrogen bonds ===========================


def _config_my_task_factory_for_repack(added_residues):
    repack_residues = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
    for res_num in added_residues:
        repack_residues.append_index(res_num)
    # repack
    restrict_to_repack = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        pyrosetta.rosetta.core.pack.task.operation.RestrictToRepackingRLT(), repack_residues, False)  # not flip the selection
    freeze_rest = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT(), repack_residues, True)
    # add task ops to a TaskFactory
    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
    tf.push_back(restrict_to_repack)
    tf.push_back(freeze_rest)
    return tf

def repack_pose(pose, sf, reslist):
    pack_rotamers = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(sf)
    pack_rotamers.task_factory(_config_my_task_factory_for_repack(reslist))
    pack_rotamers.apply(pose)
    return pose

def setup_movemap(pose, bblist=[], chilist=[]):
    #print('doing setup_movemap')
    mm = pyrosetta.rosetta.core.kinematics.MoveMap()
    if len(bblist) == 0 and len(chilist) == 0:
        mm.set_chi( True )
        mm.set_bb( True )        
    else:
        mm.set_chi( False )
        mm.set_bb( False )
        for resid in range(1, pose.size()+1):
            if resid in bblist:
                mm.set_bb( resid, True )
                mm.set_chi( resid, True )
            elif resid in chilist:
                mm.set_chi( resid, True )
    mm.set_jump( True )
    #mm.set_bb ( pose.size(), False ) # # for the virtual residue?
    #mm.set_chi( pose.size(), False ) # for the virtual residue?
    #print('finished setup_movemap')
    return mm

def minimize_pose(pose, sf, reslist, cartesian_=False, coordcst_=False):
    #print('doing minimize_pose')
    #movemap = setup_movemap(pose, bblist=[], chilist=reslist)
    movemap = setup_movemap(pose, bblist=reslist, chilist=reslist)
    if coordcst_:
        sf.set_weight(pyrosetta.rosetta.core.scoring.coordinate_constraint, 1.0)
        pyrosetta.rosetta.core.scoring.constraints.add_coordinate_constraints(pose, 0.1, False)
    
    use_nblist = True
    deriv_check = True
    deriv_check_verbose = False
    min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover(movemap,sf,'lbfgs_armijo_nonmonotone',0.01,use_nblist,
                                                                          deriv_check,deriv_check_verbose)
    min_mover.max_iter(100)
    #min_mover.min_type('lbfgs_armijo_nonmonotone')
    if cartesian_:
        min_mover.cartesian(True)
    min_mover.apply( pose )
    return pose

def compute_seq_identity(seq1, seq2):
    assert(len(seq1)==len(seq2))
    ic = 0
    for i in range(len(seq1)):
        if seq1[i]==seq2[i]:
            ic += 1
    return ic*1.0/len(seq1)


parser = argparse.ArgumentParser()
parser.add_argument('--scaffold', type=str,
                    help='a single scaffold')
parser.add_argument('--peptide_header', type=str,
                    help='peptide_header') 
parser.add_argument('--peptide_seq', type=str,
                    help='peptide_seq')
parser.add_argument('--interface_dist_cutoff', type=float, default=8.0,
                    help='distance cutoff for neighborhood selector for interface residues. Default=8.0') 
parser.add_argument('--min_seq_identity', type=float, default=0.0,
                    help='min seq identity required between threaded sequence and original peptide sequence in the scaffold. Default=0.0') 
parser.add_argument('--min_blosum', type=float, default=-10,
                    help='min blosum score required between threaded sequence and original peptide sequence in the scaffold. Default=-10') 
parser.add_argument('--max_farep', type=float, default=300.0,
                    help='max allowed value for farep(protein+peptide_seq)-farep(protein+peptide_backbone). Default=300.0') 
parser.add_argument('--keep_bidentates', action='store_true',
                    help='if enabled, keep the bidentate hbonds residues; otherwise, these residues are mutated to ala')

args = parser.parse_args()    

pyrosetta.init('-pep_refine -ex1 -ex2aro -restore_talaris_behavior -default_max_cycles 100 \
        -dunbrack_prob_buried 0.8 -dunbrack_prob_nonburied 0.8 -dunbrack_prob_buried_semi 0.8 \
        -dunbrack_prob_nonburied_semi 0.8 -boost_fa_atr False -rep_ramp_cycles 5 -mcm_cycles 5') #DRH HACK -mute all

#these defaults were
#boost_fa_atr True
#rep_ramp_cycles 10
#mcm_cycles 8


#pyrosetta.init()
sf = pyrosetta.get_score_function()
sf_cart = sf.clone()
sf_cart.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.pro_close, 0.0) 
sf_cart.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded, 0.5) 
sf_farep = sf.clone()
for st in sf_farep.get_nonzero_weighted_scoretypes():
    sf_farep.set_weight(st, 0)
sf_farep.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.fa_rep, 1)   


peptides = {}
peptides[args.peptide_header] = args.peptide_seq


terms = ['pdb','scaffold','scaffold_bidentates','target_peptide','scaffold_peptide_start','scaffold_peptide_end',
                    'target_peptide_start','target_peptide_end','fa_rep','sequence_identity','blosum_score']
data = {x:[] for x in terms}
scaffolds = []
for scaffold in [args.scaffold]:

    scaffold_file = scaffold.strip()
    #print(scaffold_file)    
    scaffold_name = scaffold_file.split('/')[-1].replace('.pdb','')
    input_pose = pyrosetta.pose_from_file(scaffold_file)
    binder_size = input_pose.split_by_chain(1).size()
    pose = input_pose.clone()

    hbond_list, hbond_set = find_hbonds(input_pose, list(range(1, input_pose.split_by_chain(1).size()+1)), 
                                                    list(range(input_pose.split_by_chain(1).size()+1, input_pose.size()+1)))
    bidentates = find_bidentate_hbond(input_pose, hbond_set['sc-bb'])
    #print(bidentates)

    # prepare scaffolds (mutate interface residues to ala)
    peptide_reslist = list(range(binder_size+1, pose.size()+1))
    #print(peptide_reslist)
    peptide_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
    for res_num in peptide_reslist:
        peptide_selector.append_index(res_num)
    interface_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(
        peptide_selector, args.interface_dist_cutoff, True)
    interface_vec = interface_selector.apply(pose)
    interface_reslist = [i+1 for i, j in enumerate(interface_vec) if j]
    #print('+'.join([str(x) for x in interface_reslist]))
    for resi in interface_reslist:
        if args.keep_bidentates and resi in bidentates:
            continue
        #mt = pyrosetta.rosetta.protocols.simple_moves.MutateResidue(resi, 'ALA')
        #mt.apply(pose)
    scaffolds.append([pose, input_pose, interface_reslist, scaffold_file])

    bb_farep = sf_farep(pose)
    #print(sf_farep(pose))


    # thread sequence
    for pep in peptides:

        print(f'threading {pep} onto {scaffold_file}')

        pose_thread_list = []
        this_peptide_reslist = peptide_reslist.copy()
        this_ref_bidentates = bidentates.copy()


        if len(peptides[pep]) < len(peptide_reslist):
            for start_pos in range(len(peptide_reslist)-len(peptides[pep])+1):
                scaffold_seq = input_pose.sequence(binder_size+start_pos+1, binder_size+start_pos+len(peptides[pep]))
                thread_seq = peptides[pep]
                seq_id = compute_seq_identity(scaffold_seq, thread_seq)
                blosum_score = blosum62.compute_blosum62_score(scaffold_seq, thread_seq)
                if seq_id >= args.min_seq_identity and blosum_score >= args.min_blosum:
                    pose_thread = pose.clone()
                    for resid in range(len(peptides[pep])):
                        mt = pyrosetta.rosetta.protocols.simple_moves.MutateResidue(binder_size+start_pos+resid+1, codemap[peptides[pep][resid]])
                        mt.apply(pose_thread)
                    deleted_reslist = []
                    if binder_size+start_pos+len(peptides[pep]) < pose_thread.size():
                        pose_thread.delete_residue_range_slow(binder_size+start_pos+len(peptides[pep])+1, pose_thread.size())
                        deleted_reslist += list(range(binder_size+start_pos+len(peptides[pep])+1, pose_thread.size()+1))
                    if start_pos > 0:
                        pose_thread.delete_residue_range_slow(binder_size+1, binder_size+start_pos)
                        deleted_reslist += list(range(binder_size+1, binder_size+start_pos+1))
                    #pose_thread.dump_pdb(f'{scaffold_name}_{start_pos}.pdb')

                    # update peptide_reslist and bidentate dictionary accordingly
                    this_peptide_reslist = list(range(binder_size+1, pose_thread.size()+1))
                    for sc in bidentates:
                        for bbresi in [bidentates[sc][0][0], bidentates[sc][0][2], bidentates[sc][1][0], bidentates[sc][1][2]]:
                            if bbresi in deleted_reslist and sc in this_ref_bidentates:
                                this_ref_bidentates.pop(sc)
                    #print(start_pos)
                    #print(deleted_reslist)
                    #print(this_ref_bidentates)


                    pose_thread_list.append([pose_thread, [start_pos+1, start_pos+len(peptides[pep])], [1, len(peptides[pep])], 
                                            seq_id, blosum_score])


        else:
            for start_pos in range(len(peptides[pep])-len(peptide_reslist)+1):
                scaffold_seq = input_pose.sequence(binder_size+1, binder_size+len(peptide_reslist))
                thread_seq = peptides[pep][start_pos:start_pos+len(peptide_reslist)]
                seq_id = compute_seq_identity(scaffold_seq, thread_seq)
                blosum_score = blosum62.compute_blosum62_score(scaffold_seq, thread_seq)
                if seq_id >= args.min_seq_identity and blosum_score >= args.min_blosum:
                    pose_thread = pose.clone()
                    for resid in range(len(peptide_reslist)):
                        mt = pyrosetta.rosetta.protocols.simple_moves.MutateResidue(binder_size+resid+1, codemap[peptides[pep][start_pos+resid]])
                        mt.apply(pose_thread)
                    #pose_thread.dump_pdb(f'{scaffold_name}_{start_pos}.pdb')

                    pose_thread_list.append([pose_thread, [1, len(peptide_reslist)], [start_pos+1, start_pos+len(peptide_reslist)],
                                            seq_id, blosum_score])


        #print(pose_thread_list)

        # repack and min
        pose_packmin_list = []
        for pt_id, pt in enumerate(pose_thread_list):
            pt[0] = repack_pose(pt[0], sf, this_peptide_reslist)
            pt[0] = minimize_pose(pt[0], sf.clone(), this_peptide_reslist, coordcst_=True)

            #DRH HACK
            for pert in range(0, 10):

                pt[0] = pep_flex_dock(pt[0], sf)

                #pt[0] = minimize_pose(pt[0], sf_cart, peptide_reslist, cartesian_=True)

                #pt[0].dump_pdb(f'{scaffold_name}_{pt[1][0]}_{pt[1][1]}_{pt[2][0]}_{pt[2][1]}.pdb')
                #print(f'{scaffold_name}_{pt[1][0]}_{pt[1][1]}_{pt[2][0]}_{pt[2][1]}.pdb:   {sf_farep(pt[0])}')

                if sf_farep(pt[0])-bb_farep <= args.max_farep:

                    if args.keep_bidentates:
                        this_hbond_list, this_hbond_set = find_hbonds(pt[0], list(range(1, pt[0].split_by_chain(1).size()+1)), 
                                                        list(range(pt[0].split_by_chain(1).size()+1, pt[0].size()+1)))
                        this_bidentates = find_bidentate_hbond(pt[0], this_hbond_set['sc-bb'])
                        bidentate_check = True
                        for bidentate_resi in this_ref_bidentates:
                            if bidentate_resi not in this_bidentates:
                                bidentate_check = False 
                                #print('bidentate check fail: ', pt_id, bidentate_resi)
                                break
                        if not bidentate_check:
                            continue

                    pose_packmin_list.append(pt+[sf_farep(pt[0])-bb_farep])

                #else:
                #    print('farep fail: ', pt_id)


        for ipt, pt in enumerate(pose_packmin_list):
            #print(pp)
            pt[0].dump_pdb(f'{scaffold_name}_{pep}_{pt[1][0]}_{pt[1][1]}_{pt[2][0]}_{pt[2][1]}_{ipt}.pdb')
            data['pdb'].append(f'{scaffold_name}_{pt[1][0]}_{pt[1][1]}_{pt[2][0]}_{pt[2][1]}_{ipt}.pdb')
            data['scaffold'].append(scaffold_name)
            data['scaffold_bidentates'].append('_'.join([str(x) for x in sorted(list(this_ref_bidentates.keys()))]))
            data['target_peptide'].append(pep)
            data['scaffold_peptide_start'].append(pt[1][0])
            data['scaffold_peptide_end'].append(pt[1][1])
            data['target_peptide_start'].append(pt[2][0])
            data['target_peptide_end'].append(pt[2][1])
            data['fa_rep'].append(pt[-1])
            data['sequence_identity'].append(pt[-3])
            data['blosum_score'].append(pt[-2])

df = pd.DataFrame(data, columns=terms)
df.to_csv('thread_peptide_sequence_stats.csv', index=False)
