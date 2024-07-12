import sys, os
import numpy as np
import pyrosetta 
from collections import OrderedDict

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

pyrosetta.init( '-in:file:silent_struct_type binary -beta_nov16 -holes:dalphaball /home/bcov/dev_rosetta/main/source/external/DAlpahBall/DAlphaBall.gcc')

alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
states = len(alpha_1)
alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
           'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']

aa_1_N = {a:n for n,a in enumerate(alpha_1)}
aa_3_N = {a:n for n,a in enumerate(alpha_3)}
aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}

fr_cart_fast_xml = f'''
<SCOREFXNS>
    <ScoreFunction name="sfxn" weights="beta_nov16" />
    <ScoreFunction name="sfxn_pack" weights="beta_nov16_cart" >
        <Reweight scoretype="coordinate_constraint" weight="1.0" />
        <Reweight scoretype="cart_bonded" weight="0.75" />
    </ScoreFunction>
    <ScoreFunction name="sfxn_min" weights="beta_nov16_cart" >
        <Reweight scoretype="coordinate_constraint" weight="0.1" />
        <Reweight scoretype="cart_bonded" weight="0.75" />
    </ScoreFunction>
</SCOREFXNS>

<RESIDUE_SELECTORS>
    <Chain name="chainA" chains="A"/>
    <Chain name="chainB" chains="B"/>
    <Neighborhood name="interface_chA" selector="chainB" distance="14.0" />
    <Neighborhood name="interface_chB" selector="chainA" distance="14.0" />
    <And name="AB_interface" selectors="interface_chA,interface_chB" />
    <Not name="Not_interface" selector="AB_interface" />

    <Slice name="chainA_last_res" indices="-1" selector="chainA" />
    <Slice name="chainB_first_res" indices="1" selector="chainB" />

    <And name="chainB_not_interface" selectors="Not_interface,chainB" />

    <And name="chainB_fixed" >
        <Or selectors="chainB_not_interface" />
    </And>
    <And name="chainB_not_fixed" selectors="chainB">
        <Not selector="chainB_fixed"/>
    </And>

    <True name="all" />

</RESIDUE_SELECTORS>

<TASKOPERATIONS>
    <IncludeCurrent name="current" />
    <ExtraRotamersGeneric name="ex1_ex2" ex1="1" ex2="1" />
    <LimitAromaChi2 name="limitchi2" chi2max="110" chi2min="70" include_trp="True" />

    <OperateOnResidueSubset name="restrict2repacking" selector="all">
        <RestrictToRepackingRLT/>
    </OperateOnResidueSubset>

    <OperateOnResidueSubset name="restrict_target_not_interface" selector="chainB_fixed">
        <PreventRepackingRLT/>
    </OperateOnResidueSubset>

</TASKOPERATIONS>

<MOVERS>
    <SwitchChainOrder name="chain1only" chain_order="1" />
    <SwitchChainOrder name="chain2only" chain_order="2" />
</MOVERS>
<SIMPLE_METRICS>
    <SapScoreMetric name="sap_score" score_selector="chainA" sap_calculate_selector="chainA" sasa_selector="chainA" />
</SIMPLE_METRICS>
<FILTERS>
    <ContactMolecularSurface name="contact_molecular_surface" distance_weight="0.5" target_selector="chainB" binder_selector="chainA" confidence="0" />
    <Ddg name="ddg_no_repack"  threshold="-10" jump="1" repeats="1" repack="0" confidence="0" scorefxn="sfxn" />
</FILTERS>

<MOVERS>

    <ModifyVariantType name="remove_lower_terminus" remove_type="LOWER_TERMINUS_VARIANT,UPPER_TERMINUS_VARIANT" residue_selector="chainA_last_res" />
    <ModifyVariantType name="remove_upper_terminus" remove_type="UPPER_TERMINUS_VARIANT,LOWER_TERMINUS_VARIANT" residue_selector="chainB_first_res" />
    <AddChainBreak name="add_break" find_automatically="1" distance_cutoff="4"/>

    <PackRotamersMover name="cst_pack" scorefxn="sfxn_pack" task_operations="current,restrict2repacking,restrict_target_not_interface"/>
    <MinMover name="cart_min" max_iter="200" type="lbfgs_armijo_nonmonotone" tolerance="0.01" 
    cartesian="false" bondangle="true" bondlength="true" jump="1" bb="1" chi="1" scorefxn="sfxn_min" >
        <MoveMap name="MM"  >
            <Chain number="1" chi="true" bb="true" />
            <ResidueSelector selector="chainB_fixed" chi="false" bb="false" />
            <ResidueSelector selector="chainB_not_fixed" chi="true" bb="false" />
        </MoveMap>
    </MinMover>
</MOVERS>
'''

#dfpmin_armijo_nonmonotone, lbfgs_armijo_nonmonotone

chainA = pyrosetta.rosetta.core.select.residue_selector.ChainSelector("A")
chainB = pyrosetta.rosetta.core.select.residue_selector.ChainSelector("B")

objs = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(fr_cart_fast_xml)

cst_pack = objs.get_mover("cst_pack")
cart_min = objs.get_mover("cart_min")

termini1 = objs.get_mover("remove_lower_terminus")
termini2 = objs.get_mover("remove_upper_terminus")
add_break = objs.get_mover("add_break")

def add2scorefile(tag, scorefilename, write_header=False, score_dict=None):
        with open(scorefilename, "a") as f:
                add_to_score_file_open(tag, f, write_header, score_dict)

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

def pose_from_silent(sfd_in, tag):
    pose = pyrosetta.Pose()
    sfd_in.get_structure(tag).fill_pose(pose)
    return pose

def get_filter_by_name(filtername):
    try:
        the_filter = objs.get_filter(filtername)
    except:
        the_filter = objs.get_simple_metric(filtername)
    # Get rid of stochastic filter
    if ( isinstance(the_filter, pyrosetta.rosetta.protocols.filters.StochasticFilter) ):
        the_filter = the_filter.subfilter()

    return the_filter

def filter_to_results(pose, filtername):
    this_filter = get_filter_by_name(filtername)
    if (isinstance(this_filter, pyrosetta.rosetta.protocols.simple_filters.ShapeComplementarityFilter)):
        value = this_filter.compute(pose)
    else:
        value = this_filter.report_sm(pose)
    return value

def score_with_this_filter(pose, this_filter):
        return filter_to_results(pose, this_filter)

# harmonic near center. 0.98*max_val at radius
def my_topout_func(center, max_val, radius):
    limit = radius/2
    return pyrosetta.rosetta.core.scoring.func.TopOutFunc(max_val/limit**2, center, limit)


def generate_csts(pose, other_pose, subset, correct_rotamer_bonus, max_dev, visualize=True):

    ambiguous_pairs = {
        'E':[('OE1', 'OE2')],
        'D':[("OD1", 'OD2')],
        'F':[("CD1", 'CD2'),("CE1", 'CE2')],
        'Y':[("CD1", 'CD2'),("CE1", 'CE2')],
    }


    my_starts = []
    my_stops = []


    cst_set = pyrosetta.rosetta.utility.vector1_std_shared_ptr_const_core_scoring_constraints_Constraint_t()

    for seqpos in range(1, pose.size()+1):

        if ( not subset[seqpos] ):
            continue

        res = pose.residue(seqpos)
        other_res = other_pose.residue(seqpos)

        assert(res.name1() == other_res.name1())

        num_sc_atoms = res.nheavyatoms() - res.first_sidechain_atom() + 1

        this_weight = correct_rotamer_bonus / num_sc_atoms

        my_pairs = []
        if ( res.name1() in ambiguous_pairs ):
            my_pairs = ambiguous_pairs[res.name1()]

        # cs

        for i_sc_atom in range(num_sc_atoms):
            our_atom = res.first_sidechain_atom() + i_sc_atom

            if ( res.atom_type(our_atom).is_virtual() ):
                continue

            our_name = res.atom_name(our_atom)
            other_atom = other_res.atom_index(our_name)

            goal_xyz = other_res.xyz(other_atom)

            atom_id = pyrosetta.rosetta.core.id.AtomID( our_atom, seqpos )
            root = pyrosetta.rosetta.core.id.AtomID( 1, pose.size() )

            cst = pyrosetta.rosetta.core.scoring.constraints.CoordinateConstraint( atom_id, root, goal_xyz, my_topout_func(0, this_weight, max_dev) )

            for my_pair in my_pairs:
                if ( our_name in my_pair ):
                    our_amb_atom_name = my_pair[1-my_pair.index(our_name)]

                    amb_cst = pyrosetta.rosetta.core.scoring.constraints.AmbiguousConstraint()
                    amb_cst.add_individual_constraint(cst)

                    amb_goal_xyz = other_res.xyz(our_amb_atom_name)

                    cst = pyrosetta.rosetta.core.scoring.constraints.CoordinateConstraint( atom_id, root, amb_goal_xyz, my_topout_func(0, this_weight, max_dev) )

                    amb_cst.add_individual_constraint(cst)

                    cst = amb_cst

            cst_set.append(cst)

    return cst_set

def get_sap(pose):
    sap = pyrosetta.rosetta.core.pack.guidance_scoreterms.sap.SapScoreMetric()
    return sap.calculate(pose)

def relax_and_ddg(pose):
    correct_rotamer_bonus = 30
    max_dev = 2
    
    monomer_cst_set = generate_csts(pose, pose, chainA.apply(pose), correct_rotamer_bonus, max_dev, visualize=False)
    pose.add_constraints(monomer_cst_set)

    target_cst_set = generate_csts(pose, pose, chainB.apply(pose), correct_rotamer_bonus, max_dev, visualize=False)
    pose.add_constraints(target_cst_set)

    termini1.apply(pose)
    termini2.apply(pose)
    add_break.apply(pose)

    cst_pack.apply(pose)
    cart_min.apply(pose)
    cst_pack.apply(pose)
    cart_min.apply(pose)
    
    filters = ["ddg_no_repack", "contact_molecular_surface"]
    filter_scores = {}
    for this_filter in filters:
        this_score = score_with_this_filter(pose, this_filter)
        filter_scores[this_filter] = this_score

    filter_scores["sap_score"] = get_sap(pose.split_by_chain()[1])

    filter_scores["ddg_per_sap"] = filter_scores["ddg_no_repack"] / filter_scores["sap_score"]

    return pose, filter_scores

# If the script is executed directly, run a test or example function
if __name__ == "__main__":
    silent = sys.argv[1]

    #create silent out, scorefile, and checkpoint if needed
    sfd_out = pyrosetta.rosetta.core.io.silent.SilentFileData("out.silent", False, False, "binary", pyrosetta.rosetta.core.io.silent.SilentFileOptions())
    checkpoint_filename = "check.point"
    scorefilename = "out.sc"

    write_header = not os.path.exists(scorefilename)

    finished_structs = determine_finished_structs(checkpoint_filename)

    sfd_in = pyrosetta.rosetta.core.io.silent.SilentFileData(pyrosetta.rosetta.core.io.silent.SilentFileOptions())
    sfd_in.read_file(silent)

    silent_index = silent_tools.get_silent_index(silent)
    for raw_jobname in silent_index['tags']:

        pose = pose_from_silent(sfd_in, raw_jobname)
        pose, filter_scores = relax_and_ddg(pose)
        add2silent(f"{raw_jobname}_min", pose, filter_scores, sfd_out)
        add2scorefile(f"{raw_jobname}_min", scorefilename, write_header=write_header, score_dict=filter_scores)
        write_header = False
