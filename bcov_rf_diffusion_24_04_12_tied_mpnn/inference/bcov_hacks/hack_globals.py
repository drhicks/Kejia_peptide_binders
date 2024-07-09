


import functools
import os


class TrueGlobals:

    def __init__(self):
        self.xml_obj_cache = {}
        self.pyrosetta = None
        self.rosetta_packer = None
        self.mpnn = None
        self.abego_man = None
        pass

    def get_abego_man(self):
        if ( self.abego_man is None ):
            self.abego_man = ros().core.sequence.ABEGOManager()

        return self.abego_man

    def get_rosetta_packer(self):
        if ( self.rosetta_packer is None ):
            self.rosetta_packer = RosettaPacker()
        return self.rosetta_packer

    # don't import pyrosetta unless someone wants it
    def get_pyrosetta(self):
        if ( self.pyrosetta is None ):
            import pyrosetta
            import pyrosetta.rosetta

            flags_that_make_rosetta_load_faster = ("-chemical:exclude_patches LowerDNA UpperDNA Cterm_amidation"
                                                   " SpecialRotamer VirtualBB ShoveBB VirtualDNAPhosphate VirtualNTerm"
                                                   " CTermConnect sc_orbitals pro_hydroxylated_case1 pro_hydroxylated_case2"
                                                   " ser_phosphorylated thr_phosphorylated tyr_phosphorylated tyr_sulfated"
                                                   " lys_dimethylated lys_monomethylated  lys_trimethylated lys_acetylated"
                                                   " glu_carboxylated cys_acetylated tyr_diiodinated N_acetylated C_methylamidated"
                                                   " MethylatedProteinCterm"
                                                   )


            pyrosetta.init("-mute all -beta_nov16 " + flags_that_make_rosetta_load_faster)
            self.pyrosetta = pyrosetta
        return self.pyrosetta

    def load_xml(self, xml, script_var_string):
        lookup = xml + "@" + script_var_string
        if ( lookup in self.xml_obj_cache ):
            return self.xml_obj_cache[lookup]

        # you can actually change options inside rosetta, they just don't always get refreshed
        #  in this case, setting parser:script_vars is ok
        script_vars = ros().utility.vector1_std_string()
        for var_pair in script_var_string.split("~"):
            if ( len(var_pair) == 0 ):
                continue
            script_vars.append(var_pair)
        ros().basic.options.set_string_vector_option("parser:script_vars", script_vars)

        objs = ros().protocols.rosetta_scripts.XmlObjects().create_from_file(os.path.expanduser(xml))
        self.xml_obj_cache[lookup] = objs
        return objs

    # pass a pdb string to the function this returns to get a sequence
    def get_mpnn(self):
        if ( self.mpnn ):
            return self.mpnn

        import inference.bcov_hacks.hacked_protein_mpnn_run as hacked_protein_mpnn_run
        argparser = hacked_protein_mpnn_run.load_argparser()
        args = argparser.parse_args(("--pdb_path_chains A --out_folder ./ --path_to_model_weights= --omit_AAs C").split())

        self.mpnn = functools.partial(hacked_protein_mpnn_run.main, args)
        return self.mpnn


_true_globals = TrueGlobals()

def glo():
    global _true_globals
    return _true_globals

def pyro():
    return glo().get_pyrosetta()

def ros():
    return glo().get_pyrosetta().rosetta







# don't base these on xmls so that there's no loading time
class RosettaPacker:

    def __init__(self):

        self.chainA = ros().core.select.residue_selector.ChainSelector("A")
        self.chainB = ros().core.select.residue_selector.ChainSelector("B")
        self.interface_on_A = ros().core.select.residue_selector.NeighborhoodResidueSelector(self.chainB, 10.0, False)
        self.interface_on_B = ros().core.select.residue_selector.NeighborhoodResidueSelector(self.chainA, 10.0, False)
        self.AB_interface = ros().core.select.residue_selector.OrResidueSelector( self.interface_on_A, self.interface_on_B )
        self.Not_interface = ros().core.select.residue_selector.NotResidueSelector( self.AB_interface )
        self.chainA_not_interface = ros().core.select.residue_selector.AndResidueSelector( self.Not_interface, self.chainA )
        self.chainB_not_interface = ros().core.select.residue_selector.AndResidueSelector( self.Not_interface, self.chainB )

        self.scorefxn_insta = pyro().get_fa_scorefxn()
        for term in self.scorefxn_insta.get_nonzero_weighted_scoretypes():
            name = ros().core.scoring.name_from_score_type(term)

            if ( "_dun" in name ):
                continue
            if ( "rama" in name ):
                continue
            if ( "p_aa_pp" in name ):
                continue
            if ( "fa_rep" in name ):
                continue
            if ( "fa_atr" in name ):
                continue
            if ( "fa_sol" in name ):
                continue
            if ( "hbond" in name ):
                continue
            if ( "pro_close" in name ):
                continue

            self.scorefxn_insta.set_weight(term, 0)

        self.scorefxn_insta_soft = self.scorefxn_insta.clone()
        self.scorefxn_insta_soft.set_weight(ros().core.scoring.fa_rep, 0.15)


        self.scorefxn_none = ros().core.scoring.ScoreFunctionFactory.create_score_function("none")
        self.scorefxn_atr = ros().core.scoring.ScoreFunctionFactory.create_score_function("none")
        self.scorefxn_atr.set_weight(ros().core.scoring.fa_atr, 1)
        self.scorefxn_beta = pyro().get_fa_scorefxn()


    # pack with only dunbrack, vdw, and hbonds
    #  elec is super slow so we can't have that
    def insta_pack(self, pose):

        tf = ros().core.pack.task.TaskFactory()
        tf.push_back( ros().core.pack.task.operation.RestrictToRepacking() )
        tf.push_back( ros().core.pack.task.operation.OperateOnResidueSubset( ros().core.pack.task.operation.PreventRepackingRLT(),
                          self.chainB_not_interface, False ))
        tf.push_back( ros().core.pack.task.operation.IncludeCurrent() )

        packer = ros().protocols.minimization_packing.PackRotamersMover()
        packer.score_function( self.scorefxn_insta )
        packer.task_factory( tf )

        packer.apply( pose )

    def beta_pack(self, pose):

        tf = ros().core.pack.task.TaskFactory()
        tf.push_back( ros().core.pack.task.operation.RestrictToRepacking() )
        tf.push_back( ros().core.pack.task.operation.OperateOnResidueSubset( ros().core.pack.task.operation.PreventRepackingRLT(),
                          self.chainB_not_interface, False ))
        tf.push_back( ros().core.pack.task.operation.IncludeCurrent() )

        packer = ros().protocols.minimization_packing.PackRotamersMover()
        packer.score_function( self.scorefxn_beta )
        packer.task_factory( tf )

        packer.apply( pose )


    # use the packer to do this to prevent issues with disulfides and to ensure we get variant types right
    def thread_seq(self, pose, new_seq):

        old_seq = pose.sequence()

        locked_subset = ros().utility.vector1_bool( pose.size() )

        tf = ros().core.pack.task.TaskFactory()
        for seqpos in range(1, pose.size()+1 ):
            old_letter = old_seq[seqpos-1]
            new_letter = new_seq[seqpos-1]

            if ( old_letter == new_letter ):
                locked_subset[seqpos] = True
                continue

            restrict_aa = ros().core.pack.task.operation.RestrictAbsentCanonicalAASRLT()
            restrict_aa.aas_to_keep( new_letter )

            subset = ros().utility.vector1_bool( pose.size() )
            subset[seqpos] = True
            tf.push_back( ros().core.pack.task.operation.OperateOnResidueSubset( restrict_aa, subset ) )

        tf.push_back( ros().core.pack.task.operation.OperateOnResidueSubset( ros().core.pack.task.operation.PreventRepackingRLT(),
                          locked_subset ) )

        packer = ros().protocols.minimization_packing.PackRotamersMover()
        packer.score_function( self.scorefxn_none )
        packer.task_factory( tf )

        packer.apply( pose )


    def fast_hydrophobic_interface(self, pose):

        tf = ros().core.pack.task.TaskFactory()
        tf.push_back( ros().core.pack.task.operation.OperateOnResidueSubset( ros().core.pack.task.operation.PreventRepackingRLT(),
                          self.chainB, False ))
        tf.push_back( ros().core.pack.task.operation.OperateOnResidueSubset( ros().core.pack.task.operation.PreventRepackingRLT(),
                          self.chainA_not_interface, False ))
        tf.push_back( ros().core.pack.task.operation.IncludeCurrent() )

        restrict_aa = ros().core.pack.task.operation.RestrictAbsentCanonicalAASRLT()
        restrict_aa.aas_to_keep( "GAFILMVW" )
        tf.push_back( ros().core.pack.task.operation.OperateOnResidueSubset( restrict_aa,
                          self.interface_on_A, False ))

        packer = ros().protocols.minimization_packing.PackRotamersMover()
        packer.score_function( self.scorefxn_insta_soft )
        packer.task_factory( tf )

        packer.apply( pose )



