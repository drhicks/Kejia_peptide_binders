import torch
import numpy as np
import os
from util import writepdb_multi, writepdb
import chemical

from potentials.manager import PotentialManager
import potentials.potentials as potentials
import inference.bcov_hacks.hack_utils as hack_utils
from inference.bcov_hacks.hack_globals import glo, pyro, ros
from util import generate_Cbeta

from collections import defaultdict
import scipy.spatial

import omegaconf

import copy


class Filter:

    @staticmethod
    def parse_filter_list(filter_list, no_t=False):
        filters = []
        for filt_line in filter_list:
            # this is to make .sh files look nicer
            # you can have empty filters
            if ( filt_line == "" ):
                continue
            filt = Filter.generate_filter(filt_line, no_t=no_t)
            filters.append(filt)
        return filters


    @staticmethod
    def generate_filter(init_string, no_t=False):

        # this should be a static function...
        settings_dict = PotentialManager.parse_potential_string(None, init_string)
        settings_dict['init_string'] = init_string

        if ( no_t ):
            t_range = set()
        else:
            assert('t' in settings_dict)
            t_range = hack_utils.parse_t_range(settings_dict['t'])

        settings_dict['t'] = t_range

        assert(settings_dict['type'] in all_filters)

        return all_filters[settings_dict['type']](**settings_dict)



    def __init__(self, t=set(), prefix="", suffix="", confidence=1, init_string="", pass_expr=None, **kwargs):
        self.t = t
        self.prefix = prefix
        self.suffix = suffix
        self.confidence = confidence
        self.init_string = init_string
        self.pass_expr = pass_expr

    def do_filter(self, binderlen, pose_dict):
        passing, scores = self.filter(binderlen, pose_dict)

        new_scores = {}
        for key in scores:
            new_scores[self.prefix + key + self.suffix] = scores[key]

        if ( self.pass_expr ):
            expr_success = hack_utils.run_eval_on_score_expression(self.pass_expr, new_scores)

            passing = expr_success and passing


        if ( self.confidence < 1 and not passing ):
            passing = np.random.random() > self.confidence
        if ( self.confidence == 0 ):
            passing = True

        return passing, new_scores


    def filter(self, binderlen, pose_dict):
        assert(False)

    def specific_name(self):
        return type(self).__name__ + "_" + self.init_string


class XmlFilter(Filter):


    def __init__(self, style="gly", src='pX0', xml=None, script_vars="", **kwargs):
        super().__init__(**kwargs)


        self.style = style
        self.src = src
        self.xml = xml
        self.script_vars = ""

        assert( not xml is None )
        assert( self.src in ['pX0', 'tX'])


    def filter(self, binderlen, pose_dict):

        pose = AtomsToPose.get_pose_of_style(self.src, self.style, binderlen, pose_dict)

        mover = glo().load_xml(self.xml, self.script_vars).get_mover("ParsedProtocol")

        mover.apply(pose)

        float_map = ros().std.map_std_string_double()
        string_map = ros().std.map_std_string_std_string()
        ros().core.io.raw_data.ScoreMap.add_arbitrary_score_data_from_pose(pose, float_map)
        ros().core.io.raw_data.ScoreMap.add_arbitrary_string_data_from_pose(pose, string_map)

        score_map = {}
        for key in float_map:
            score_map[key] = float_map[key]
        for key in string_map:
            score_map[key] = string_map[key]

        success = mover.get_last_move_status() == ros().protocols.moves.MoverStatus.MS_SUCCESS


        return success, score_map



class SSSchainAtrFilter(Filter):


    def __init__(self, style="gly", src='pX0', real_dssp=True, min_angle=70, min_helix_size=5, **kwargs):
        super().__init__(**kwargs)


        self.style = style
        self.src = src
        self.min_angle = min_angle
        self.min_helix_size = min_helix_size
        self.real_dssp = bool(real_dssp)


    def filter(self, binderlen, pose_dict):

        pose = AtomsToPose.get_pose_of_style(self.src, self.style, binderlen, pose_dict)
        binder = pose.split_by_chain()[1]

        scorefxn = glo().get_rosetta_packer().scorefxn_atr
        oneb_scores, bb_bb_scores, bb_sc_scores, sc_sc_scores = hack_utils.get_oneb_bb_and_sc_pair_scores(binder, scorefxn)

        if ( self.real_dssp ):
            dssp = hack_utils.better_dssp3(pose)
            ss_parts = [x for x in hack_utils.repeating_regions( list(dssp) ) if x[0] != "L"]
        else:
            cas = pose_dict[self.src]['atoms'][:binderlen,1,:]
            is_helix = hack_utils.linear_chunk_helical_dssp( cas, min_helix_size=self.min_helix_size, angle_cutoff=self.min_angle )
            ss_parts = hack_utils.repeating_regions( is_helix, only_keep=True )


        ss_pure_schain_atr = 0
        ss_schain_atr = 0
        ss_atr = 0
        for _, s1, e1 in ss_parts:

            for _, s2, e2 in ss_parts:
                if ( s1 == s2 ):
                    continue

                bb_bb = bb_bb_scores[s1:e1+1,s2:e2+1].sum()/2
                bb_sc = bb_sc_scores[s1:e1+1,s2:e2+1].sum()
                sc_sc = sc_sc_scores[s1:e1+1,s2:e2+1].sum()

                ss_atr += bb_bb + bb_sc + sc_sc
                ss_schain_atr += bb_sc + sc_sc
                ss_pure_schain_atr += sc_sc

        score_map = {}
        score_map['ss_atr'] = ss_atr
        score_map['ss_schain_atr'] = ss_schain_atr
        score_map['ss_pure_schain_atr'] = ss_pure_schain_atr


        return True, score_map



class CorrectNumHelicesFilter(Filter):

    def __init__(self, src='pX0', override_ss_helices=None, x_more_ok=0, x_fewer_ok=0, min_angle=70, min_helix_size=5, **kwargs):
        super().__init__(**kwargs)

        self.src = src
        self.x_more_ok = x_more_ok
        self.x_fewer_ok = x_fewer_ok
        self.min_angle = min_angle
        self.min_helix_size = min_helix_size
        self.override_ss_helices = override_ss_helices

        assert( self.src in ['pX0', 'tX'])


    def filter(self, binderlen, pose_dict):

        ss = None
        if ( self.override_ss_helices is not None ):
            ss_num_helices = self.override_ss_helices
        else:
            ss = pose_dict['hacks'].binder_blockadjacency_ss
            
            ss_num_helices = np.nan
            if ( not ss is None ):
                ss_num_helices = len( hack_utils.repeating_regions( ss, only_keep=0 ) )

        cas = pose_dict[self.src]['atoms'][:binderlen,1,:]
        real_is_helix = hack_utils.linear_chunk_helical_dssp( cas, min_helix_size=self.min_helix_size, angle_cutoff=self.min_angle )

        real_num_helices = len( hack_utils.repeating_regions( real_is_helix, only_keep=True ) )

        extra_helices = real_num_helices - ss_num_helices

        score_dict = dict(num_helices=real_num_helices, extra_helices=extra_helices)

        passing = True

        if ( not np.isnan(ss_num_helices) ):
            if ( extra_helices > 0 ):
                if ( extra_helices > self.x_more_ok ):
                    passing = False
            if ( extra_helices < 0 ):
                if ( extra_helices < -self.x_fewer_ok ):
                    passing = False

        return passing, score_dict


class AdjCorrectFilter(Filter):

    def __init__(self, src='pX0', close_dist=10, far_dist=10, expand_ss=True, min_angle=70, min_helix_size=5, **kwargs):
        super().__init__(**kwargs)

        self.src = src
        self.close_dist = close_dist
        self.far_dist = far_dist
        self.expand_ss = bool(expand_ss)
        self.min_angle = min_angle
        self.min_helix_size = min_helix_size

        assert( self.src in ['pX0', 'tX'])


    def filter(self, binderlen, pose_dict):

        ss = pose_dict['hacks'].binder_blockadjacency_ss
        adj = pose_dict['hacks'].binder_blockadjacency_adj

        cas = pose_dict[self.src]['atoms'][:binderlen,1,:]

        ss_elems, should_be_close_pairs, should_be_far_pairs, conflict = hack_utils.parse_ss_adj_pairs(
                                ss, adj, cas, self.expand_ss, self.min_helix_size, self.min_angle)


        close_correct = torch.zeros(len(should_be_close_pairs), dtype=bool)
        far_correct = torch.zeros(len(should_be_far_pairs), dtype=bool)

        for ipair, (iss1, iss2) in enumerate(should_be_close_pairs):
            _, start1, end1 = ss_elems[iss1]
            _, start2, end2 = ss_elems[iss2]

            cas1 = cas[start1:end1+1]
            cas2 = cas[start2:end2+1]

            closest = torch.min(torch.cdist(cas1, cas2))

            correct = closest <= self.close_dist
            print("Close " + ("correct" if correct else "wrong") + " %i-%i -- %i-%i -- %.1fA"%(start1+1, end1+1, start2+1, end2+1, closest))
            close_correct[ipair] = correct

        for ipair, (iss1, iss2) in enumerate(should_be_far_pairs):
            _, start1, end1 = ss_elems[iss1]
            _, start2, end2 = ss_elems[iss2]

            cas1 = cas[start1:end1+1]
            cas2 = cas[start2:end2+1]

            closest = torch.min(torch.cdist(cas1, cas2))

            correct = closest >= self.far_dist
            print("Far " + ("correct" if correct else "wrong") + " %i-%i -- %i-%i -- %.1fA"%(start1+1, end1+1, start2+1, end2+1, closest))
            far_correct[ipair] = correct


        score_dict = {}
        if ( len(close_correct) > 0 ):
            score_dict['close_correct'] = torch.mean(close_correct.float()).item()
        else:
            score_dict['close_correct'] = 1

        if ( len(far_correct) > 0 ):
            score_dict['far_correct'] = torch.mean(far_correct.float()).item()
        else:
            score_dict['far_correct'] = 1

        passing = not conflict
        return passing, score_dict



class SetBinderNearSetTargetFilter(Filter):

    def __init__(self, src='pX0', setBinder=None, setTarget=None, 
                modes=["min-min","min-median","median-min","com-min","min-com","com-median","median-com","com-com"], 
                **kwargs):
        super().__init__(**kwargs)

        self.src = src
        self.setBinder = setBinder
        self.setTarget = setTarget
        self.modes = modes

        assert(not self.setBinder is None)
        assert(not self.setTarget is None)

        assert( self.src in ['pX0', 'tX'])


    def filter(self, binderlen, pose_dict):

        L = len(pose_dict['pX0']['atoms'])

        binder_mask, binder_pt_strings = parse_ss_location_string(
                        self.setBinder, binderlen, pose_dict['hacks'].binder_blockadjacency_ss, L, True)

        target_mask, target_pt_strings = parse_ss_location_string(
                        self.setTarget, binderlen, pose_dict['hacks'].target_blockadjacency_ss, L, False)


        atoms = pose_dict[self.src]['atoms']
        N  = atoms[:,0]
        Ca = atoms[:,1]
        C  = atoms[:,2]
        Cb = generate_Cbeta(N,Ca,C) # (Lb,3)

        binder_pts = Cb[binder_mask]
        target_pts = Cb[target_mask]

        if ( len(binder_pt_strings) > 0 ):
            binder_pts = torch.cat((binder_pts, torch.stack([potentials.parse_pt_string(x, Ca[:binderlen]) for x in binder_pt_strings])))
        if ( len(target_pt_strings) > 0 ):
            target_pts = torch.cat((target_pts, torch.stack([potentials.parse_pt_string(x, Ca[binderlen:]) for x in target_pt_strings])))


        all_by_dist = torch.cdist( binder_pts, target_pts )

        binder_com = torch.mean( binder_pts, axis=0 )
        target_com = torch.mean( target_pts, axis=0 )

        com_to_target = torch.cdist( binder_com[None,:], target_pts )[0]
        com_to_binder = torch.cdist( target_com[None,:], binder_pts )[0]

        score_dict = {}

        for mode in self.modes:

            if ( mode == "min-min" ):
                score_dict['min-min'] = all_by_dist.min().item()

            elif ( mode == "min-median" ):
                score_dict['min-median'] = torch.median(all_by_dist.min(axis=0).values).item()

            elif ( mode == "median-min" ):
                score_dict['median-min'] = torch.median(all_by_dist.min(axis=1).values).item()

            elif ( mode == "com-min" ):
                score_dict['com-min'] = com_to_target.min().item()

            elif ( mode == "min-com" ):
                score_dict['min-com'] = com_to_binder.min().item()

            elif ( mode == "com-median" ):
                score_dict['com-median'] = torch.median(com_to_target).item()

            elif ( mode == "median-com" ):
                score_dict['median-com'] = torch.median(com_to_binder).item()

            elif ( mode == "com-com" ):
                score_dict['com-com'] = torch.cdist( target_com[None,:], binder_com[None,:] )[0,0].item()


        return True, score_dict



class AtomsInsideBinderFilter(Filter):

    def __init__(self, src='pX0', setBinder="1_-1", setTarget="1_-1", extend_cb_dist=2, **kwargs):
        super().__init__(**kwargs)

        self.src = src
        self.setBinder = setBinder
        self.setTarget = setTarget
        self.extend_cb_dist = extend_cb_dist

        assert(not self.setBinder is None)
        assert(not self.setTarget is None)

        assert( self.src in ['pX0', 'tX'])


    def filter(self, binderlen, pose_dict):

        L = len(pose_dict['pX0']['atoms'])

        binder_mask, binder_pt_strings = parse_ss_location_string(
                        self.setBinder, binderlen, pose_dict['hacks'].binder_blockadjacency_ss, L, True)

        target_mask, target_pt_strings = parse_ss_location_string(
                        self.setTarget, binderlen, pose_dict['hacks'].target_blockadjacency_ss, L, False)


        atoms = pose_dict[self.src]['atoms']
        N  = atoms[:,0]
        Ca = atoms[:,1]
        C  = atoms[:,2]
        O  = atoms[:,3]
        Cb = generate_Cbeta(N,Ca,C) # (Lb,3)

        Ca_to_Cb_unit = Cb - Ca
        Ca_to_Cb_unit /= torch.sqrt( torch.sum( torch.square( Ca_to_Cb_unit ), axis=-1))[:,None]

        extended_cb = Cb + Ca_to_Cb_unit * self.extend_cb_dist

        binder_pts = extended_cb[binder_mask]
        target_pts = torch.cat((N[target_mask], Ca[target_mask], C[target_mask], O[target_mask], Cb[target_mask]))

        if ( len(binder_pt_strings) > 0 ):
            binder_pts = torch.cat((binder_pts, torch.stack([potentials.parse_pt_string(x, Ca[:binderlen]) for x in binder_pt_strings])))
        if ( len(target_pt_strings) > 0 ):
            target_pts = torch.cat((target_pts, torch.stack([potentials.parse_pt_string(x, Ca[binderlen:]) for x in target_pt_strings])))



        hull = scipy.spatial.ConvexHull(binder_pts)
        delaunay = scipy.spatial.Delaunay(hull.points[hull.vertices])
        is_inside = delaunay.find_simplex(target_pts)>= 0

        score_dict = {}
        score_dict['atoms_inside_binder'] = is_inside.sum()

        return True, score_dict




class PercentCoreSCNFilter(Filter):

    def __init__(self, src='pX0', cutoff=None, **kwargs):
        super().__init__(**kwargs)

        self.src = src
        self.cutoff = cutoff

        assert( self.src in ['pX0', 'tX'])


    def filter(self, binderlen, pose_dict):

        atoms = pose_dict[self.src]['atoms']
        N  = atoms[:,0]
        Ca = atoms[:,1]
        C  = atoms[:,2]
        Cb = generate_Cbeta(N,Ca,C) # (Lb,3)

        sc_neigh = hack_utils.sidechain_neighbors( Ca, Cb, Ca )

        is_core = sc_neigh > 5.2



        score_dict = {}
        score_dict['percent_core_scn'] = is_core.float().mean()

        passing = True
        if ( self.cutoff ):
            passing = is_core.mean() > self.cutoff

        return passing, score_dict



class plddtFilter(Filter):

    def __init__(self,  **kwargs):
        super().__init__(**kwargs)


    def filter(self, binderlen, pose_dict):

        score_dict = {}
        score_dict['plddt'] = pose_dict['plddt'].mean().item()
        score_dict['plddt_binder'] = pose_dict['plddt'][:binderlen].mean().item()
        score_dict['plddt_target'] = pose_dict['plddt'][binderlen:].mean().item()


        return True, score_dict



# Allows SS# where # can be negative and is 0 indexed
# Allows #_# where # can be negative and is 1 indexed
def parse_ss_location_string(location_string, binderlen, ss, L, is_binder):

    mask = torch.zeros(L, dtype=bool)
    pt_strings = []

    adder = 0 if is_binder else binderlen
    our_size = binderlen if is_binder else L - binderlen

    if ( not isinstance( location_string, list ) ):
        location_string = [location_string]

    for cst_str in location_string:

        cst_str = str(cst_str)
        if ( cst_str.startswith("SS") ):
            ss_num = int(cst_str.replace("SS", ""))

            assert( not ss is None )

            ss_parts = [ (ss_type, start, end) for (ss_type, start, end) in hack_utils.repeating_regions( ss ) if ss_type <= 1 ]

            if ( ss_num >= 0 ):
                assert(ss_num < len(ss_parts))
            if ( ss_num < 0 ):
                assert( abs(ss_num) <= len(ss_parts))

            _, start, end = ss_parts[ss_num]

            mask[adder+start:adder+end+1] = True

        elif ( cst_str.startswith("T") ):
            pt_strings.append(cst_str)
        else:
            if ( "_" in cst_str ):
                start, end = cst_str.split("_")
                start = int(start)
                end = int(end)
            else:
                start = int(cst_str)
                end = start

            if ( start < 0 ):
                start = our_size + start
                assert( start >= 0)
            else:
                start -= 1
                assert(start >= 0)
            if ( end < 0 ):
                end = our_size + end
                assert(end >= 0)
            else:
                end -= 1
                assert(end >= 0)

            mask[adder+start:adder+end+1] = True

    return mask, pt_strings



class SSConstraint:

    @classmethod
    def generate_constraint(cls, init_string):
        # this should be a static function...
        settings_dict = PotentialManager.parse_potential_string(None, init_string)

        return cls(**settings_dict)

    def __init__(self, t="1_200", cst_pairs=[], mask_loops=True, fill_out_whole_ss=True):

        assert(len(cst_pairs) % 2 == 0)

        self.t = hack_utils.parse_t_range(t)
        self.cst_pairs = cst_pairs
        self.mask_loops = bool(mask_loops)
        self.fill_out_whole_ss = bool(fill_out_whole_ss)



    def cst_to_mask(self, cst_str, binderlen, L, hacks, is_binder=True):


        adder = 0 if is_binder else binderlen
        ss = hacks.binder_blockadjacency_ss if is_binder else hacks.target_blockadjacency_ss

        mask, pt_strings = parse_ss_location_string( cst_str, binderlen, ss, L, is_binder )

        assert(len(pt_strings) == 0)

        if ( self.mask_loops ):
            mask[torch.where((ss == 2) | (ss == 3))[0] + adder] = False

        if ( self.fill_out_whole_ss ):
            ss_parts = [ (ss_type, start, end) for (ss_type, start, end) in hack_utils.repeating_regions( ss ) if ss_type <= 1 ]

            for _, start, end in ss_parts:
                start += adder
                end += adder
                if ( mask[start:end+1].any() ):
                    mask[start:end+1] = True


        return mask




    def do_constraint(self, t, full_ss, full_adj, binderlen, hacks):

        if ( t not in self.t ):
            return

        for ipair in range(len(self.cst_pairs)//2):

            binder_str = self.cst_pairs[ipair*2]
            target_str = self.cst_pairs[ipair*2+1]

            binder_mask = self.cst_to_mask(binder_str, binderlen, len(full_ss), hacks, is_binder=True)
            target_mask = self.cst_to_mask(target_str, binderlen, len(full_ss), hacks, is_binder=False)

            if ( not binder_mask.any() ):
                assert False, "Binder ss str evaluated to nothing"
            if ( not target_mask.any() ):
                assert False, "Target ss str evaluated to nothing"

            full_adj[np.ix_(binder_mask, target_mask)] = torch.tensor([0, 1, 0], dtype=full_adj.dtype)
            full_adj[np.ix_(target_mask, binder_mask)] = torch.tensor([0, 1, 0], dtype=full_adj.dtype)


            # import IPython
            # IPython.embed()



class OutputRange:

    @staticmethod
    def parse_output_range_list(filter_list):
        filters = []
        for filt_line in filter_list:
            # this is to make .sh files look nicer
            # you can have empty filters
            if ( filt_line == "" ):
                continue
            filt = OutputRange.generate_output_range(filt_line)
            filters.append(filt)
        return filters

    @classmethod
    def generate_output_range(cls, init_string):
        # this should be a static function...
        settings_dict = PotentialManager.parse_potential_string(None, init_string)

        return cls(**settings_dict)

    def __init__(self, t=None, highest=True, cutoff=None, n_outputs=1, filters=[], expr="1", top_n_by_multiple=None ):

        self.t = hack_utils.parse_t_range(t)

        self.highest = bool(highest)
        self.cutoff = cutoff
        self.filters = Filter.parse_filter_list(filters, no_t=True)
        self.expr = expr
        self.n_outputs = n_outputs
        self.top_n_by_multiple = top_n_by_multiple

        self.pose_dicts = []
        self.scores = []
        self.score_dicts = []
        self.pose_t = []

    def add_pose(self, binderlen, pose_dict, t):

        success = True
        score_dict = {}
        for filt in self.filters:
            filt_success, scores = filt.do_filter(binderlen, pose_dict)
            if ( not success ):
                return
            for key in scores:
                score_dict[key] = scores[key]

        score = hack_utils.run_eval_on_score_expression(self.expr, score_dict)

        if ( score is None ):
            return

        if ( not self.cutoff is None ):
            if ( self.highest and score < self.cutoff ):
                return
            if ( not self.highest and score > self.cutoff ):
                return

        self.pose_dicts.append(pose_dict)
        self.scores.append(score)
        self.score_dicts.append(score_dict)
        self.pose_t.append(t)


    def write_poses(self, binderlen, hacks):

        if ( len(self.pose_dicts) == 0 ):
            return

        if ( self.top_n_by_multiple ):

            data = np.zeros((len(self.pose_dicts), len(self.top_n_by_multiple)))
            for i_pose in range(len(self.pose_dicts)):
                scores = self.score_dicts[i_pose]

                for i_score, score_name in enumerate(self.top_n_by_multiple):
                    multiplier = 1
                    if ( score_name.startswith("-") ):
                        score_name = score_name[1:]
                        multiplier = -1
                    data[i_pose,i_score] = multiplier * scores[score_name]

            the_order = hack_utils.top_x_by_multiple(data, self.n_outputs)

        else:
            the_order = np.argsort(self.scores)
            if ( self.highest ):
                the_order = the_order[::-1]

        for i_pose in the_order[:self.n_outputs]:
            suffix = "_t%03i"%(self.pose_t[i_pose])

            hacks.write_output(binderlen, self.pose_dicts[i_pose], suffix=suffix, extra_scores=self.score_dicts[i_pose])




class SaveState:

    @staticmethod
    def parse_save_state_list(filter_list):
        filters = []
        for filt_line in filter_list:
            # this is to make .sh files look nicer
            # you can have empty filters
            if ( filt_line == "" ):
                continue
            filt = SaveState.generate_output_range(filt_line)
            filters.append(filt)
        return filters

    @classmethod
    def generate_output_range(cls, init_string):
        # this should be a static function...
        settings_dict = PotentialManager.parse_potential_string(None, init_string)

        return cls(**settings_dict)

    def __init__(self, t=None, write_at_end=False ):

        self.t = hack_utils.parse_t_range(t)
        self.write_at_end = bool(write_at_end)

        self.pending_writes = {}

    def save_state(self, t, seq_t, x_t, seq_init, sampler, hacks):

        d = {}
        d['t'] = t
        d['seq_t'] = seq_t.clone()
        d['x_t'] = x_t.clone()
        d['seq_init'] = seq_init.clone()
        if ( hasattr(sampler, "prev_pred") ):
            d['prev_pred'] = sampler.prev_pred.clone()
        if ( "all_scores" in hacks.all_scores ):
            d['scaff_name'] = hacks.all_scores['scaff_name']

        save_name = hacks.prefix + hacks.tag.replace("/", "_-_") + "_t%03i"%t + "_savestate.pt"

        if ( self.write_at_end ):
            self.pending_writes[save_name] = d
        else:
            with open(save_name, "wb") as f:
                torch.save(d, f)


    def save_pending(self):
        for save_name in self.pending_writes:
            with open(save_name, "wb") as f:
                d = self.pending_writes[save_name]
                torch.save(d, f)

    @staticmethod
    def load_save_state(save_state_path, sampler):
        print("Loading saved state from:", save_state_path)

        d = torch.load(save_state_path)

        if ( 'prev_pred' in d ):
            sampler.prev_pred = d['prev_pred']
        if ( 'scaff_name' in d ):
            ba = self.blockadjacency
            ba.systematic = True
            ba.item = ba.scaffold_list.index(d['scaff_name'])

        return d



class TimeWarp:

    @staticmethod
    def parse_time_warp_list(filter_list):
        filters = []
        for filt_line in filter_list:
            # this is to make .sh files look nicer
            # you can have empty filters
            if ( filt_line == "" ):
                continue
            filt = TimeWarp.generate_time_warp(filt_line)
            filters.append(filt)
        return filters

    @classmethod
    def generate_time_warp(cls, init_string):
        # this should be a static function...
        settings_dict = PotentialManager.parse_potential_string(None, init_string)

        return cls(**settings_dict)

# old default settings
# crunch_factor = 0.5
# explode_loops = 4

# crunch_factor=1 is disabled
    def __init__(self, t=None, src='pX0', fake_frowards=False, fake_origin_t=150, dont_diff_frames=False, crunch_factor=1, explode_loops=0 ):

        self.t = hack_utils.parse_t_range(t)

        lb = np.min(list(self.t))
        ub = np.max(list(self.t))
        for t in range(lb, ub+1):
            self.t.add(t)
        self.t.remove(ub)
        if ( ub == lb ):
            self.t.add(lb)

        self.src = src
        self.fake_frowards = bool(fake_frowards)
        self.fake_origin_t = fake_origin_t
        self.dont_diff_frames = bool(dont_diff_frames)

        self.crunch_factor = crunch_factor
        self.explode_loops = explode_loops

        assert( self.src in ['pX0', 'tX'])


    def warp(self, t, seq_t, x_t, px0, sampler):
        final_t = np.min(list(self.t))

        if ( t != final_t ):
            return True, x_t

        source = x_t if self.src == 'tX' else px0
        assert not source is None, "Can't timewarp on first frame"
        source = source.clone()

        print("================ Time warp to t=%i ================="%final_t)


        ################## Crunch #########################################

        if ( self.crunch_factor < 1 or self.explode_loops > 0):


            ss = hacks().binder_blockadjacency_ss
            adj = hacks().binder_blockadjacency_adj

            cas = source[:,1,:].clone()

            ss_elems, should_be_close_pairs, should_be_far_pairs, conflict, ss_pcas = hack_utils.parse_ss_adj_pairs(
                                    ss, adj, cas[:sampler.binderlen], return_pcas=True)


            overall_pca = hack_utils.mean_pca(np.array(ss_pcas))

            coms = []
            for _, start, end in ss_elems:
                coms.append(torch.mean(cas[start:end+1], axis=0))

            coms = torch.stack(coms, axis=0)

            # potentials.dump_lines(coms.detach().numpy(), np.array(ss_pcas), 5, "pcas_%i.pdb"%t)

            iss_is_paired_with = defaultdict(set)

            for iss1, iss2 in should_be_close_pairs:
                iss_is_paired_with[iss1].add(iss2)
                iss_is_paired_with[iss2].add(iss1)



            # crunch_factor = 0.5
            crunch_factor = self.crunch_factor
            interpolate_loops = True

            # how much to move them at t=100
            # explode_loops = 4
            explode_loops = self.explode_loops

            furtherst_atom = torch.max(torch.linalg.norm(source[:sampler.binderlen,1], axis=-1))
            residuals = sampler.diffuser.get_expected_residual(furtherst_atom).clip(1, None)
            explode_loops *= float(residuals[100-1] / residuals[t-1])
            print("Explode dist %.1i"%explode_loops)

            unassigned = torch.ones(len(cas), dtype=bool)
            unassigned[sampler.binderlen:] = False
            for _, start, end in ss_elems:
                unassigned[start:end+1] = False

            loop_interp_past = torch.zeros(len(cas), dtype=bool)
            is_assigned = torch.where(~unassigned)[0]
            past_assigned_multi = (torch.arange(len(cas))[:,None] - is_assigned[None,:]).float()
            past_assigned_multi[past_assigned_multi < 0] = np.nan
            past_assigned = torch.from_numpy(np.nanmin(past_assigned_multi, axis=-1))

            before_assigned_multi = (is_assigned[None,:] - torch.arange(len(cas))[:,None]).float()
            before_assigned_multi[before_assigned_multi < 0] = np.nan
            before_assigned = torch.from_numpy(np.nanmin(before_assigned_multi, axis=-1))


            frames = []
            frames.append(source.clone())

            # first make helices parallel to motif

            for iss in iss_is_paired_with:
                _, start, end = ss_elems[iss]

                contains_motif = sampler.diffusion_mask.squeeze()[start:end+1].any()
                if ( contains_motif ):
                    continue

                pcas_to_match = []
                for other_iss in iss_is_paired_with[iss]:

                    _, ostart, oend = ss_elems[other_iss]
                    contains_motif = sampler.diffusion_mask.squeeze()[ostart:oend+1].any()
                    if ( not contains_motif ):
                        continue

                    pcas_to_match.append( ss_pcas[other_iss].copy() )

                if ( len(pcas_to_match) == 0 ):
                    continue

                pcas_to_match = np.array(pcas_to_match)


                mean_pca = hack_utils.mean_pca(pcas_to_match)

                # try to make whole scaffold parallel as well
                mean_pca = hack_utils.mean_pca( np.array([overall_pca, mean_pca]))


                ################## now do the rotation around com ############

                our_pca = ss_pcas[iss]
                mean_pca = hack_utils.orient_a_vector_to_b( mean_pca, our_pca )

                rotation_axis = np.cross(our_pca, mean_pca)
                rotation_axis /= np.linalg.norm(rotation_axis)

                our_frame = np.identity(3)
                our_frame[:,0] = our_pca
                our_frame[:,1] = rotation_axis
                our_frame[:,2] = np.cross( our_pca, rotation_axis )

                mean_frame = np.identity(3)
                mean_frame[:,0] = mean_pca
                mean_frame[:,1] = rotation_axis
                mean_frame[:,2] = np.cross( mean_pca, rotation_axis )

                rotation_matrix = torch.from_numpy( mean_frame @ np.linalg.inv( our_frame ) ).float()


                our_com = torch.mean( cas[start:end+1], axis=0 )

                source[start:end+1,:,:] = (rotation_matrix @ (source[start:end+1,:,:] - our_com).transpose(-1,-2)).transpose(-1,-2) + our_com

                if ( interpolate_loops ):
                    weight = torch.zeros(len(cas))
                    if ( start > 0 ):
                        loop_size = int(past_assigned[start-1])
                        loop_start = start-loop_size
                        assert(~is_assigned[loop_start])
                        weight[loop_start:start] = past_assigned[loop_start:start] / ( loop_size + 1 )
                    if ( end < len(cas) - 1 ):
                        loop_size = int(before_assigned[end+1])
                        loop_end = end + loop_size
                        assert(~is_assigned[loop_end])
                        weight[end+1:loop_end+1] = before_assigned[end+1:loop_end+1] / ( loop_size + 1 )

                    mask = (weight > 0) & ~sampler.diffusion_mask.squeeze()
                    full_location = (rotation_matrix @ (source[mask,:,:] - our_com).transpose(-1,-2)).transpose(-1,-2) + our_com
                    delta = full_location - source[mask,:,:]
                    source[mask] += weight[mask][:,None,None] * delta

                frames.append(source.clone())


            favor_template_weight = 1
            upweight_mask = sampler.diffusion_mask.squeeze().float()*(favor_template_weight-1) + 1


            # final = initial^crunch_factor
            # final = (initial^factor2)^(factor2)
            # final = (initial^(factor2^2))
            # crunch_factor = factor2^2
            # crunch_factor ^ 1/2 = factor2

            smash_iters = 2

            crunch_factor = np.power(crunch_factor, 1/smash_iters)
            for smash_loop in range(smash_iters):

                # refresh cas after making helices parallel
                cas = source[:,1,:].clone()

                # next smash helices into each other


                for iss in iss_is_paired_with:
                    _, start, end = ss_elems[iss]

                    contains_motif = sampler.diffusion_mask.squeeze()[start:end+1].any()
                    if ( contains_motif ):
                        continue

                    coms = []

                    other_mask = torch.zeros(len(source), dtype=bool)
                    for other_iss in iss_is_paired_with[iss]:
                        _, ostart, oend = ss_elems[other_iss]
                        other_mask[ostart:oend+1] = True

                        coms.append(torch.mean(cas[ostart:oend+1], axis=0))


                    # pt_in_space = torch.sum( (cas * upweight_mask[:,None])[other_mask], axis=0 ) / upweight_mask[other_mask].sum()
                    pt_in_space = torch.mean(torch.stack(coms, axis=0), axis=0)

                    # pt_in_space = torch.mean( cas[other_mask], axis=0 )

                    cur_com = torch.mean( cas[start:end+1], axis=0 )

                    cur_to_pt = pt_in_space - cur_com
                    distance = torch.sqrt(torch.sum(torch.square(cur_to_pt)))

                    move_unit = cur_to_pt / distance
                    move_dist = distance - torch.pow(distance, crunch_factor)

                    delta = move_dist * move_unit

                    our_mask = torch.zeros((len(source)), dtype=bool)
                    our_mask[start:end+1] = True
                    source[our_mask & ~sampler.diffusion_mask.squeeze()] += delta[None,None,:]

                    if ( interpolate_loops ):
                        weight = torch.zeros(len(cas))
                        if ( start > 0 ):
                            loop_size = int(past_assigned[start-1])
                            loop_start = start-loop_size
                            assert(~is_assigned[loop_start])
                            weight[loop_start:start] = past_assigned[loop_start:start] / ( loop_size + 1 )
                        if ( end < len(cas) - 1 ):
                            loop_size = int(before_assigned[end+1])
                            loop_end = end + loop_size
                            assert(~is_assigned[loop_end])
                            weight[end+1:loop_end+1] = before_assigned[end+1:loop_end+1] / ( loop_size + 1 )

                        source[~sampler.diffusion_mask.squeeze()] += weight[~sampler.diffusion_mask.squeeze()][:,None,None] * delta[None,None,:]

                    frames.append(source.clone())


                iss_is_anti_paired_with = defaultdict(set)

                for iss1, iss2 in should_be_far_pairs:
                    iss_is_anti_paired_with[iss1].add(iss2)
                    iss_is_anti_paired_with[iss2].add(iss1)


                for iss in iss_is_anti_paired_with:
                    _, start, end = ss_elems[iss]

                    contains_motif = sampler.diffusion_mask.squeeze()[start:end+1].any()
                    if ( contains_motif ):
                        continue

                    pushes = []

                    for other_iss in iss_is_anti_paired_with[iss]:
                        _, ostart, oend = ss_elems[other_iss]

                        # only push away from templates
                        if ( ~ torch.any( sampler.diffusion_mask.squeeze()[ostart:oend+1] ) ):
                            continue

                        our_cas = source[start:end+1,1,:]
                        their_cas = source[ostart:oend+1,1,:]

                        all_by_dist = torch.cdist(our_cas, their_cas)
                        closest = all_by_dist.min()

                        if ( closest > 10 ):
                            continue


                        to_push = 10 - closest

                        our_com = torch.mean(our_cas, axis=0)
                        their_com = torch.mean(their_cas, axis=0)

                        push_unit = our_com - their_com
                        push_unit /= torch.sqrt(torch.sum(torch.square(push_unit)))

                        delta = push_unit * to_push

                        pushes.append(delta)

                    if ( len(pushes) == 0 ):
                        continue

                    delta = torch.mean(torch.stack(pushes, axis=0), axis=0)

                    our_mask = torch.zeros((len(source)), dtype=bool)
                    our_mask[start:end+1] = True

                    source[our_mask & ~sampler.diffusion_mask.squeeze()] += delta[None,None,:]

                    if ( interpolate_loops ):
                        weight = torch.zeros(len(cas))
                        if ( start > 0 ):
                            loop_size = int(past_assigned[start-1])
                            loop_start = start-loop_size
                            assert(~is_assigned[loop_start])
                            weight[loop_start:start] =  1 #past_assigned[loop_start:start] / ( loop_size + 1 )
                        if ( end < len(cas) - 1 ):
                            loop_size = int(before_assigned[end+1])
                            loop_end = end + loop_size
                            assert(~is_assigned[loop_end])
                            weight[end+1:loop_end+1] = 1 #before_assigned[end+1:loop_end+1] / ( loop_size + 1 )

                        source[~sampler.diffusion_mask.squeeze()] += weight[~sampler.diffusion_mask.squeeze()][:,None,None] * delta[None,None,:]



                    frames.append(source.clone())

                    new_closest = torch.min(torch.cdist(source[start:end+1,1,:], source[ostart:oend+1,1,:]))

                    print("Push %i-%i from %i-%i Old closest: %.1f  New Closest: %.1f"%(start+1, end+1, ostart+1, oend+1, closest, new_closest))


            if ( explode_loops > 0 ):
                com = torch.mean( source[:sampler.binderlen,1,:], axis=0)
                mask = unassigned & ~sampler.diffusion_mask.squeeze()
                com_to = source[mask,:,:] - com[None,None,:]
                com_to /= torch.linalg.norm(com_to, axis=-1)[:,:,None]

                source[mask,:,:] += com_to * explode_loops

            frames.append(source.clone())

            # potentials.dump_pts(source[:,1,:], "crunch%i.pdb"%t)


            sampler.prev_pred = source.clone().to(sampler.prev_pred.device)[None,...]

            frames = torch.stack(frames, axis=0)

            full_frames = torch.full((frames.shape[0], frames.shape[1], 27, 3), np.nan)
            full_frames[:,:,:14,:] = frames

            # fname = "warp_%i.pdb"%t
            # writepdb_multi( fname, full_frames, torch.ones(len(source)), torch.full((len(source),), chemical.aa2num["GLY"]) )

        ###################################################################


        # the forward diffusion process is too noisy and doesn't produce stuff that looks like an actual diffusion run
        #  just moving stuff towards the origin centered around the "collapse" frame works well
        if ( self.fake_frowards ):
            cas = source[:,1,:]
            frac = 1 - (self.fake_origin_t - t) / self.fake_origin_t

            translations = -cas * frac

            x_t = source.clone()
            x_t[~sampler.diffusion_mask.squeeze(),:,:] += translations[~sampler.diffusion_mask.squeeze()][:,None,:]

        else:
            xyz_mapped = torch.full((len(x_t),27,3), np.nan)
            xyz_mapped[:,:14,:] = source
            atom_mask = torch.isnan(xyz_mapped)

            t_list = np.arange(1, t+1)
            fa_stack, aa_masks, xyz_true = sampler.diffuser.diffuse_pose(
                xyz_mapped,
                torch.clone(seq_t),  # TODO: Check if copy is needed.
                atom_mask.squeeze(),
                diffusion_mask=sampler.diffusion_mask.squeeze(),
                t_list=t_list,
                diffuse_sidechains=sampler.preprocess_conf.sidechain_input,
                include_motif_sidechains=sampler.preprocess_conf.motif_sidechain_input)

            if ( self.dont_diff_frames ):
                ca_deltas = fa_stack[-1].squeeze()[:,1,:] - source[:,1,:]

                x_t = source[:,:,:] + ca_deltas[:,None,:]

            else:
                x_t = fa_stack[-1].squeeze()[:,:14,:]
                x_t = torch.clone(x_t)

        return False, x_t

    def skip_next_diffuser(self, t):

        return t-1 in self.t


def partially_diffuse_to_t(px0, t, seq_t, sampler):
    xyz_mapped = torch.full((len(px0),27,3), np.nan)
    xyz_mapped[:,:14,:] = px0[:,:14,:]
    atom_mask = torch.isnan(xyz_mapped)

    t_list = np.arange(1, t+1)
    fa_stack, aa_masks, xyz_true = sampler.diffuser.diffuse_pose(
        xyz_mapped,
        torch.clone(seq_t),  # TODO: Check if copy is needed.
        atom_mask.squeeze(),
        diffusion_mask=sampler.diffusion_mask.squeeze(),
        t_list=t_list,
        diffuse_sidechains=sampler.preprocess_conf.sidechain_input,
        include_motif_sidechains=sampler.preprocess_conf.motif_sidechain_input)

    x_t = fa_stack[-1].squeeze()[:,:14,:]
    x_t = torch.clone(x_t)

    return x_t

class InitMotifsCloser:

    @staticmethod
    def parse_init_motifs_closer_list(filter_list):
        filters = []
        for filt_line in filter_list:
            # this is to make .sh files look nicer
            # you can have empty filters
            if ( filt_line == "" ):
                continue
            filt = InitMotifsCloser.generate_init_motifs_closer(filt_line)
            filters.append(filt)
        return filters

    @classmethod
    def generate_init_motifs_closer(cls, init_string):
        # this should be a static function...
        settings_dict = PotentialManager.parse_potential_string(None, init_string)

        return cls(**settings_dict)


    def __init__(self, power=0.6 ):

        self.power = power



    def init_motifs_closer(self, x_t, binderlen, diffusion_mask):

        L = len(diffusion_mask)
        is_motif = diffusion_mask.clone()
        is_motif[binderlen:] = False

        cas = x_t[:,1,:]

        # use a wacky trick from init_xyz
        motif_idx = torch.where(is_motif)[0]
        from_motif_seqpos = (torch.arange(L)[:,None] - motif_idx[None,:]).abs()
        closest_motif = torch.argmin(from_motif_seqpos, dim=-1)
        closest_motif_idx = motif_idx[closest_motif]

        closest_motif_idx[diffusion_mask] = -1

        for my_index_is in torch.unique( closest_motif_idx ):
            if ( my_index_is == -1 ):
                continue

            seg_mask = closest_motif_idx == my_index_is
            seg_com = torch.mean( cas[seg_mask], axis=0 )

            motif_pt = cas[my_index_is]
            dist = torch.sqrt(torch.sum(torch.square(motif_pt)))
            actual_dist = torch.pow( dist, self.power )
            goal_com = actual_dist / dist * motif_pt

            x_t[seg_mask] += goal_com - seg_com

            wh = np.where(seg_mask)[0]+1
            frac = np.linalg.norm(goal_com.detach().numpy())/np.linalg.norm(cas[my_index_is].detach().numpy())
            print("Moving %i-%i %.2f of the way to the motif"%(wh[0], wh[-1], frac))


class bcovHacks:


    def __init__(self, conf):
        self.conf = conf

        self.filters = defaultdict(list)

        if ( conf.bcov_hacks.filters ):
            filters = Filter.parse_filter_list(conf.bcov_hacks.filters)
            for filt in filters:
                for t in filt.t:
                    self.filters[t].append(filt)


        self.output_ranges = defaultdict(list)
        self.output_ranges_single = []
        if ( conf.bcov_hacks.output_ranges ):
            self.output_ranges_single = OutputRange.parse_output_range_list(conf.bcov_hacks.output_ranges)
            for ran in self.output_ranges_single:
                for t in ran.t:
                    self.output_ranges[t].append(ran)

        self.save_states = defaultdict(list)
        self.save_states_single = []
        if ( conf.bcov_hacks.save_states ):
            self.save_states_single = SaveState.parse_save_state_list(conf.bcov_hacks.save_states)
            for save in self.save_states_single:
                for t in save.t:
                    self.save_states[t].append(save)

        self.time_warps = []
        if ( conf.bcov_hacks.time_warps ):
            self.time_warps = TimeWarp.parse_time_warp_list(conf.bcov_hacks.time_warps)

            for i in range(len(self.time_warps)):
                for j in range(len(self.time_warps)):
                    if ( i <= j ):
                        continue
                    if ( len( self.time_warps[i].t & self.time_warps[j].t ) > 0 ):
                        assert False, "Overlapping timewarps at t=%s"%str(self.time_warps[i].t & self.time_warps[j].t)

        self.my_init_motifs_closer = []
        if ( conf.bcov_hacks.init_motifs_closer ):
            self.my_init_motifs_closer = InitMotifsCloser.parse_init_motifs_closer_list(conf.bcov_hacks.init_motifs_closer)
            assert len(self.my_init_motifs_closer) <= 1, "It makes no sense to specify init_motifs_closer twice!"


        self.constraints = []
         # surely this will clean up with time
        init_map = [
            (conf.bcov_hacks.ss_constraints, SSConstraint),
        ]
        for source, clas in init_map:
            if ( source ):
                for filt_line in source:
                    self.constraints.append( clas.generate_constraint(filt_line) )
                        

        self.all_scores = {}


        self.binder_blockadjacency_ss = None
        self.binder_blockadjacency_adj = None
        self.target_blockadjacency_ss = None
        self.target_blockadjacency_ss = None


        self.last_rf2_vars = None

        self.rf2_frameskip = None
        if ( conf.bcov_hacks.rf2_frameskip ):
            self.rf2_frameskip = conf.bcov_hacks.rf2_frameskip.replace("_", " ")


        self.prefix = None
        self.tag = None
        self.output_style = conf.bcov_hacks.output_style
        self.silent = conf.bcov_hacks.silent
        self.no_regular_output = conf.bcov_hacks.no_regular_output
        self.no_score_file = conf.bcov_hacks.no_score_file
        self.load_save_state_file = conf.bcov_hacks.load_save_state_file
        self.force_contig = conf.bcov_hacks.force_contig
        self.actually_center_motif = conf.bcov_hacks.actually_center_motif
        self.actually_center_motif2 = conf.bcov_hacks.actually_center_motif2
        self.dont_show_any_input = conf.bcov_hacks.dont_show_any_input
        self.i_know_no_output = conf.bcov_hacks.i_know_no_output

        self.timewarp_subtrajectories = conf.bcov_hacks.timewarp_subtrajectories
        self.really_slow_mpnn_nstruct = conf.bcov_hacks.really_slow_mpnn_nstruct
        self.twpst_unconstrain_binder = conf.bcov_hacks.twpst_unconstrain_binder


        self.set_target_com = None
        if ( conf.bcov_hacks.set_target_com ):
            self.set_target_com = np.array(conf.bcov_hacks.set_target_com, dtype=float)


        self.preserve_center = (self.set_target_com is not None) or (self.actually_center_motif)

######################################################################

# Settings from weird places

#######################################################################

    def set_prefix_tag(self, prefix, tag):
        self.prefix = prefix
        self.tag = tag

        if ( self.silent ):
            self.tag = tag.replace("/", "_-_")

    def set_last_rf2_vars(self, vars):
        self.last_rf2_vars = vars

    def get_last_rf2_vars(self):
        return self.last_rf2_vars

    def _process_ss_adj(self, ss_adj):
        if ( ss_adj is None ):
            return None
        return ss_adj.argmax(axis=-1)

    def set_blockadjacency_info(self, sampler):

        if ( hasattr( sampler, "ss") ):
            self.binder_blockadjacency_ss = self._process_ss_adj( sampler.ss )
        if ( hasattr( sampler, "adj") ):
            self.binder_blockadjacency_adj = self._process_ss_adj( sampler.adj )
        if ( hasattr( sampler, "target_ss") ):
            self.target_blockadjacency_ss = self._process_ss_adj( sampler.target_ss )
        if ( hasattr( sampler, "target_adj") ):
            self.target_blockadjacency_adj = self._process_ss_adj( sampler.target_adj )
        


    def should_skip_rf2_frame(self, t):
        if ( not self.rf2_frameskip ):
            return False
        return eval(self.rf2_frameskip)




########################################################################################

#      pdb manipulation

########################################################################################


    def center_target( self, target_struct, L ):

        if ( self.actually_center_motif2 ):

            is_motif = np.ones(len(target_struct['xyz']), bool)

            for idx, (chain, seqpos) in enumerate(target_struct['pdb_idx']):
                is_motif[idx] = chain != "B"

            assert is_motif.sum() > 0, "actually_center_motif specified but entire target is chain B"

            motif_center = np.mean(target_struct['xyz'][:,1,:][is_motif], axis=0)

            # ok, we're gonna move the last residue super far away such that the motif_center is at the origin

            # current_com = np.mean(target_struct['xyz'][:-1,1,:], axis=0)


            # move the last residue such that the center of mass is at motif_center

            # motif_center = ( np.sum( ca[:-1], axis=0 ) + last_res ) / L
            # motif_center * L =  np.sum( ca[:-1], axis=0 ) + last_res
            # motif_center * L - np.sum( ca[:-1], axis=0 ) = last_res

            L = len(target_struct['xyz'])

            cas = target_struct['xyz'][:,1,:]
            last_ca = cas[-1]
            
            last_ca_needs_to_be = motif_center * L - np.sum( cas[:-1], axis=0 )
            delta = last_ca_needs_to_be - last_ca
            target_struct['xyz'][-1,1,:] += delta


            return


        if ( self.actually_center_motif ):
            

            is_motif = np.ones(len(target_struct['xyz']), bool)

            for idx, (chain, seqpos) in enumerate(target_struct['pdb_idx']):
                is_motif[idx] = chain != "B"

            assert is_motif.sum() > 0, "actually_center_motif specified but entire target is chain B"

            motif_center = np.mean(target_struct['xyz'][:,1,:][is_motif], axis=0)

            target_struct['xyz'] -= motif_center

            return

        if ( self.set_target_com is not None ):

            cas = target_struct['xyz'][:,1,:]
            com = np.mean( cas, axis=0 )

            target_struct['xyz'] -= com

            target_struct['xyz'] += self.set_target_com




########################################################################################

#      Constraints

########################################################################################

    def do_constraints( self, t, full_ss, full_adj, binderlen, sampler ):

        self.set_blockadjacency_info(sampler)

        for constraint in self.constraints:
            constraint.do_constraint( t, full_ss, full_adj, binderlen, self )


    def init_motifs_closer( self, x_t, binderlen, diffusion_mask ):
        if ( self.my_init_motifs_closer ):
            for init in self.my_init_motifs_closer:
                init.init_motifs_closer(x_t, binderlen, diffusion_mask)


########################################################################################

#      Save States

########################################################################################

    def maybe_save_state( self, t, seq_t, x_t, seq_init, sampler ):

        for save_state in self.save_states[t]:
            save_state.save_state(t, seq_t, x_t, seq_init, sampler, self)

    def maybe_load_state( self, sampler ):

        start_t = sampler.t_step_input
        to_ret = None
        if ( not self.load_save_state_file is None ):
            to_ret = SaveState.load_save_state( self.load_save_state_file, sampler)
            start_t = to_ret['t']

        if ( self.no_regular_output ):
            there_will_be_output = False

            end_t = sampler.inf_conf.final_step

            if ( self.output_ranges ):
                for t in range(end_t, start_t+1):
                    if ( len(self.output_ranges[t]) > 0 ):
                        there_will_be_output = True

            # print(end_t, start_t, self.output_ranges)

            if ( not self.i_know_no_output ):
                assert there_will_be_output, ('bcov detects that your run will not produce an output file. Thank me later.'
                                                    + ' Try bcov_hacks.i_know_no_output=True')



        return to_ret



########################################################################################

#      Time Warps

########################################################################################


    def maybe_time_warp( self, t, seq_t, x_t, px0, sampler ):

        we_are_warping = False
        for warp in self.time_warps:
            if ( t in warp.t ):
                we_are_warping, x_t = warp.warp( t, seq_t, x_t, px0, sampler )

        skip_next_diffuser = False
        for warp in self.time_warps:
            skip_next_diffuser = skip_next_diffuser or warp.skip_next_diffuser( t )

        return we_are_warping, x_t, skip_next_diffuser


#########################################################################################

#      Things related to run scoring/filtering

#########################################################################################

    def add_scores_to_run(self, t, scores):
        for key in scores:
            if ( t is None ):
                new_key = key
            else:
                new_key = "t%03i-"%t + key
            self.all_scores[new_key] = scores[key]

    def apply_run_filters(self, t, px0, x_t, seq_t, tors_t, plddt, sampler):

        self.set_blockadjacency_info(sampler)

        success = True
        terminal_message = f't={t}'

        if ( t in self.filters or t in self.output_ranges ):

            pose_dict = AtomsToPose.make_pose_dict(px0, x_t, seq_t, sampler.binderlen, sampler.chain_idx)

            # this is here for the block adjacency info
            # need to think more about that
            pose_dict['hacks'] = self
            pose_dict['plddt'] = plddt

            for filt in self.filters[t]:
                filt_success, scores = filt.do_filter(sampler.binderlen, pose_dict)
                success = success and filt_success
                self.add_scores_to_run(t, scores)
                if ( not filt_success ):
                    terminal_message += ' ' + filt.specific_name()

            for output_range in self.output_ranges[t]:
                output_range.add_pose(sampler.binderlen, pose_dict, t)


        return success, terminal_message


    def write_scorefile(self, suffix="", extra_scores={}):
        tag = self.tag + suffix

        write_header = True #not os.path.exists(fname)

        merged_dict = {}
        for key in self.all_scores:
            merged_dict[key] = self.all_scores[key]
        for key in extra_scores:
            merged_dict[key] = extra_scores[key]

        alpha_keys = sorted( merged_dict.keys() )
        with open(self.prefix + "_score.sc", "a") as f:


            header_parts = []
            parts = []
            for key in alpha_keys:
                value = merged_dict[key]
                if ( hasattr( value, "item" ) ):
                    value = value.item()
                if ( isinstance(value, (int, np.integer)) ):
                    value = "%i"%value
                if ( isinstance(value, (float, np.floating)) ):
                    value = "%.3f"%value

                length = max(max( 8, len(key) ), len(value))
                fmt = "%%%is"%length

                header_parts.append(fmt%key)
                parts.append(fmt%value)

            if ( write_header ):
                f.write("SCORE:     %s description\n"%(" ".join(header_parts)))

            f.write("SCORE:     %s %s\n"%(" ".join(parts), tag))



    def write_output(self, binderlen, pose_dict, suffix="", extra_scores={}, inside_nstruct=False):

        if ( self.really_slow_mpnn_nstruct and not inside_nstruct ):
            for i in range(self.really_slow_mpnn_nstruct ):
                self.write_output(binderlen, pose_dict, suffix=suffix + "_ns%02i"%(i+1), extra_scores=extra_scores, inside_nstruct=True)
            return

        os.makedirs(os.path.dirname(self.prefix), exist_ok=True)

        tag = self.tag + suffix

        if ( self.output_style == "raw" ):
            px0 = pose_dict['pX0']['atoms']
            seq = pose_dict['pX0']['seq']
            chain_idx = pose_dict['pX0']['chain_idx']

            writepdb(tag + ".pdb", px0, seq, binderlen, chain_idx=chain_idx)
        else:
            pose = AtomsToPose.get_pose_of_style( "pX0", self.output_style, binderlen, pose_dict, force_recalculate=inside_nstruct)

            if ( self.silent ):
                silent_name = self.prefix + "_out.silent"
                sfd_out = ros().core.io.silent.SilentFileData( silent_name, False, False, "binary", ros().core.io.silent.SilentFileOptions())
                struct = sfd_out.create_SilentStructOP()
                struct.fill_struct(pose, tag)
                hack_utils.add_dict_to_silent(struct, self.all_scores)
                hack_utils.add_dict_to_silent(struct, extra_scores)
                sfd_out.add_structure(struct)
                sfd_out.write_all(silent_name, False)
            else:
                pose.dump_pdb(tag + ".pdb")


        if ( not self.no_score_file ):
            self.write_scorefile(suffix=suffix, extra_scores=extra_scores)


    def finish_run(self, binderlen, pose_dict):

        if ( not self.no_regular_output ):
            self.write_output(binderlen, pose_dict )

        for output_range in self.output_ranges_single:
            output_range.write_poses(binderlen, self)

        for save_state in self.save_states_single:
            save_state.save_pending()

        return not self.no_regular_output


_hacks = None


def reinit_hacks(conf):
    global _hacks
    _hacks = bcovHacks(conf)


def hacks():
    global _hacks
    return _hacks



class AtomsToPose:

    def __init__(self):
        assert(False)

    @staticmethod
    def make_pose_dict(px0, x_t, seq_t, binderlen, chain_idx):
        pose_dict = {}
        pose_dict['pX0'] = dict(atoms=px0, seq=seq_t, binderlen=binderlen, chain_idx=chain_idx)
        pose_dict['Xt'] = dict(atoms=x_t, seq=seq_t, binderlen=binderlen, chain_idx=chain_idx)
        return pose_dict

    @staticmethod
    def get_pose_of_style(src, style, binderlen, pose_dict, force_recalculate=False):

        lookup = src + "@" + style
        if ( lookup not in pose_dict or force_recalculate ):

            lookup_gly = src + "@" + "gly"

            if ( lookup_gly not in pose_dict ):
                pose_dict[lookup_gly] = AtomsToPose.atoms_to_gly_pose(**pose_dict[src])

            next_pose = pose_dict[lookup_gly].clone()
            AtomsToPose.pack_pose_with_style( next_pose, binderlen, style) 
            pose_dict[lookup] = next_pose

        pose = pose_dict[lookup].clone()
        return pose


    @staticmethod
    def atoms_to_gly_pose(atoms, seq, binderlen, chain_idx):
        # use glycine to prevent rosetta from filling tons of missing atoms
        gly_seq = seq.clone()
        gly_seq[:binderlen] = chemical.aa2num["GLY"]

        pdb_string = writepdb(None, atoms, gly_seq, binderlen, chain_idx=chain_idx)

        pose = pyro().Pose()
        ros().core.import_pose.pose_from_pdbstring(pose, pdb_string)

        return pose

    @staticmethod
    def pack_pose_with_style(pose, binderlen, style):


        if ( style.startswith("mpnn") ):
            mpnn_part, pack_part = style.split("-")
            mpnn_part = mpnn_part.replace("mpnn", "")

            best_of_n = 1
            if ( len(mpnn_part) > 0 ):
                best_of_n = int(mpnn_part)

            ss = ros().std.stringstream()
            pose.dump_pdb(ss)
            pdb_str = ss.str()

            mpnn_seq, score = glo().get_mpnn()(pdb_str, best_of_n=best_of_n)
            print("Mpnn done: %.3f"%score)

            assert(len(mpnn_seq) == binderlen)
            target_seq = pose.sequence()[binderlen:]
            glo().get_rosetta_packer().thread_seq(pose, mpnn_seq + target_seq)

            if ( pack_part == "insta" ):
                glo().get_rosetta_packer().insta_pack(pose)
            elif ( pack_part == "beta" ):
                glo().get_rosetta_packer().beta_pack(pose)
            else:
                assert(False)

            assert(pose.sequence() == mpnn_seq + target_seq)


        elif ( style == "hydrophobic-interface" ):
            glo().get_rosetta_packer().fast_hydrophobic_interface(pose)
        elif ( style == "gly" ):
            pass
        else:
            assert(False)

    @staticmethod
    def atoms_to_packed_pose(atoms, seq, binderlen, chain_idx, style="mpnn-insta" ):

        pose = AtomsToPose.atoms_to_gly_pose(atoms, seq, binderlen, chain_idx)

        AtomsToPose.pack_pose_with_style(pose, binderlen, style)
        return pose






def save_checkpoint(fname, tag, terminal_message):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "a") as f:
        f.write("%s %s\n"%(tag, terminal_message))

def load_checkpoint(fname):
    if ( not os.path.exists(fname) ):
        return dict()

    was_success = {}
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if ( len(line) == 0 ):
                continue
            sp = line.split()

            success = False
            if ( len(sp) >= 2 and sp[1] == "Success" ):
                success = True

            was_success[sp[0]] = success

    return was_success




def load_argreplace(conf):
    if ( not conf.bcov_hacks.arg_replace_file ):
        return [None]

    lines = []
    with open(conf.bcov_hacks.arg_replace_file) as f:
        for line in f:
            line = line.strip()
            if ( len(line) == 0 ):
                continue
            lines.append(line)

    confs = []
    for line in lines:
        if ( line == "None" ):
            confs.append(line)
        else:
            # python doesn't know how to respect "\ " so we help it out
            escaped = line.replace("\\ ", "\0")
            splitted = [x.replace("\0", " ") for x in escaped.split()]

            confs.append(omegaconf.OmegaConf.from_dotlist(splitted))

    return confs



all_filters = {
    'XmlFilter':XmlFilter,
    'CorrectNumHelicesFilter':CorrectNumHelicesFilter,
    'SetBinderNearSetTargetFilter':SetBinderNearSetTargetFilter,
    'AtomsInsideBinderFilter':AtomsInsideBinderFilter,
    'PercentCoreSCNFilter':PercentCoreSCNFilter,
    'SSSchainAtrFilter':SSSchainAtrFilter,
    'plddtFilter':plddtFilter,
    'AdjCorrectFilter':AdjCorrectFilter

}




