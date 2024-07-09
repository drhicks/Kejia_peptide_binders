import torch
import numpy as np
import ast

import itertools
from collections import defaultdict

import sklearn.decomposition
import scipy.stats


from inference.bcov_hacks.hack_globals import glo, pyro, ros



def mean_pca(pcas):

    all_by_dot = np.sum( pcas[:,None] * pcas[None,:] )

    imost_parallel = np.argmax( np.abs(all_by_dot.sum(axis=0) ) )

    most_parallel = pcas[imost_parallel]

    for i in range(len(pcas)):
        pcas[i] = orient_a_vector_to_b( pcas[i], most_parallel )

    assert( np.all( np.sum( most_parallel[None,:] * pcas, axis=-1 ) >= 0 ) )

    mean_pca = np.mean(pcas, axis=0)
    mean_pca /= np.linalg.norm(mean_pca)

    return mean_pca


def parse_ss_adj_pairs(ss, adj, cas, expand_ss=True, min_helix_size=5, angle_cutoff=70, return_pcas=False):

        
    assert( torch.allclose( adj, adj.T ) )


    ss_elems = [ x for x in repeating_regions( ss ) if x[0] <= 1 ]

    # print("SS elems: ", ss_elems)
    res_has_pca = np.full((len(cas), 3), np.nan)

    if ( expand_ss ):
        real_is_helix = linear_chunk_helical_dssp( cas, min_helix_size=min_helix_size, angle_cutoff=angle_cutoff )

        real_ss_elems = repeating_regions( real_is_helix, only_keep=True )
        # print("Real SS elems: ", real_ss_elems)

        whoami = np.ones(len(ss), dtype=int)*-1
        for ireal, (_, start, end) in enumerate(real_ss_elems):
            whoami[start:end+1] = ireal

        real_belongs_to = defaultdict(list)
        for iss, (_, start, end) in enumerate(ss_elems):
            for ireal in np.unique(whoami[start:end+1]):
                if ( ireal == -1 ):
                    continue
                real_belongs_to[ireal].append(iss)

        for ireal in real_belongs_to:
            isss = real_belongs_to[ireal]
            if ( len(isss) == 1 ):
                iss = isss[0]
                ss_elems[iss][1] = min(ss_elems[iss][1], real_ss_elems[ireal][1])
                ss_elems[iss][2] = max(ss_elems[iss][2], real_ss_elems[ireal][2])
            else:
                # multiple helices have merged, divide unused space evenly
                _, real_start, real_end = real_ss_elems[ireal]
                isss = list(sorted(isss))

                # assign first and last directly
                ss_elems[isss[0]][1] = min(ss_elems[isss[0]][1], real_start)
                ss_elems[isss[-1]][2] = max(ss_elems[isss[-1]][2], real_end)

                # all of the space between these two must be assigned
                for i_junct in range(len(isss)-1):
                    end1 =   ss_elems[isss[i_junct  ]][2]
                    start2 = ss_elems[isss[i_junct+1]][1]
                    center = (end1 + start2)//2

                    # using this method, the previous segment wins ties
                    # gaps that were 0-size get assigned back to the starting values
                    ss_elems[isss[i_junct  ]][2] = center
                    ss_elems[isss[i_junct+1]][1] = center + 1


        if ( return_pcas ):
            # use the real ss_elems to get pcas because they're likely longer
            for _, start, end in real_ss_elems:
                print("Real: %i-%i"%(start+1,end+1))
                this_pca = get_pca_and_orient_forwards( cas[start:end+1] )
                res_has_pca[start:end+1,:] = this_pca 


    # print("SS elems:", ss_elems)

    adj_says_something = torch.ones((len(ss), len(ss)), dtype=adj.dtype)*2
    is_ss_by_ss = (ss == 0) | (ss == 1)

    # the adj table only operates on helices and sheets. loops and mask are ignored
    adj_says_something[np.ix_(is_ss_by_ss,is_ss_by_ss)] = adj[np.ix_(is_ss_by_ss,is_ss_by_ss)]


    should_be_close_pairs = []
    should_be_far_pairs = []

    conflict = False

    for iss1, (_, start1, end1) in enumerate(ss_elems):
        for iss2, (_, start2, end2) in enumerate(ss_elems):
            if ( iss2 <= iss1 ):
                continue

            our_space = adj_says_something[start1:end1+1,start2:end2+1]

            should_be_close = (our_space == 1).any()
            should_be_far = (our_space == 0).any()

            if ( should_be_close and should_be_far ):
                print("AdjCorrectnessFilter reports conflict at. %i-%i, %i-%i"%(start1, end1, start2, end2))
                print("SS_elems:", ss_elems)
                conflict = True
            else:

                if ( should_be_close ):
                    should_be_close_pairs.append([iss1, iss2])
                if ( should_be_far ):
                    should_be_far_pairs.append([iss1, iss2])


    pcas = []
    if ( return_pcas ):
        for _, start, end in ss_elems:
            real_pca_at_res = res_has_pca[start:end+1].copy()
            is_null = np.isnan(real_pca_at_res[:,0])
            if ( np.all( is_null ) ):
                this_pca = get_pca_and_orient_forwards( cas[start:end+1] ).detach().numpy()
            else:
                this_pca = np.mean( real_pca_at_res[~is_null], axis=0 )
                this_pca /= np.linalg.norm(this_pca)

            pcas.append(this_pca)




    if ( return_pcas ):
        return ss_elems, should_be_close_pairs, should_be_far_pairs, conflict, pcas
    else:
        return ss_elems, should_be_close_pairs, should_be_far_pairs, conflict


# Uses a pareto-front style method to take the top x by multiple different values
# The general idea is to rank all values in every list (with the highest being best)
#   and then say: "If I take the top 1 best from each, do I have enough?"
#                 "If I take the top 2 best from each, do I have enough?"
#      Where when taking the top n from each, the process uses & logic, so there may actually
#          be 0 elements when taking the top 1 best from each
#
#   Each row is a design, each column is a scoreterm. Higher is better
#
#   Return value is indices to keep
def top_x_by_multiple(data, x):

    desired_num = x

    # returns indices
    tags = np.arange(len(data))

    # shuffle data so that in perfect ties, the output is a random subsample
    to_shuf = np.arange(0, len(tags), dtype=np.uint32)
    np.random.shuffle(to_shuf)
    tags = tags[to_shuf]
    data = data[to_shuf]
    total = len(data)

    # sort of like argsort for each scoreterm
    ranked = scipy.stats.rankdata(data, axis=0)

    bigger_2 = 0
    for i in range(1000):
        bigger_2 = 2**i
        if ( bigger_2 // 2 > total):
            break

    percentiles = np.linspace(0, 1, bigger_2)

    space_size = bigger_2
    next_cut = bigger_2 // 2 - 1

    remaining = 0
    cutoff = total

    last_mask_above_threshold = np.ones(total, np.bool)

    # binary search
    # This should never reach that many but it's better than while True
    for i in range(10000):


        # this is the actual ranking process
        # take top X in each argsort and make sure they're in all top Xs
        eval_percentile = (percentiles[next_cut] + percentiles[next_cut+1]) / 2
        cutoff = eval_percentile * total

        mask = np.ones(total, np.bool)

        for icol in range(data.shape[-1]):
            mask &= ranked[:,icol] >= cutoff


        remaining = mask.sum()

        if ( remaining >= desired_num ):
            last_mask_above_threshold = mask
        if ( remaining == desired_num ):
            break

        space_size //= 2

        if ( space_size == 1 ):
            break

        if ( remaining < desired_num ):
            next_cut -= space_size // 2
        else:
            next_cut += space_size // 2

    mask = last_mask_above_threshold

    keep_tags = tags[mask]
    # keep_data = data[mask]
    keep_ranked = ranked[mask]


    # not a perfect ranking. It's why we don't just use this at the start
    silly_score = keep_ranked.sum(axis=-1)

    # We need to do this because in extreme ties, we can output way more than we wanted
    final_sort = np.argsort(-silly_score)[:desired_num]

    
    return keep_tags[final_sort]








def get_abego(pose, seqpos):
    abego_man = glo().get_abego_man()
    return abego_man.index2symbol(abego_man.torsion2index_level1( pose.phi(seqpos), pose.psi(seqpos), pose.omega(seqpos)))

def get_consensus(letters):
    counts = defaultdict(lambda : 0, {})
    for letter in letters:
        counts[letter] += 1

    maxx_letter = 0
    maxx = 0
    for key in counts:
        if ( counts[key] > maxx ):
            maxx = counts[key]
            maxx_letter = key
    return maxx_letter


# this is 0 indexed with the start and end with loops converted to nearby dssp
# and HHHHHH turns identified
def better_dssp3(pose, length=-1, force_consensus=None, consensus_size=6):
    if ( length < 0 ):
        length = pose.size()

    dssp = ros().core.scoring.dssp.Dssp(pose)
    dssp.dssp_reduced()
    the_dssp = "x" + dssp.get_dssp_secstruct()[:length]
    the_dssp = list(the_dssp)

    n_consensus = get_consensus(the_dssp[3:consensus_size+1])
    if ( not force_consensus is None ):
        n_consensus = force_consensus

    for i in range(1, consensus_size+1):
        the_dssp[i] = n_consensus

    c_consensus = get_consensus(the_dssp[-(consensus_size):-2])
    if ( not force_consensus is None ):
        c_consensus = force_consensus

    for i in range(1, consensus_size+1):
        the_dssp[-i] = c_consensus

    the_dssp = "".join(the_dssp)

    # print(the_dssp)

    my_dssp = "x"

    for seqpos in range(1, length+1):
        abego = get_abego(pose, seqpos)
        this_dssp = the_dssp[seqpos]
        if ( the_dssp[seqpos] == "H" and abego != "A" ):
            # print("!!!!!!!!!! Dssp - abego mismatch: %i %s %s !!!!!!!!!!!!!!!"%(seqpos, the_dssp[seqpos], abego))

            # This is the Helix-turn-helix HHHH case. See the test_scaffs folder
            if ( (abego == "B" or abego == "E") and seqpos > consensus_size and seqpos < len(the_dssp)-consensus_size ):
                this_dssp = "L"

        my_dssp += this_dssp

    # print(my_dssp)

    return my_dssp[1:]




def get_oneb_bb_and_sc_pair_scores(pose, scorefxn):

    size = pose.size()

    full_score = scorefxn(pose)

    oneb_scores = np.zeros(size)
    bb_bb_scores = np.zeros((size, size))
    bb_sc_scores = np.zeros((size, size))
    sc_sc_scores = np.zeros((size, size))


    for seqpos in range(1, size+1):
        oneb_scores[seqpos-1] = pose.energies().onebody_energies(seqpos).dot(scorefxn.weights())

    for seqpos1 in range(1, size+1):
        for seqpos2 in range(1, size+1):
            if ( seqpos1 == seqpos2 ):
                continue
            res1 = pose.residue(seqpos1)
            res2 = pose.residue(seqpos2)

            bb_bb_emap = ros().core.scoring.EMapVector()
            scorefxn.eval_ci_2b_bb_bb(res1, res2, pose, bb_bb_emap)
            scorefxn.eval_cd_2b_bb_bb(res1, res2, pose, bb_bb_emap)

            bb_bb_scores[seqpos1-1, seqpos2-1] = bb_bb_emap.dot(scorefxn.weights())

            bb_sc_emap = ros().core.scoring.EMapVector()
            scorefxn.eval_ci_2b_bb_sc(res1, res2, pose, bb_sc_emap)
            scorefxn.eval_cd_2b_bb_sc(res1, res2, pose, bb_sc_emap)

            bb_sc_scores[seqpos1-1, seqpos2-1] = bb_sc_emap.dot(scorefxn.weights())

            sc_sc_emap = ros().core.scoring.EMapVector()
            scorefxn.eval_ci_2b_sc_sc(res1, res2, pose, sc_sc_emap)
            scorefxn.eval_cd_2b_sc_sc(res1, res2, pose, sc_sc_emap)

            sc_sc_scores[seqpos1-1, seqpos2-1] = sc_sc_emap.dot(scorefxn.weights())



    assert(np.allclose( sc_sc_scores, sc_sc_scores.T ))
    assert(np.allclose( bb_bb_scores, bb_bb_scores.T ))

    my_score = oneb_scores.sum() + bb_bb_scores.sum()/2 + sc_sc_scores.sum()/2 + bb_sc_scores.sum()


    assert(np.isclose(my_score, full_score))


    return oneb_scores, bb_bb_scores, bb_sc_scores, sc_sc_scores

# rosetta/main/source/src/core/select/util/SelectResiduesByLayer.cc
def sidechain_neighbors(binder_Ca, binder_Cb, else_Ca):

    conevect = binder_Cb - binder_Ca
    conevect /= torch.sqrt(torch.sum(torch.square(conevect), axis=-1))[:,None]

    vect = else_Ca[:,None] - binder_Cb[None,:]
    vect_lengths = torch.sqrt(torch.sum(torch.square(vect), axis=-1))
    vect_normalized = vect / vect_lengths[:,:,None]

    dist_term = 1 / ( 1 + torch.exp( vect_lengths - 9  ) )

    angle_term = (((conevect[None,:] * vect_normalized).sum(axis=-1) + 0.5) / 1.5).clip(0, None)

    sc_neigh = (dist_term * np.square( angle_term )).sum(axis=0)

    return sc_neigh


def run_eval_on_score_expression(expr, score_dict):
    # we replace the score terms in expr with their values
    og_expr = expr
    keys_long_to_short = sorted(score_dict.keys(), key=lambda x: -len(x))

    for key in keys_long_to_short:
        expr = expr.replace(key, str(score_dict[key]))

    try:
        result = eval(expr)
    except:
        print("Error in score eval:")
        print("     Eval statement: ", og_expr)
        print("  Tried to evaluate: ", expr)
        return None

    return result

def add_dict_to_silent(struct, d):
    for key in d:
        value = d[key]
        if ( isinstance(value, str) ):
            struct.add_string_value(key, value)
        else:
            struct.add_energy(key, value)


# _ means range
# , means comma
def parse_t_range(t_range):

    if ( t_range is None ):
        return None

    this_range = set()

    if ( not isinstance(t_range, str) ):
        this_range.add(int(t_range))
        return this_range

    for part in t_range.split(","):
        if ( "_" in part ):
            start, end = part.split("_")
            start = int(start)
            end = int(end)
            if ( end < start ):
                tmp = end
                end = start
                start = tmp
            for elem in range(start, end+1):
                this_range.add(elem)
        else:
            this_range.add(int(part))

    return this_range


# returns regions of repeating elements
# (value, start, stop)
# stop is the last element of the region for slicing using stop+1
def repeating_regions(vector, only_keep=None):
    offset = 0
    regions = []
    for value, group in itertools.groupby(vector):
        this_len = len(list(group))
        next_offset = offset + this_len
        if ( only_keep is None or only_keep == value ):
            regions.append( [value, offset, next_offset-1])
        offset = next_offset

    return regions

def orient_a_vector_to_b(a, b):
    
    if ( ( a * b ).sum() < 0 ):
        a = a * -1

    return a


def get_pca_and_orient_forwards(cas):

    pca = sklearn.decomposition.PCA()
    pca.fit(cas)
    axis = torch.tensor(pca.components_[0], dtype=cas.dtype)
    
    axis /= torch.sqrt(torch.sum(torch.square(axis)))
    
    # now we orient the PCAs so that they all point forwards
    #  this handles sharp kinks better than dotting with the previous pca
    first_to_last = cas[-1] - cas[0]

    axis = orient_a_vector_to_b( axis, first_to_last )

    return axis

# the basic idea is to get a general sense of the directionality of the protein at any given point
#  and look for places where it suddenly turns
#    min_helix_size:      After the calculations, fill in gaps smaller than this
#    correlation_length:  size of the region to calculate directionality over
#    angle_cutoff:        minimum angle between sections at big_lookahead to define as loop
#    big_lookahead:       lookahead this many to generally find the turn locations
#    small_lookahead:     used to expand the loops, finds edges of long loops
def linear_chunk_helical_dssp(cas, min_helix_size=5, correlation_length=8,
                               angle_cutoff=70, big_lookahead=11, small_lookahead=7):
    
    assert( (correlation_length + big_lookahead) % 2 == 1)
    assert( (correlation_length + small_lookahead) % 2 == 1)
    
    
    # find the directionaly of every overlapping section of the protein
    my_pcas = torch.zeros((len(cas)-correlation_length, 3))
    for isection in range(len(cas)-correlation_length):

        my_pcas[isection] = get_pca_and_orient_forwards(cas[isection:isection+correlation_length])
        
    # corr = 8
    # look = 5
    # adder = 6
    # 012345678901234567890
    # cccccccc
    # -----cccccccc
    
    # add this to the starting address to find the center of the region we're looking at
    big_adder = (correlation_length + big_lookahead) // 2
    small_adder = (correlation_length + small_lookahead) // 2
        
    # angles between sections using big_lookahead
    big_angles = torch.zeros(len(my_pcas)-big_lookahead, dtype=cas.dtype)
    for isection in range(len(my_pcas)-big_lookahead):
        lb = my_pcas[isection]
        ub = my_pcas[isection+big_lookahead]
        
        angle = torch.arccos( torch.sum( lb * ub) )
        big_angles[isection] = angle * 180 / np.pi
        
        # center = isection + big_adder
        
    # angles between sections using small_lookahead
    small_angles = torch.zeros(len(my_pcas)-small_lookahead, dtype=cas.dtype)
    for isection in range(len(my_pcas)-small_lookahead):
        lb = my_pcas[isection]
        ub = my_pcas[isection+small_lookahead]
        
        angle = torch.arccos( torch.sum( lb * ub) )
        small_angles[isection] = angle * 180 / np.pi
        
        # center = isection + small_adder
        
#     plt.figure(figsize=(17, 4))
#     plt.plot( np.arange(0, len(big_angles)) + big_adder, big_angles, color='red' )
#     plt.plot( np.arange(0, len(small_angles)) + small_adder, small_angles, color='green' )
#     plt.axhline(angle_cutoff)
    
    is_loop = torch.zeros(len(cas), dtype=bool)
    
    # step 1, big_lookahead defines regions that are all above threshold
    above_threshold = big_angles > angle_cutoff
    

    pre_loop_regions = repeating_regions(above_threshold, only_keep=True)
    loop_regions = [(lb + big_adder, ub + big_adder) for _, lb, ub in pre_loop_regions]
        
    # step 2, big_lookahead defines a loop at the local max of each region
    for lb, ub in loop_regions:
        lb -= big_adder
        ub -= big_adder
        
        our_slice = big_angles[lb:ub+1]
        peak = torch.argmax(our_slice)
        
        loop_lb = torch.clip( peak + lb + big_adder - 1, 0, len(cas)-1)
        loop_ub = torch.clip( peak + lb + big_adder + 1, 0, len(cas)-1)
        
#         print(lb+big_adder, ub+big_adder, loop_lb, loop_ub)
        
#         plt.axvline(loop_lb, color='red')
#         plt.axvline(loop_ub, color='red')
        
        is_loop[loop_lb:loop_ub+1] = True
        
    # step 3, small_lookahead defines loop at any local maximum in side region
        for lb, ub in loop_regions:
            lb -= small_adder
            ub -= small_adder

            our_slice = small_angles[lb:ub+1]

            for i in range(len(our_slice)):
                prev = -100
                if ( i > 0 ):
                    prev = our_slice[i-1]
                nextt = -100
                if ( i < len(our_slice)-1 ):
                    nextt = our_slice[i+1]

                if ( prev < our_slice[i] and nextt < our_slice[i] ):
                    peak = i

#                     loop_lb = np.clip( peak + lb + small_adder - 1, 0, len(cas)-1)
#                     loop_ub = np.clip( peak + lb + small_adder + 1, 0, len(cas)-1)
                    
#                     plt.axvline(loop_lb+0.1, color='green')
#                     plt.axvline(loop_ub+0.1, color='green')

#                     is_loop[loop_lb:loop_ub+1] = True

                    pt = peak + lb + small_adder
                    is_loop[pt] = True
#                     plt.axvline(pt, color='green')
    
    
    # step 4, remove any regions smaller than min helix length
    
    is_helix = ~is_loop
    this_helix_start = 0
    on_helix = False
    for i in range(len(is_helix)):
        if ( is_helix[i] ):
            on_helix = True
            continue

        if ( on_helix ):
            last_helical_length = i - this_helix_start
            if ( last_helical_length < min_helix_size ):
                is_helix[this_helix_start:i] = False
                
#                 plt.axvline(this_helix_start+0.1, color='blue')
#                 plt.axvline(i-1+0.1, color='blue')
        on_helix = False
        this_helix_start = i+1
        
    
    return is_helix















