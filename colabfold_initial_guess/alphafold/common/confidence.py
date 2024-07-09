# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for processing confidence metrics."""

import jax.numpy as jnp
import jax
import numpy as np
from alphafold.common import residue_constants
import scipy.special


########################## update bcov dict #############################
def update_bcov_dict(bcov_filters_dict, result, run_all=False):
  binderlen = len(bcov_filters_dict['len_is_binderlen'])

  pae = result['predicted_aligned_error']
  
  this_scores = {}
  """
  pae_interaction1 = jnp.mean( pae[:binderlen,binderlen:] )
  pae_interaction2 = jnp.mean( pae[binderlen:,:binderlen] )
  pae_interaction = ( pae_interaction1 + pae_interaction2 ) / 2
  """

  this_scores['pae_interaction'] = jnp.mean( jnp.concatenate( [pae[binderlen:,:binderlen], pae[:binderlen,binderlen:].T], axis=0 ) )

  is_passing = True

  if 'input_ca' in bcov_filters_dict:
    input_ca = bcov_filters_dict['input_ca']
    current_ca = result['structure_module']['final_atom_positions'][:,1,:]
  
    binder_rmsd, _, _, _ = jax_super_rmsd( current_ca[:binderlen], input_ca[:binderlen] )
    target_rmsd, post, rotate, pre = jax_super_rmsd( current_ca[binderlen:], input_ca[binderlen:] )
    aligned_current = ( rotate @ ( current_ca + pre[None,:] ).T ).T + post[None,:]
    interface_rmsd = jnp.sqrt(jnp.mean(jnp.sum(jnp.square(aligned_current[:binderlen] - input_ca[:binderlen]), axis=-1)))
  
    this_scores['binder_rmsd'] = binder_rmsd
    this_scores['target_rmsd'] = target_rmsd
    this_scores['interface_rmsd'] = interface_rmsd

    is_passing = jnp.logical_and( is_passing, this_scores['pae_interaction'] < bcov_filters_dict['pae_interaction_cut'] )
    is_passing = jnp.logical_and( is_passing, this_scores['interface_rmsd'] < bcov_filters_dict['interface_rmsd_cut'] )

    bcov_filters_dict.update(this_scores)
  
  bcov_filters_dict['bcov_continue'] = is_passing

  if not run_all:
      return bcov_filters_dict

  this_scores['pae_binder'] = jnp.mean( pae[:binderlen,:binderlen] )
  this_scores['pae_target'] = jnp.mean( pae[binderlen:,binderlen:] )
  
  plddt = result['plddt']
  this_scores['plddt_total'] = jnp.mean( plddt )
  this_scores['plddt_binder'] = jnp.mean( plddt[:binderlen] )
  this_scores['plddt_target'] = jnp.mean( plddt[binderlen:] )

  interface_mask = find_interface_mask(result['structure_module']['final_atom_positions'], binderlen)
  iplddt = calculate_average_plddt(plddt, interface_mask)
  binder_contacts, target_contacts, total_contacts = count_interface_residues(interface_mask, binderlen)

  this_scores["iplddt"] = iplddt
  this_scores["binder_contacts"] = binder_contacts
  this_scores["target_contacts"] = target_contacts
  this_scores["total_contacts"] = total_contacts

  bcov_filters_dict.update(this_scores)

  return bcov_filters_dict

########################## update bcov dict #############################


########################### bcov functions ##############################

def jax_superposition_xform(from_pts, to_pts):

    from_com = jnp.mean(from_pts, axis=0)
    to_com = jnp.mean(to_pts, axis=0)

    from_pts = from_pts - from_com
    to_pts = to_pts - to_com

    A = to_pts
    B = from_pts

    C = jnp.matmul(A.T, B)

    U,S,Vt = jnp.linalg.svd(C)

    # ensure right handed coordinate system
    d = jnp.eye(3)
    d = d.at[-1,-1].set( jnp.sign(jnp.linalg.det(Vt.T@U.T)) )

    rotation = U@d@Vt

    pre_translate = -from_com
    post_translate = to_com


    return post_translate, rotation, pre_translate

def jax_super_rmsd(from_pts, to_pts):

  post, rotate, pre = jax_superposition_xform(from_pts, to_pts)

  new_from_pts = ( rotate @ ( from_pts + pre[None,:] ).T ).T + post[None,:]

  rmsd = jnp.sqrt(jnp.mean(jnp.sum(jnp.square(new_from_pts - to_pts), axis=-1)))

  return rmsd, post, rotate, pre

def find_interface_mask(all_atom_positions, binder_length, contact_threshold=5.0, min_distance=0.1):
    """
    Find interface residues between binder and target. Returns a boolean mask.
    """
    # Split positions into binder and target
    binder_positions = all_atom_positions[:binder_length]
    target_positions = all_atom_positions[binder_length:]

    # Filter out placeholder atoms (coordinates not at (0, 0, 0))
    binder_valid_atoms = np.any(binder_positions != 0, axis=2)
    target_valid_atoms = np.any(target_positions != 0, axis=2)

    # Pre-calculate the valid positions for each residue
    valid_binder_positions = [binder_positions[i, binder_valid_atoms[i]] for i in range(binder_positions.shape[0])]
    valid_target_positions = [target_positions[j, target_valid_atoms[j]] for j in range(target_positions.shape[0])]

    # Initialize empty masks
    binder_mask = np.zeros(binder_positions.shape[0], dtype=bool)
    target_mask = np.zeros(target_positions.shape[0], dtype=bool)

    # Compute pairwise distances for all valid atom pairs
    for binder_idx, valid_binder in enumerate(valid_binder_positions):
        for target_idx, valid_target in enumerate(valid_target_positions):
            if valid_binder.size > 0 and valid_target.size > 0:
                distances = np.linalg.norm(valid_binder[:, None, :] - valid_target[None, :, :], axis=-1)
                if np.any((distances > min_distance) & (distances <= contact_threshold)):
                    binder_mask[binder_idx] = True
                    target_mask[target_idx] = True

    # Combine masks for binder and target
    interface_mask = np.concatenate([binder_mask, target_mask])
    
    return interface_mask


def calculate_average_plddt(plddt_scores, interface_mask):
    """
    Calculate the average pLDDT of interface residues.
    """
    interface_plddt = np.where(interface_mask, plddt_scores, 0)
    sum_interface_plddt = np.sum(interface_plddt)
    count_interface_residues = np.sum(interface_mask)
    average_plddt = sum_interface_plddt / max(count_interface_residues, 1)
    return average_plddt

def count_interface_residues(interface_mask, binder_length):
    """
    Count the number of residues in contact in both the binder and target.
    """
    binder_contacts = np.sum(interface_mask[:binder_length])
    target_contacts = np.sum(interface_mask[binder_length:])
    total_contacts = binder_contacts + target_contacts
    return binder_contacts, target_contacts, total_contacts

##########################################################################

def compute_tol(prev_pos, current_pos, mask, use_jnp=False):
    # Early stopping criteria based on criteria used in
    # AF2Complex: https://www.nature.com/articles/s41467-022-29394-2    
    _np = jnp if use_jnp else np
    dist = lambda x:_np.sqrt(((x[:,None] - x[None,:])**2).sum(-1))
    ca_idx = residue_constants.atom_order['CA']
    sq_diff = _np.square(dist(prev_pos[:,ca_idx])-dist(current_pos[:,ca_idx]))
    mask_2d = mask[:,None] * mask[None,:]
    return _np.sqrt((sq_diff * mask_2d).sum()/mask_2d.sum() + 1e-8)

def compute_plddt(logits, use_jnp=False):
  """Computes per-residue pLDDT from logits.
  Args:
    logits: [num_res, num_bins] output from the PredictedLDDTHead.
  Returns:
    plddt: [num_res] per-residue pLDDT.
  """
  if use_jnp:
    _np, _softmax = jnp, jax.nn.softmax
  else:
    _np, _softmax = np, scipy.special.softmax
  
  num_bins = logits.shape[-1]
  bin_width = 1.0 / num_bins
  bin_centers = _np.arange(start=0.5 * bin_width, stop=1.0, step=bin_width)
  probs = _softmax(logits, axis=-1)
  predicted_lddt_ca = (probs * bin_centers[None, :]).sum(-1)
  return predicted_lddt_ca * 100

def _calculate_bin_centers(breaks, use_jnp=False):
  """Gets the bin centers from the bin edges.
  Args:
    breaks: [num_bins - 1] the error bin edges.
  Returns:
    bin_centers: [num_bins] the error bin centers.
  """
  _np = jnp if use_jnp else np
  step = breaks[1] - breaks[0]

  # Add half-step to get the center
  bin_centers = breaks + step / 2

  # Add a catch-all bin at the end.
  return _np.append(bin_centers, bin_centers[-1] + step)

def _calculate_expected_aligned_error(
  alignment_confidence_breaks,
  aligned_distance_error_probs,
  use_jnp=False):
  """Calculates expected aligned distance errors for every pair of residues.
  Args:
    alignment_confidence_breaks: [num_bins - 1] the error bin edges.
    aligned_distance_error_probs: [num_res, num_res, num_bins] the predicted
      probs for each error bin, for each pair of residues.
  Returns:
    predicted_aligned_error: [num_res, num_res] the expected aligned distance
      error for each pair of residues.
    max_predicted_aligned_error: The maximum predicted error possible.
  """
  bin_centers = _calculate_bin_centers(alignment_confidence_breaks, use_jnp=use_jnp)
  # Tuple of expected aligned distance error and max possible error.
  pae = (aligned_distance_error_probs * bin_centers).sum(-1)
  return (pae, bin_centers[-1])

def compute_predicted_aligned_error(logits, breaks, use_jnp=False):
  """Computes aligned confidence metrics from logits.
  Args:
    logits: [num_res, num_res, num_bins] the logits output from
      PredictedAlignedErrorHead.
    breaks: [num_bins - 1] the error bin edges.

  Returns:
    aligned_confidence_probs: [num_res, num_res, num_bins] the predicted
      aligned error probabilities over bins for each residue pair.
    predicted_aligned_error: [num_res, num_res] the expected aligned distance
      error for each pair of residues.
    max_predicted_aligned_error: The maximum predicted error possible.
  """
  _softmax = jax.nn.softmax if use_jnp else scipy.special.softmax
  aligned_confidence_probs = _softmax(logits,axis=-1)
  predicted_aligned_error, max_predicted_aligned_error = \
  _calculate_expected_aligned_error(breaks, aligned_confidence_probs, use_jnp=use_jnp)

  return {
      'aligned_confidence_probs': aligned_confidence_probs,
      'predicted_aligned_error': predicted_aligned_error,
      'max_predicted_aligned_error': max_predicted_aligned_error,
  }

def predicted_tm_score(logits, breaks, residue_weights = None,
    asym_id = None, use_jnp=False):
  """Computes predicted TM alignment or predicted interface TM alignment score.

  Args:
    logits: [num_res, num_res, num_bins] the logits output from
      PredictedAlignedErrorHead.
    breaks: [num_bins] the error bins.
    residue_weights: [num_res] the per residue weights to use for the
      expectation.
    asym_id: [num_res] the asymmetric unit ID - the chain ID. Only needed for
      ipTM calculation.

  Returns:
    ptm_score: The predicted TM alignment or the predicted iTM score.
  """
  if use_jnp:
    _np, _softmax = jnp, jax.nn.softmax
  else:
    _np, _softmax = np, scipy.special.softmax

  # residue_weights has to be in [0, 1], but can be floating-point, i.e. the
  # exp. resolved head's probability.
  if residue_weights is None:
    residue_weights = _np.ones(logits.shape[0])

  bin_centers = _calculate_bin_centers(breaks, use_jnp=use_jnp)
  num_res = residue_weights.shape[0]

  # Clip num_res to avoid negative/undefined d0.
  clipped_num_res = _np.maximum(residue_weights.sum(), 19)

  # Compute d_0(num_res) as defined by TM-score, eqn. (5) in Yang & Skolnick
  # "Scoring function for automated assessment of protein structure template
  # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
  d0 = 1.24 * (clipped_num_res - 15) ** (1./3) - 1.8

  # Convert logits to probs.
  probs = _softmax(logits, axis=-1)

  # TM-Score term for every bin.
  tm_per_bin = 1. / (1 + _np.square(bin_centers) / _np.square(d0))
  # E_distances tm(distance).
  predicted_tm_term = (probs * tm_per_bin).sum(-1)

  if asym_id is None:
    pair_mask = _np.full((num_res,num_res),True)
  else:
    pair_mask = asym_id[:, None] != asym_id[None, :]

  predicted_tm_term *= pair_mask

  pair_residue_weights = pair_mask * (residue_weights[None, :] * residue_weights[:, None])
  normed_residue_mask = pair_residue_weights / (1e-8 + pair_residue_weights.sum(-1, keepdims=True))
  per_alignment = (predicted_tm_term * normed_residue_mask).sum(-1)

  return (per_alignment * residue_weights).max()

def get_confidence_metrics(prediction_result, mask, rank_by = "plddt", use_jnp=False):
  """Post processes prediction_result to get confidence metrics."""  
  confidence_metrics = {}
  plddt = compute_plddt(prediction_result['predicted_lddt']['logits'], use_jnp=use_jnp)
  confidence_metrics['plddt'] = plddt  
  confidence_metrics["mean_plddt"] = (plddt * mask).sum()/mask.sum()

  if 'predicted_aligned_error' in prediction_result:
    confidence_metrics.update(compute_predicted_aligned_error(
        logits=prediction_result['predicted_aligned_error']['logits'],
        breaks=prediction_result['predicted_aligned_error']['breaks'],
        use_jnp=use_jnp))
    
    confidence_metrics['ptm'] = predicted_tm_score(
        logits=prediction_result['predicted_aligned_error']['logits'],
        breaks=prediction_result['predicted_aligned_error']['breaks'],
        residue_weights=mask,
        use_jnp=use_jnp)    

    if "asym_id" in prediction_result["predicted_aligned_error"]:
      # Compute the ipTM only for the multimer model.
      confidence_metrics['iptm'] = predicted_tm_score(
          logits=prediction_result['predicted_aligned_error']['logits'],
          breaks=prediction_result['predicted_aligned_error']['breaks'],
          residue_weights=mask,
          asym_id=prediction_result['predicted_aligned_error']['asym_id'],
          use_jnp=use_jnp)

  # compute mean_score
  if rank_by == "multimer":
    mean_score = 80 * confidence_metrics["iptm"] + 20 * confidence_metrics["ptm"]
  elif rank_by == "iptm":
    mean_score = 100 * confidence_metrics["iptm"]
  elif rank_by == "ptm":
    mean_score = 100 * confidence_metrics["ptm"]
  else:
    mean_score = confidence_metrics["mean_plddt"]
  confidence_metrics["ranking_confidence"] = mean_score
  return confidence_metrics
