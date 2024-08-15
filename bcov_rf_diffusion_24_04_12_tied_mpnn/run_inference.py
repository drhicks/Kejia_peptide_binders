#! /home/dimaio/.conda/envs/SE3nv/bin/python
"""
Inference script.

To run with base.yaml as the config,

> python run_inference.py

To specify a different config,

> python run_inference.py --config-name symmetry

where symmetry can be the filename of any other config (without .yaml extension)
See https://hydra.cc/docs/advanced/hydra-command-line-flags/ for more options.

"""

import os, time, pickle
import torch 
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
from util import writepdb_multi, writepdb
from inference import utils as iu
from icecream import ic
from hydra.core.hydra_config import HydraConfig

import numpy as np
import chemical



import inference.bcov_hacks.bcov_hacks as hax

@hydra.main(version_base=None, config_path='config/inference', config_name='base')
def main(conf: HydraConfig) -> None:
    log = logging.getLogger(__name__)

    sampler = iu.sampler_selector(conf)
    
    arg_replaces = hax.load_argreplace(conf)
    for arg_replace in arg_replaces:

        # Initialize sampler and target/contig.
        if ( arg_replace is not None ):
            sampler.initialize(conf, arg_replace)

        checkpoint_name = f'{sampler.output_prefix()}_ckpt'
        checkpoint_successes = hax.load_checkpoint(checkpoint_name)
        
        # Loop over number of designs to sample.
        for i_des in range(sampler.inf_conf.design_startnum, sampler.inf_conf.design_startnum + sampler.inf_conf.num_designs):

            if ( conf.inference.stop_after_x_successes ):
                if ( sum(checkpoint_successes.values()) >= conf.inference.stop_after_x_successes ):
                    log.info(f'inference.stop_after_x_successes={conf.inference.stop_after_x_successes} reached. Stopping')
                    break

            start_time = time.time()
            out_prefix = f'{sampler.output_prefix()}_{i_des}'
            log.info(f'Making design {out_prefix}')

            if ( out_prefix in checkpoint_successes ):
                log.info(f'Skipping this design because it already exists in ckpt')
                sampler.note_checkpointed_design()
                continue
            if sampler.inf_conf.cautious and os.path.exists(out_prefix+'.pdb'):
                log.info(f'(cautious mode) Skipping this design because {out_prefix}.pdb already exists.')
                sampler.note_checkpointed_design()
                continue

            hax.reinit_hacks(conf)
            hax.hacks().set_prefix_tag(sampler.output_prefix(False), out_prefix)
            state_dict = hax.hacks().maybe_load_state( sampler )

            x_init, seq_init = sampler.sample_init()

            denoised_xyz_stack = []
            px0_xyz_stack = []
            seq_stack = []
            chi1_stack = []
            plddt_stack = []
            t_stack = []

            x_t = torch.clone(x_init)
            seq_t = torch.clone(seq_init)

            success = False
            terminal_message = "Sampler never ran"

            first_t = int(sampler.t_step_input)

            if ( state_dict ):
                first_t = state_dict['t']
                seq_t = state_dict['seq_t']
                x_t = state_dict['x_t']
                seq_init = state_dict['seq_init']

            px0 = None

            # Loop over number of reverse diffusion time steps.
            for t in range(first_t, sampler.inf_conf.final_step-1, -1):

                warping, x_t, skip_diffuser = hax.hacks().maybe_time_warp(t, seq_t, x_t, px0, sampler )
                if ( warping ):
                    continue

                px0, x_t, seq_t, tors_t, plddt = sampler.sample_step(
                    t=t, seq_t=seq_t, x_t=x_t, seq_init=seq_init, final_step=sampler.inf_conf.final_step, skip_diffuser=skip_diffuser)
                px0_xyz_stack.append(px0)
                denoised_xyz_stack.append(x_t)
                seq_stack.append(seq_t)
                chi1_stack.append(tors_t[:,:])
                plddt_stack.append(plddt[0]) # remove singleton leading dimension
                t_stack.append(t)

                success, terminal_message = hax.hacks().apply_run_filters(t, px0, x_t, seq_t, tors_t, plddt, sampler)
                if ( not success ):
                    break

            if ( not success ):
                log.info(f'Filter failure: {out_prefix}')
                log.info(f'Failed filter: {terminal_message}')
                hax.save_checkpoint( checkpoint_name, out_prefix, terminal_message )
                checkpoint_successes[out_prefix] = False
                continue
            
            # Flip order for better visualization in pymol
            denoised_xyz_stack = torch.stack(denoised_xyz_stack)
            denoised_xyz_stack = torch.flip(denoised_xyz_stack, [0,])
            px0_xyz_stack = torch.stack(px0_xyz_stack)
            px0_xyz_stack = torch.flip(px0_xyz_stack, [0,])
            t_stack = list(reversed(t_stack))
            seq_stack = list(reversed(seq_stack))

            # For logging -- don't flip
            plddt_stack = torch.stack(plddt_stack)

            # Save outputs 
            os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
            final_seq = seq_stack[0]
            bfacts = torch.ones_like(final_seq.squeeze())

            # pX0 last step
            out = f'{out_prefix}.pdb'

            # replace mask and unknown tokens in the final seq with alanine
            final_seq = torch.where(final_seq == 20, 0, final_seq)
            final_seq = torch.where(final_seq == 21, 0, final_seq)

            # write output
            hax.hacks().add_scores_to_run(None, dict(time="%i"%(time.time()-start_time)))
            pose_dict = hax.AtomsToPose.make_pose_dict(px0_xyz_stack[0], denoised_xyz_stack[0], final_seq, sampler.binderlen, sampler.chain_idx)

            regular_output = hax.hacks().finish_run(sampler.binderlen, pose_dict )

            #if ( regular_output ):
            #    # run metadata
            #    trb = dict(
            #        config = OmegaConf.to_container(sampler._conf, resolve=True),
            #        plddt = plddt_stack.cpu().numpy(),
            #        device = torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else 'CPU',
            #        time = time.time() - start_time
            #    )
            #    if hasattr(sampler, 'contig_map'):
            #        for key, value in sampler.contig_map.get_mappings().items():
            #            trb[key] = value
            #    with open(f'{out_prefix}.trb','wb') as f_out:
            #        pickle.dump(trb, f_out)

                ## trajectory pdbs
                #traj_prefix = os.path.dirname(out_prefix)+'/traj/'+os.path.basename(out_prefix)
                #os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)

                #if ( True ):
                #    out = f'{traj_prefix}_Xt-1_traj.pdb'
                #    writepdb_multi(out, denoised_xyz_stack, bfacts, 
                #        final_seq.squeeze(), use_hydrogens=False, backbone_only=False, chain_ids=sampler.chain_idx)

                #    out=f'{traj_prefix}_pX0_traj.pdb'
                #    writepdb_multi(out, px0_xyz_stack, bfacts, 
                #        final_seq.squeeze(), use_hydrogens=False, backbone_only=False, chain_ids=sampler.chain_idx)



            if ( hax.hacks().timewarp_subtrajectories ):
                saved_structures = {}

                hax.hacks().rf2_frameskip=None

                frames = []


                if hax.hacks().twpst_unconstrain_binder:
                    sampler.mask_str[:,:sampler.binderlen] = False
                    sampler.mask_seq[:,:sampler.binderlen] = False
                    sampler.diffusion_mask[:,:sampler.binderlen] = False
                    seq_init[:sampler.binderlen] = 21



                saved_partials = {}
                for subtraj in hax.hacks().timewarp_subtrajectories:

                    print("Starting subtrajectory", subtraj)

                    # [20, 70, 60]
                    # first thing to lookup is [20,70] == subtraj[:2]
                    start_i = 0
                    have_partial = False
                    for i in reversed(range(1, len(subtraj))):
                        lookup = "-".join(str(x) for x in subtraj[:i+1])
                        if ( lookup in saved_partials ):
                            have_partial = True
                            px0 = saved_partials[lookup]

                            print(" starting from %s"%lookup)
                            start_i = i+1
                            break
                    if ( not have_partial ):
                        initial_t = subtraj[0]
                        init_idx = t_stack.index(initial_t)
                        assert(init_idx >= 0)
                        px0 = px0_xyz_stack[init_idx]
                        start_i = 1
                        print(" starting from t=%i"%initial_t)

                    frames.append(px0.clone())

                    for i_t in range(start_i, len(subtraj)):
                        t = subtraj[i_t]
                        seq_t = torch.clone(seq_init)
                        x_t = hax.partially_diffuse_to_t(px0, t, seq_t, sampler)
                        frames.append(x_t)
                        sampler.prev_pred = px0.to(sampler.device)[None,...]
                        px0, x_t, seq_t, tors_t, plddt = sampler.sample_step(
                            t=t, seq_t=seq_t, x_t=x_t, seq_init=seq_init, final_step=0, skip_diffuser=True)

                        frames.append(x_t.clone())
                        frames.append(px0.clone())
                        lookup = "-".join(str(x) for x in subtraj[:i_t+1])
                        saved_partials[lookup] = px0.clone()

                        
                    suffix = "twpst-" + "-".join(str(x) for x in subtraj)
                    pose_dict = hax.AtomsToPose.make_pose_dict(px0, px0, seq_t, sampler.binderlen, sampler.chain_idx)
                    hax.hacks().write_output(sampler.binderlen, pose_dict, suffix=suffix )

                frames = torch.stack(frames, axis=0)

                full_frames = torch.full((frames.shape[0], frames.shape[1], 27, 3), np.nan)
                full_frames[:,:,:14,:] = frames

                # fname = "look.pdb"
                # writepdb_multi( fname, full_frames, torch.ones(len(px0)), torch.full((len(px0),), chemical.aa2num["GLY"]) )




            hax.save_checkpoint( checkpoint_name, out_prefix, "Success" )
            checkpoint_successes[out_prefix] = True

            log.info(f'Finished design in {(time.time()-start_time)/60:.2f} minutes')

if __name__ == '__main__':
    main()
