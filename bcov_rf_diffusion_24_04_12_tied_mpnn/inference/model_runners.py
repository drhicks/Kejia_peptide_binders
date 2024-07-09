import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from icecream import ic
from RoseTTAFoldModel import RoseTTAFoldModule
from kinematics import get_init_xyz, xyz_to_t2d
from diffusion import Diffuser
from chemical import seq2chars, INIT_CRDS
from util_module import ComputeAllAtomCoords
from contigs import ContigMap
from inference import utils as iu
from potentials.manager import PotentialManager
from inference import symmetry
import logging
import torch.nn.functional as nn
import util
import hydra
from hydra.core.hydra_config import HydraConfig
import os
import inference.bcov_hacks.bcov_hacks as hax

import sys
sys.path.append('../') # to access RF structure prediction stuff 

# When you import this it causes a circular import due to the changes made in apply masks for self conditioning
# This import is only used for SeqToStr Sampling though so can be fixed later - NRB
# import data_loader 
from model_input_logger import pickle_function_call

TOR_INDICES  = util.torsion_indices
TOR_CAN_FLIP = util.torsion_can_flip
REF_ANGLES   = util.reference_angles

class Sampler:

    def __init__(self, conf: DictConfig):
        """Initialize sampler.
        Args:
            conf: Configuration.
        """
        self.initialized = False
        self.initialize(conf, None)
    
    def initialize(self, conf: DictConfig, arg_replace: DictConfig):
        self._log = logging.getLogger(__name__)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        needs_model_reload = not self.initialized or conf.inference.ckpt_path != self._conf.inference.ckpt_path

        # Assign config to Sampler
        self._conf = conf

        # Initialize inference only helper objects to Sampler
        self.ckpt_path = conf.inference.ckpt_path

        if needs_model_reload:
            # Load checkpoint, so that we can assemble the config
            self.load_checkpoint()
            self.assemble_config_from_chk(arg_replace)
            # Now actually load the model weights into RF
            self.model = self.load_model()
        else:
            self.assemble_config_from_chk(arg_replace)

        # self.initialize_sampler(conf)
        self.initialized=True

        # Assemble config from the checkpoint
        print(' ')
        print('-'*100)
        print(' ')
        print("WARNING: The following options are not currently implemented at inference. Decide if this matters.")
        print("Delete these in inference/model_runners.py once they are implemented/once you decide they are not required for inference -- JW")
        print(" -predict_previous")
        print(" -prob_self_cond")
        print(" -seqdiff_b0")
        print(" -seqdiff_bT")
        print(" -seqdiff_schedule_type")
        print(" -seqdiff")
        print(" -freeze_track_motif")
        print(" -use_motif_timestep")
        print(" ")
        print("-"*100)
        print(" ")
        # Initialize helper objects
        self.inf_conf = self._conf.inference
        self.contig_conf = self._conf.contigmap
        self.denoiser_conf = self._conf.denoiser
        self.ppi_conf = self._conf.ppi
        self.potential_conf = self._conf.potentials
        self.diffuser_conf = self._conf.diffuser
        self.preprocess_conf = self._conf.preprocess
        self.diffuser = Diffuser(**self._conf.diffuser)
        # TODO: Add symmetrization RMSD check here
        if self.inf_conf.symmetry is not None:
            self.symmetry = symmetry.SymGen(
                self.inf_conf.symmetry,
                self.inf_conf.model_only_neighbors,
                self.inf_conf.recenter,
                self.inf_conf.radius, 
            )
        else:
            self.symmetry = None


        self.allatom = ComputeAllAtomCoords().to(self.device)

        if self.diffuser_conf.partial_T:
            assert self.diffuser_conf.partial_T <= self.diffuser_conf.T
            self.t_step_input = int(self.diffuser_conf.partial_T)
        else:
            self.t_step_input = int(self.diffuser_conf.T)
        
        # Get recycle schedule    
        recycle_schedule = str(self.inf_conf.recycle_schedule) if self.inf_conf.recycle_schedule is not None else None
        self.recycle_schedule = iu.recycle_schedule(self.T, recycle_schedule, self.inf_conf.num_recycles)
        
    @property
    def T(self):
        '''
            Return the maximum number of timesteps
            that this design protocol will perform.

            Output:
                T (int): The maximum number of timesteps to perform
        '''
        return self.diffuser_conf.T
    
    def load_checkpoint(self) -> None:
        """Loads RF checkpoint, from which config can be generated."""
        self._log.info(f'Reading checkpoint from {self.ckpt_path}')
        print('This is inf_conf.ckpt_path')
        print(self.ckpt_path)
        self.ckpt  = torch.load(
            self.ckpt_path, map_location=self.device)

    def assemble_config_from_chk(self, arg_replace: DictConfig) -> None:
        """
        Function for loading model config from checkpoint directly.
    
        Takes:
            - config file
    
        Actions:
            - Replaces all -model and -diffuser items
            - Throws a warning if there are items in -model and -diffuser that aren't in the checkpoint
        
        This throws an error if there is a flag in the checkpoint 'config_dict' that isn't in the inference config.
        This should ensure that whenever a feature is added in the training setup, it is accounted for in the inference script.

        JW
        """
        
        # get overrides to re-apply after building the config from the checkpoint
        overrides = []
        if HydraConfig.initialized():
            overrides = HydraConfig.get().overrides.task
            ic(overrides)
        if 'config_dict' in self.ckpt.keys():
            print("Assembling -model, -diffuser and -preprocess configs from checkpoint")

            # First, check all flags in the checkpoint config dict are in the config file
            for cat in ['model','diffuser','seqdiffuser','preprocess']:
                #assert all([i in self._conf[cat].keys() for i in self.ckpt['config_dict'][cat].keys()]), f"There are keys in the checkpoint config_dict {cat} params not in the config file"
                for key in self._conf[cat]:
                    if key == 'chi_type' and self.ckpt['config_dict'][cat][key] == 'circular':
                        ic('---------------------------------------------SKIPPPING CIRCULAR CHI TYPE')
                        continue
                    try:
                        print(f"USING MODEL CONFIG: self._conf[{cat}][{key}] = {self.ckpt['config_dict'][cat][key]}")
                        self._conf[cat][key] = self.ckpt['config_dict'][cat][key]
                    except:
                        print(f'WARNING: config {cat}.{key} is not saved in the checkpoint. Check that conf.{cat}.{key} = {self._conf[cat][key]} is correct')
            # add back in overrides again
            for override in overrides:
                if override.split(".")[0] in ['model','diffuser','seqdiffuser','preprocess']:
                    print(f'WARNING: You are changing {override.split("=")[0]} from the value this model was trained with. Are you sure you know what you are doing?') 
                    mytype = type(self._conf[override.split(".")[0]][override.split(".")[1].split("=")[0]])
                    self._conf[override.split(".")[0]][override.split(".")[1].split("=")[0]] = mytype(override.split("=")[1])
        else:
            print('WARNING: Model, Diffuser and Preprocess parameters are not saved in this checkpoint. Check carefully that the values specified in the config are correct for this checkpoint')     

        if ( arg_replace ):
            ic(arg_replace)
            self._conf.merge_with(arg_replace)
        # ic(self._conf)

    def load_model(self):
        """Create RosettaFold model from preloaded checkpoint."""
        
        # Now read input dimensions from checkpoint.
        self.d_t1d=self._conf.preprocess.d_t1d
        self.d_t2d=self._conf.preprocess.d_t2d
        model = RoseTTAFoldModule(**self._conf.model, d_t1d=self.d_t1d, d_t2d=self.d_t2d, T=self._conf.diffuser.T).to(self.device)
        if self._conf.logging.inputs:
            pickle_dir = pickle_function_call(model, 'forward', 'inference')
            print(f'pickle_dir: {pickle_dir}')
        model = model.eval()
        self._log.info(f'Loading checkpoint.')
        model.load_state_dict(self.ckpt['model_state_dict'], strict=True)
        return model

    def construct_contig(self, target_feats):
        self._log.info(f'Using contig: {self.contig_conf.contigs}')
        return ContigMap(target_feats, **self.contig_conf)

    def construct_denoiser(self, L, visible):
        """Make length-specific denoiser."""
        # TODO: Denoiser seems redundant. Combine with diffuser.
        denoise_kwargs = OmegaConf.to_container(self.diffuser_conf)
        denoise_kwargs.update(OmegaConf.to_container(self.denoiser_conf))
        aa_decode_steps = min(denoise_kwargs['aa_decode_steps'], denoise_kwargs['partial_T'] or 999)
        denoise_kwargs.update({
            'L': L,
            'diffuser': self.diffuser,
            'potential_manager': self.potential_manager,
            'visible': visible,
            'aa_decode_steps': aa_decode_steps,
        })
        return iu.Denoise(**denoise_kwargs)

    def sample_init(self, return_forward_trajectory=False):
        """Initial features to start the sampling process.
        
        Modify signature and function body for different initialization
        based on the config.
        
        Returns:
            xt: Starting positions with a portion of them randomly sampled.
            seq_t: Starting sequence with a portion of them set to unknown.
        """
        # process target and reinitialise potential_manager. This is here because the 'target' is always set up to be the second chain in out inputs. Could change this down the line
        self.target_feats = iu.process_target(self.inf_conf.input_pdb)
        # moved this here as should be updated each iteration of diffusion
        self.contig_map = self.construct_contig(self.target_feats)
        self.mappings = self.contig_map.get_mappings()
        self.mask_seq = torch.from_numpy(self.contig_map.inpaint_seq)[None,:]
        self.mask_str = torch.from_numpy(self.contig_map.inpaint_str)[None,:]
        self.binderlen =  len(self.contig_map.inpaint)        
        self.hotspot_0idx=iu.get_idx0_hotspots(self.mappings, self.ppi_conf, self.binderlen) 
        # Now initialise potential manager here. This allows variable-length binder design
        self.potential_manager = PotentialManager(self.potential_conf,
                                                  self.ppi_conf,
                                                  self.diffuser_conf,
                                                  self.inf_conf,
                                                  self.hotspot_0idx,
                                                  self.binderlen)
        target_feats = self.target_feats
        contig_map = self.contig_map

        xyz_27 = target_feats['xyz_27']
        mask_27 = target_feats['mask_27']
        seq_orig = target_feats['seq']
        L_mapped = len(self.contig_map.ref)
        
        diffusion_mask = self.mask_str
        self.diffusion_mask = diffusion_mask
        
        self.chain_idx=['A' if i < self.binderlen else 'B' for i in range(L_mapped)]
        
        preserve_center = hax.hacks().preserve_center
        # adjust size of input xt according to residue map 
        if self.diffuser_conf.partial_T:
            assert xyz_27.shape[0] == L_mapped, f"there must be a coordinate in the input PDB for each residue implied by the contig string for partial diffusion.  length of input PDB != length of contig string: {xyz_27.shape[0]} != {L_mapped}"
            assert contig_map.hal_idx0 == contig_map.ref_idx0, f'for partial diffusion there can be no offset between the index of a residue in the input and the index of the residue in the output, {contig_map.hal_idx0} != {contig_map.ref_idx0}'
            # Partially diffusing from a known structure
            xyz_mapped=xyz_27
            atom_mask_mapped = mask_27
        else:
            # Fully diffusing from points initialised at the origin
            # adjust size of input xt according to residue map
            xyz_mapped = torch.full((1,1,L_mapped,27,3), np.nan)
            xyz_mapped[:, :, contig_map.hal_idx0, ...] = xyz_27[contig_map.ref_idx0,...]
            xyz_mapped = get_init_xyz(xyz_mapped, preserve_center=preserve_center).squeeze()

            # adjust the size of the input atom map
            atom_mask_mapped = torch.full((L_mapped, 27), False)
            atom_mask_mapped[contig_map.hal_idx0] = mask_27[contig_map.ref_idx0]
        # adjust size of input seq, slice sequence into it, mask resulting tensor
        seq_t = torch.full((1,L_mapped), 21).squeeze()
        seq_t[contig_map.hal_idx0] = seq_orig[contig_map.ref_idx0]
        seq_t[~self.mask_seq.squeeze()] = 21

        # Diffuse the contig-mapped coordinates 
        if self.diffuser_conf.partial_T:
            assert self.diffuser_conf.partial_T <= self.diffuser_conf.T
            self.t_step_input = int(self.diffuser_conf.partial_T)
        else:
            self.t_step_input = int(self.diffuser_conf.T)
        t_list = np.arange(1, self.t_step_input+1)
        fa_stack, aa_masks, xyz_true = self.diffuser.diffuse_pose(
            xyz_mapped,
            torch.clone(seq_t),  # TODO: Check if copy is needed.
            atom_mask_mapped.squeeze(),
            diffusion_mask=diffusion_mask.squeeze(),
            t_list=t_list,
            diffuse_sidechains=self.preprocess_conf.sidechain_input,
            include_motif_sidechains=self.preprocess_conf.motif_sidechain_input,
            preserve_center=preserve_center)
        xT = fa_stack[-1].squeeze()[:,:14,:]
        xt = torch.clone(xT)

        hax.hacks().init_motifs_closer(xt, self.binderlen, self.diffusion_mask.squeeze())

        is_motif = self.mask_seq.squeeze()
        is_shown_at_t = torch.tensor(aa_masks[-1])
        if ( hax.hacks().dont_show_any_input ):
            is_shown_at_t[:] = False
        visible = is_motif | is_shown_at_t
        if self.diffuser_conf.partial_T:
            seq_t[visible] = seq_orig[visible]

        self.denoiser = self.construct_denoiser(len(self.contig_map.ref), visible=visible)
        if self.symmetry is not None:
            xt, seq_t = self.symmetry.apply_symmetry(xt, seq_t)
        self._log.info(f'Sequence init: {seq2chars(seq_t.numpy().tolist())}')
        
        if return_forward_trajectory:
            forward_traj = torch.cat([xyz_true[None], fa_stack[:,:,:]])
            aa_masks[:, diffusion_mask.squeeze()] = True
            return xt, seq_t, forward_traj, aa_masks, seq_orig
        
        self.msa_prev = None
        self.pair_prev = None
        self.state_prev = None

        return xt, seq_t

    def _preprocess(self, seq, xyz_t, t, repack=False):
        
        """
        Function to prepare inputs to diffusion model
        
            seq (L) integer sequence 

            msa_masked (1,1,L,48)

            msa_full (1,1, L,25)
        
            xyz_t (L,14,3) template crds (diffused) 

            t1d (1,L,28) this is the t1d before tacking on the chi angles:
                - seq + unknown/mask (21)
                - global timestep (1-t/T if not motif else 1) (1)
                - contacting residues: for ppi. Target residues in contact with biner (1)
                - chi_angle timestep (1)
                - ss (H, E, L, MASK) (4)
            
            t2d (1, L, L, 45)
                - last plane is block adjacency
    """
        L = seq.shape[-1]
        T = self.T
        binderlen = self.binderlen
        target_res = self.ppi_conf.hotspot_res
        
        ### msa_masked ###
        ##################
        msa_masked = torch.zeros((1,1,L,48))
        msa_masked[:,:,:,:22] = nn.one_hot(seq, num_classes=22)[None, None]
        msa_masked[:,:,:,22:44] = nn.one_hot(seq, num_classes=22)[None, None]

        ### msa_full ###
        ################
        msa_full = torch.zeros((1,1,L,25))
        msa_full[:,:,:,:22] = nn.one_hot(seq, num_classes=22)[None, None]

        ### t1d ###
        ########### 
        # NOTE: Not adjusting t1d last dim (confidence) from sequence mask
        t1d = torch.zeros((1,1,L,21))
        t1d[:,:,:,:21] = nn.one_hot(torch.where(seq == 21, 20, seq), num_classes=21)[None,None]
        
        if self.inf_conf.autoregressive_confidence:
            # Set confidence to 1 where diffusion mask is True, else 1-t/T
            conf = torch.zeros_like(seq).float()
            conf[self.mask_str.squeeze()] = 1.
            conf[~self.mask_str.squeeze()] = 1. - t/self.T
            conf = conf[None,None,...,None]
        else:
            #NOTE: DJ - I don't know what this does or why it's here
            conf = torch.where(self.mask_str.squeeze(), 1., 0.)[None,None,...,None]
        

        t1d = torch.cat((t1d, conf), dim=-1)
        t1d = t1d.float()

        
        ### xyz_t ###
        #############
        if self.preprocess_conf.sidechain_input:
            xyz_t[torch.where(seq == 21, True, False),3:,:] = float('nan')
        else:
            xyz_t[~self.mask_str.squeeze(),3:,:] = float('nan')
        #xyz_t[:,3:,:] = float('nan')

        xyz_t=xyz_t[None, None]
        xyz_t = torch.cat((xyz_t, torch.full((1,1,L,13,3), float('nan'))), dim=3)

        ### t2d ###
        ###########
        t2d = xyz_to_t2d(xyz_t)
        
        ### idx ###
        ###########
        """
        idx = torch.arange(L)[None]
        if ppi_design:
            idx[:,binderlen:] += 200
        """
        # JW Just get this from the contig_mapper now. This handles chain breaks
        idx = torch.tensor(self.contig_map.rf)[None]

        ### alpha_t ###
        ###############
        seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)
        alpha, _, alpha_mask, _ = util.get_torsions(xyz_t.reshape(-1,L,27,3), seq_tmp, TOR_INDICES, TOR_CAN_FLIP, REF_ANGLES)
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
        alpha[torch.isnan(alpha)] = 0.0
        alpha = alpha.reshape(1,-1,L,10,2)
        alpha_mask = alpha_mask.reshape(1,-1,L,10,1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 30)

        ### seq ###
        ###########
        #seq is now one-hot encoded
        #TODO implement handling of Blosum and seq diffusion - NRB/DJ
        seq = torch.nn.functional.one_hot(seq, num_classes=22).float() # [n,I,L,22] 

        #put tensors on device
        msa_masked = msa_masked.to(self.device)
        msa_full = msa_full.to(self.device)
        seq = seq.to(self.device)
        xyz_t = xyz_t.to(self.device)
        idx = idx.to(self.device)
        t1d = t1d.to(self.device)
        t2d = t2d.to(self.device)
        alpha_t = alpha_t.to(self.device)
        
        ### added_features ###
        ######################
        # NB the hotspot input has been removed in this branch. 
        # JW added it back in, using pdb indexing
        if self.preprocess_conf.d_t1d >= 24: # add hotpot residues
            hotspot_tens = torch.zeros(L).float()
            if self.ppi_conf.hotspot_res is None:
                print("WARNING: you're using a model trained on complexes and hotspot residues, without specifying hotspots. If you're doing monomer diffusion this is fine")
            else:
                hotspots = [(i[0],int(i[1:])) for i in self.ppi_conf.hotspot_res]
                hotspot_idx=[]
                for i,res in enumerate(self.contig_map.con_ref_pdb_idx):
                    if res in hotspots:
                        hotspot_idx.append(self.contig_map.hal_idx0[i])
                hotspot_tens[hotspot_idx] = 1.0

            # NB penultimate plane relates to sequence self conditioning. In these models set it to zero.
            t1d=torch.cat((t1d, torch.zeros_like(t1d[...,:1]), hotspot_tens[None,None,...,None].to(self.device)), dim=-1)
        """
        # t1d
        if self.preprocess_conf.d_t1d == 23: # add hotspot residues
            # NRB: Adding in dimension for target hotspot residues
            target_residue_feat = torch.zeros_like(t1d[...,0])[...,None]
            if ppi_design and not target_res is None:
                absolute_idx = [resi+binderlen for resi in target_res]
                target_residue_feat[...,absolute_idx,:] = 1
            t1d = torch.cat((t1d, target_residue_feat), dim=-1)
            t1d = t1d.float()
        """ 

        return msa_masked, msa_full, seq[None], torch.squeeze(xyz_t, dim=0), idx, t1d, t2d, xyz_t, alpha_t
        
    def sample_step(self, *, t, seq_t, x_t, seq_init, final_step, return_extra=False):
        '''Generate the next pose that the model should be supplied at timestep t-1.

        Args:
            t (int): The timestep that has just been predicted
            seq_t (torch.tensor): (L) The sequence at the beginning of this timestep
            x_t (torch.tensor): (L,14,3) The residue positions at the beginning of this timestep
            seq_t (torch.tensor): (L) The initialized sequence used in updating the sequence.
            
        Returns:
            px0: (L,14,3) The model's prediction of x0.
            x_t_1: (L,14,3) The updated positions of the next step.
            seq_t_1: (L) The updated sequence of the next step.
            tors_t_1: (L, ?) The updated torsion angles of the next  step.
            plddt: (L, 1) Predicted lDDT of x0.
        '''
        out = self._preprocess(seq_t, x_t, t)
        msa_masked, msa_full, seq_in, xt_in, idx_pdb, t1d, t2d, xyz_t, alpha_t = self._preprocess(
            seq_t, x_t, t)

        N,L = msa_masked.shape[:2]

        if self.symmetry is not None:
            idx_pdb, self.chain_idx = self.symmetry.res_idx_procesing(res_idx=idx_pdb)

        # decide whether to recycle information between timesteps or not
        if self.inf_conf.recycle_between and t < self.diffuser_conf.aa_decode_steps:
            msa_prev = self.msa_prev
            pair_prev = self.pair_prev
            state_prev = self.state_prev
        else:
            msa_prev = None
            pair_prev = None
            state_prev = None
        with torch.no_grad():
            # So recycling is done a la training
            px0=xt_in
            for _ in range(self.recycle_schedule[t-1]):
                msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = self.model(msa_masked,
                                    msa_full,
                                    seq_in,
                                    px0,
                                    idx_pdb,
                                    t1d=t1d,
                                    t2d=t2d,
                                    xyz_t=xyz_t,
                                    alpha_t=alpha_t,
                                    msa_prev = msa_prev,
                                    pair_prev = pair_prev,
                                    state_prev = state_prev,
                                    t=torch.tensor(t),
                                    return_infer=True,
                                    motif_mask=self.diffusion_mask.squeeze().to(self.device))

        self.msa_prev=msa_prev
        self.pair_prev=pair_prev
        self.state_prev=state_prev
        # prediction of X0 
        _, px0  = self.allatom(torch.argmax(seq_in, dim=-1), px0, alpha)
        px0    = px0.squeeze()[:,:14]

        #sampled_seq = torch.argmax(logits.squeeze(), dim=-1)
        seq_probs   = torch.nn.Softmax(dim=-1)(logits.squeeze()/self.inf_conf.softmax_T)
        sampled_seq = torch.multinomial(seq_probs, 1).squeeze() # sample a single value from each position 
        
        # grab only the query sequence prediction - adjustment for Seq2StrSampler
        sampled_seq = sampled_seq.reshape(N,L,-1)[0,0]

        # Process outputs.
        mask_seq = self.mask_seq
        sampled_seq[mask_seq.squeeze()] = seq_init[
            mask_seq.squeeze()].to(self.device)

        pseq_0 = torch.nn.functional.one_hot(
            sampled_seq, num_classes=22).to(self.device)
        self._log.info(
            f'Timestep {t}, current sequence: { seq2chars(torch.argmax(pseq_0, dim=-1).tolist())}')
        
        if t > final_step:
            x_t_1, seq_t_1, tors_t_1, px0 = self.denoiser.get_next_pose(
                xt=x_t,
                px0=px0,
                t=t,
                diffusion_mask=self.mask_str.squeeze(),
                seq_t=seq_t,
                pseq0=pseq_0,
                align_motif=self.inf_conf.align_motif,
            )
        else:
            x_t_1 = torch.clone(px0).to(x_t.device)
            seq_t_1 = torch.clone(sampled_seq)
            # Dummy tors_t_1 prediction. Not used in final output.
            tors_t_1 = torch.ones((self.mask_str.shape[-1], 10, 2))
            px0 = px0.to(x_t.device)
        if self.symmetry is not None:
            x_t_1, seq_t_1 = self.symmetry.apply_symmetry(x_t_1, seq_t_1)
        if return_extra:
            return px0, x_t_1, seq_t_1, tors_t_1, plddt, logits
        return px0, x_t_1, seq_t_1, tors_t_1, plddt

    def symmetrise_prev_pred(self, px0, seq_in, alpha):
        """
        Method for symmetrising px0 output, either for recycling or for self-conditioning
        """
        _,px0_aa = self.allatom(torch.argmax(seq_in, dim=-1), px0, alpha)
        px0_sym,_ = self.symmetry.apply_symmetry(px0_aa.to('cpu').squeeze()[:,:14], torch.argmax(seq_in, dim=-1).squeeze().to('cpu'))
        px0_sym = px0_sym[None].to(self.device)
        return px0_sym

    def note_checkpointed_design(self):
        pass

    def output_prefix(self, include_extra=True):
        if ( not self.inf_conf.extra_output_prefix or not include_extra ):
            return self.inf_conf.output_prefix
        else:
            return self.inf_conf.output_prefix + self.inf_conf.extra_output_prefix

class Seq2StrSampler(Sampler):
    """
    Model runner for RF_diffusion fixed sequence structure prediction with or w/o MSA
    """
    
    def __init__(self, conf: DictConfig):
        super().__init__(conf)
        raise NotImplementedError("Seq2Str Sampler not implemented yet")
        self.seq2str_conf = conf.seq2str

        # parse MSA from config
        msa, ins   = iu.parse_a3m(self.seq2str_conf.input_a3m) # msa - (N,L) integers  
        self.msa   = torch.from_numpy(msa).long()
        self.query = torch.from_numpy(msa[0]).long()
        self.ins   = torch.from_numpy(ins).long()

        self.mask_seq = torch.zeros_like(self.query).bool()
        self.mask_str = self.mask_seq.clone()

        self.L = msa.shape[-1]

        # if complex modelling, get chain lengths 
        # Assumes chains are in same order as sequence 
        if self.seq2str_conf.chain_lengths:
            self.chain_lengths = [int(a) for a in self.seq2str_conf.chain_lengths.split(',')] 
        else:
            self.chain_lengths = [self.L]
        assert sum(self.chain_lengths) == self.L, 'input chain lengths did not sum to query sequence length'
        
        if self.diffuser_conf.partial_T:
            assert self.diffuser_conf.partial_T <= self.diffuser_conf.T
            self.t_step_input = int(self.diffuser_conf.partial_T)
        else:
            self.t_step_input = int(self.diffuser_conf.T)
    
    def sample_init(self): 
        """
        Create an initialized set of residues to go through the model
        """
        # initial structure is completely unknown
        # Diffuser will take care of filling in the atoms when protein is diffused for the first time 
        x_nan = torch.full((self.L, 27,3), np.nan)
        x_nan = get_init_xyz(x_nan[None,None]).squeeze()
        atom_mask = torch.full((self.L,27), False)
        
        seq_T = self.query # query sequence
        self.diffusion_mask = torch.full((self.L,),False)

        # Setup denoiser
        self.denoiser = self.construct_denoiser(self.L)

        fa_stack,_,_,_ = self.diffuser.diffuse_pose(
            x_nan,
            torch.clone(seq_T),  # TODO: Check if copy is needed.
            atom_mask.squeeze(),
            diffusion_mask=self.diffusion_mask,
            t_list=[self.diffuser_conf.T],
            diffuse_sidechains=self.preprocess_conf.sidechain_input,
            include_motif_sidechains=self.preprocess_conf.motif_sidechain_input)
        
        # the most diffused set of atoms is the last one 
        xT = torch.clone(fa_stack[-1].squeeze()[:,:14,:]) 
        
        # from the outside it's returned as seq_T
        return xT, seq_T


    def _preprocess(self, seq_t, xyz_t, t):

        """
        Function to prepare inputs to diffusion model - but now with MSA + MSA statistics etc 
        
            seq (L) integer sequence 
        """
        msa = self.msa 
        ins = self.ins
        N,L = msa.shape        

        #### Build template features #### 

        # rename just so we dont get confused - template is indeed at timestep t 
        xyz_template = xyz_t

        #### Build MSA features #### 
        params = {
        "MINTPLT" : 0,
        "MAXTPLT" : 5,
        "MINSEQ"  : 1,
        "MAXSEQ"  : 1024,
        "MAXLAT"  : 128,
        "CROP"    : 256,
        "BLOCKCUT": 5,
        "ROWS"    : 1,
        "SEQID"   : 95.0,
        "MAXCYCLE": 1,
        }
        seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = data_loader.MSAFeaturize(msa, ins, params)
        msa_masked = msa_seed
        msa_full   = msa_extra 

        
        ### t1d ### 
        ########### 
        # seq 21, conf 1,
        t1d = torch.zeros((1,self.L, 22+9))

        # seq query sequence one hot encoding
        t1d[:,:,self.query] = 1  

        # First template (the diffused) has zero confidence 
        t1d[:1,:,21] = 0
        
        # SS 
        # all SS features are set to masked (26th element, 1-hot)
        t1d[:1,:,25] = 1 

        # Add global timestep. 
        t1d[:1,:,26] = 1-t/self.diffuser.T

        # Add chi angle timestep, same as global timestep
        t1d[:1,:,27] = 1-t/self.diffuser.T 

        # Add contacting residues.
        t1d[:1,:,28] = torch.zeros(1,L)

        # Add diffused or not (1 = not, 0 = diffused)
        t1d[:1,:,29] = torch.zeros(1,L)

        # Feature indicating whether this is a real homologous template 
        # or just the diffused input 
        # (1=template, 0=diffused)
        t1d[:1,:,30] = torch.zeros(1,L)

        t1d = t1d[None]


        ### t2d ###
        ###########
        t2d = xyz_to_t2d(xyz_template[None,None])   # (B,T,L,L,3)
        t2d = torch.cat((t2d, torch.zeros((1,1,L,L,3))), dim=-1) #three extra dimensions: adjacent, not adjacent, masked
        t2d[...,-1] = 1 # set whole block adjacency to mask token
        # t2d[...,0] = 1  # for the buggy model, set this feature to zero instead of 2 


        ### idx ###
        ###########
        idx = torch.arange(L)[None]
        # index jumps for chains 
        cur_sum=0
        for Lchain in self.chain_lengths:
            idx[:,cur_sum+Lchain:] += 200
            cur_sum += Lchain

        ### alpha_t ###
        ###############
        # get the torsions 
        xyz_template =xyz_template[None, None]
        xyz_template = torch.cat((xyz_template, torch.full((1,1,L,13,3), float('nan'))), dim=3)
        

        alpha, _, alpha_mask, _ = util.get_torsions(xyz_template.reshape(-1,L,27,3), self.query[None], TOR_INDICES, TOR_CAN_FLIP, REF_ANGLES)
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
        alpha[torch.isnan(alpha)] = 0.0
        alpha = alpha.reshape(1,-1,L,10,2)
        alpha_mask = alpha_mask.reshape(1,-1,L,10,1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 30)
        
        # device placement 
        msa_masked = msa_masked.to(self.device)
        msa_full = msa_full.to(self.device)
        xyz_template = xyz_template.to(self.device)
        idx = idx.to(self.device)
        t1d = t1d.to(self.device)
        t2d = t2d.to(self.device)
        alpha_t = alpha_t.to(self.device)
        
        return msa_masked, msa_full, self.query[None].clone().to(self.device), torch.squeeze(xyz_template, dim=0), idx, t1d, t2d, xyz_template, alpha_t

class NRBStyleSelfCond(Sampler):
    """
    Model Runner for self conditioning in the style attempted by NRB in
    frame_sql2_pdb_data_T200_sinusoidal_frozenmotif_lddt_distog_noseq_physics_selfcond_lowlr_train_session2022-10-06_1665075957.8513756
    """

    def sample_step(self, *, t, seq_t, x_t, seq_init, final_step, skip_diffuser=False):
        '''
        Generate the next pose that the model should be supplied at timestep t-1.

        Args:
            t (int): The timestep that has just been predicted
            seq_t (torch.tensor): (L) The sequence at the beginning of this timestep
            x_t (torch.tensor): (L,14,3) The residue positions at the beginning of this timestep
            seq_t (torch.tensor): (L) The initialized sequence used in updating the sequence.

        Returns:
            px0: (L,14,3) The model's prediction of x0.
            x_t_1: (L,14,3) The updated positions of the next step.
            seq_t_1: (L) The updated sequence of the next step.
            tors_t_1: (L, ?) The updated torsion angles of the next  step.
            plddt: (L, 1) Predicted lDDT of x0.
        '''

        hax.hacks().maybe_save_state( t, seq_t, x_t, seq_init, self )

        skip = hax.hacks().should_skip_rf2_frame(t)

        if ( not skip ):

            msa_masked, msa_full, seq_in, xt_in, idx_pdb, t1d, t2d, xyz_t, alpha_t = self._preprocess(
                seq_t, x_t, t)

            B,N,L = xyz_t.shape[:3]
            if (t < self.diffuser.T) and (t != self.diffuser_conf.partial_T):
                ic('Providing Self Cond')
                    
                zeros = torch.zeros(B,1,L,24,3).float().to(xyz_t.device)
                xyz_t = torch.cat((self.prev_pred.unsqueeze(1),zeros), dim=-2) # [B,T,L,27,3]

                t2d_44   = xyz_to_t2d(xyz_t) # [B,T,L,L,44]
            else:
                xyz_t = torch.zeros_like(xyz_t)
                t2d_44   = torch.zeros_like(t2d[...,:44])
            # No effect if t2d is only dim 44
            t2d[...,:44] = t2d_44

            if self.symmetry is not None:
                idx_pdb, self.chain_idx = self.symmetry.res_idx_procesing(res_idx=idx_pdb)
            with torch.no_grad():
                px0=xt_in
                for rec in range(self.recycle_schedule[t-1]):
                    msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = self.model(msa_masked,
                                    msa_full,
                                    seq_in,
                                    px0,
                                    idx_pdb,
                                    t1d=t1d,
                                    t2d=t2d,
                                    xyz_t=xyz_t,
                                    alpha_t=alpha_t,
                                    msa_prev = None,
                                    pair_prev = None,
                                    state_prev = None,
                                    t=torch.tensor(t),
                                    return_infer=True,
                                    motif_mask=self.diffusion_mask.squeeze().to(self.device))




                    if self.symmetry is not None and self.inf_conf.symmetric_self_cond:
                        px0 = self.symmetrise_prev_pred(px0=px0,seq_in=seq_in, alpha=alpha)[:,:,:3]
                    # To permit 'recycling' within a timestep, in a manner akin to how this model was trained
                    # Aim is to basically just replace the xyz_t with the model's last px0, and to *not* recycle the state, pair or msa embeddings
                    if rec < self.recycle_schedule[t-1] -1:
                        zeros = torch.zeros(B,1,L,24,3).float().to(xyz_t.device)
                        xyz_t = torch.cat((px0.unsqueeze(1),zeros), dim=-2) # [B,T,L,27,3]

                        t2d   = xyz_to_t2d(xyz_t) # [B,T,L,L,44]
                        px0=xt_in

            hax.hacks().set_last_rf2_vars([msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt, seq_in])

            self.prev_pred = torch.clone(px0) 
        else:
            msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt, seq_in = hax.hacks().get_last_rf2_vars() 

        

        # prediction of X0
        _, px0  = self.allatom(torch.argmax(seq_in, dim=-1), px0, alpha)
        px0    = px0.squeeze()[:,:14]
        #sampled_seq = torch.argmax(logits.squeeze(), dim=-1)
        seq_probs   = torch.nn.Softmax(dim=-1)(logits.squeeze()/self.inf_conf.softmax_T)
        sampled_seq = torch.multinomial(seq_probs, 1).squeeze() # sample a single value from each position

        # Process outputs.
        mask_seq = self.mask_seq
        sampled_seq[mask_seq.squeeze()] = seq_init[
            mask_seq.squeeze()].to(self.device)

        pseq_0 = torch.nn.functional.one_hot(
            sampled_seq, num_classes=22).to(self.device)
        self._log.info(
            f'Timestep {t}, current sequence: { seq2chars(torch.argmax(pseq_0, dim=-1).tolist())}')


        if t > final_step:
            x_t_1, seq_t_1, tors_t_1, px0 = self.denoiser.get_next_pose(
                xt=x_t,
                px0=px0,
                t=t,
                diffusion_mask=self.mask_str.squeeze(),
                seq_t=seq_t,
                pseq0=pseq_0,
                align_motif=self.inf_conf.align_motif,
                skip_diffuser=skip_diffuser
            )
        else:
            x_t_1 = torch.clone(px0).to(x_t.device)
            seq_t_1 = torch.clone(sampled_seq)
            # Dummy tors_t_1 prediction. Not used in final output.
            tors_t_1 = torch.ones((self.mask_str.shape[-1], 10, 2))
            px0 = px0.to(x_t.device)
        if self.symmetry is not None:
            x_t_1, seq_t_1 = self.symmetry.apply_symmetry(x_t_1, seq_t_1)
        return px0, x_t_1, seq_t_1, tors_t_1, plddt
        
class JWStyleSelfCond(Sampler):
    """
    Model Runner for self conditioning in the style attempted by JW in
    frame_sql2_pdb_data_T200_sinusoidal_frozenmotif_lddt_distog_noseq_physics_selfcond_lowlr_train_session2022-10-06_1665075957.8513756
    """
    def __init__(self, conf: DictConfig):
        super().__init__(conf)
        self.self_cond = self.inf_conf.use_jw_selfcond 
        if not self.self_cond:
            print(" ", "*" * 100, " ", "WARNING: You're using the JWStyleSelfCond sampler, but inference.use_jw_selfcond is set to False. Is this intentional?", " ", "*" * 100, " ", sep=os.linesep)
    def sample_step(self, *, t, seq_t, x_t, seq_init, final_step):
        '''
        Generate the next pose that the model should be supplied at timestep t-1.

        Args:
            t (int): The timestep that has just been predicted
            seq_t (torch.tensor): (L) The sequence at the beginning of this timestep
            x_t (torch.tensor): (L,14,3) The residue positions at the beginning of this timestep
            seq_t (torch.tensor): (L) The initialized sequence used in updating the sequence.
            
        Returns:
            px0: (L,14,3) The model's prediction of x0.
            x_t_1: (L,14,3) The updated positions of the next step.
            seq_t_1: (L) The updated sequence of the next step.
            tors_t_1: (L, ?) The updated torsion angles of the next  step.
            plddt: (L, 1) Predicted lDDT of x0.
        '''
        out = self._preprocess(seq_t, x_t, t)
        msa_masked, msa_full, seq_in, xt_in, idx_pdb, t1d, t2d, xyz_t, alpha_t = self._preprocess(
            seq_t, x_t, t)
        # Save inputs for next timestep
        self.t1d = t1d[:,:1]
        self.t2d = t2d[:,:1]
        self.alpha = alpha_t
        N,L = msa_masked.shape[:2]

        if self.symmetry is not None:
            idx_pdb, self.chain_idx = self.symmetry.res_idx_procesing(res_idx=idx_pdb)
        with torch.no_grad():
            px0=xt_in
            for _ in range(self.recycle_schedule[t-1]):
                msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = self.model(msa_masked,
                                    msa_full,
                                    seq_in,
                                    px0,
                                    idx_pdb,
                                    t1d=t1d,
                                    t2d=t2d,
                                    xyz_t=xyz_t,
                                    alpha_t=alpha_t,
                                    msa_prev = self.msa_prev,
                                    pair_prev = self.pair_prev,
                                    state_prev = self.state_prev,
                                    t=torch.tensor(t),
                                    return_infer=True,
                                    motif_mask=self.diffusion_mask.squeeze().to(self.device))

        self.msa_prev=msa_prev
        self.pair_prev=pair_prev
        self.state_prev=state_prev
        self.prev_pred = torch.clone(px0)
        # prediction of X0 
        _, px0  = self.allatom(torch.argmax(seq_in, dim=-1), px0, alpha)
        px0    = px0.squeeze()[:,:14]
        #sampled_seq = torch.argmax(logits.squeeze(), dim=-1)
        seq_probs   = torch.nn.Softmax(dim=-1)(logits.squeeze()/self.inf_conf.softmax_T)
        sampled_seq = torch.multinomial(seq_probs, 1).squeeze() # sample a single value from each position 

        # grab only the query sequence prediction - adjustment for Seq2StrSampler
        sampled_seq = sampled_seq.reshape(N,L,-1)[0,0]

        # Process outputs.
        mask_seq = self.mask_seq
        sampled_seq[mask_seq.squeeze()] = seq_init[
            mask_seq.squeeze()].to(self.device)

        pseq_0 = torch.nn.functional.one_hot(
            sampled_seq, num_classes=22).to(self.device)
        self._log.info(
            f'Timestep {t}, current sequence: { seq2chars(torch.argmax(pseq_0, dim=-1).tolist())}')

        if t > final_step:
            x_t_1, seq_t_1, tors_t_1, px0 = self.denoiser.get_next_pose(
                xt=x_t,
                px0=px0,
                t=t,
                diffusion_mask=self.mask_str.squeeze(),
                seq_t=seq_t,
                pseq0=pseq_0,
                align_motif=self.inf_conf.align_motif,
            )
        else:
            x_t_1 = torch.clone(px0).to(x_t.device)
            seq_t_1 = torch.clone(sampled_seq)
            # Dummy tors_t_1 prediction. Not used in final output.
            tors_t_1 = torch.ones((self.mask_str.shape[-1], 10, 2))
            px0 = px0.to(x_t.device)
        if self.symmetry is not None:
            x_t_1, seq_t_1 = self.symmetry.apply_symmetry(x_t_1, seq_t_1)
        return px0, x_t_1, seq_t_1, tors_t_1, plddt

    def _preprocess(self, seq, xyz_t, t):
        msa_masked, msa_full, seq_in, xt_in, idx_pdb, t1d, t2d, xyz_t, alpha_t = super()._preprocess(seq, xyz_t, t)
        t1d = torch.cat((t1d, torch.zeros_like(t1d[...,:1])), dim=-1)
        if t != self.T and self.self_cond:
            # add last step
            xyz_prev_padded = torch.full_like(xyz_t, float('nan'))
            xyz_prev_padded[:,:,:,:3,:] = self.prev_pred[None] 
            xyz_t = torch.cat((xyz_t, xyz_prev_padded), dim=1)
            t1d = t1d.repeat(1,2,1,1)
            t1d[:,1,:,21] = self.t1d[...,21]
            t1d[:,1,:,22] = 1
            t2d_temp = xyz_to_t2d(xyz_prev_padded).to(self.device, non_blocking=True)
            t2d = torch.cat((t2d, t2d_temp), dim=1)
            alpha_temp = torch.zeros_like(alpha_t).to(self.device, non_blocking=True)
            alpha_t = torch.cat((alpha_t, alpha_temp), dim=1)
        return msa_masked, msa_full, seq_in, xt_in, idx_pdb, t1d, t2d, xyz_t, alpha_t

class ScaffoldedSampler(NRBStyleSelfCond):
    """ 
    Model Runner for Scaffold-Constrained diffusion
    """
    def __init__(self, conf: DictConfig):
        """
        Initialize scaffolded sampler, which inherits from Sampler
        """
        super().__init__(conf)
        # initialize BlockAdjacency sampling class
        self.blockadjacency = iu.BlockAdjacency(conf.scaffoldguided, conf.inference.num_designs)

        if conf.scaffoldguided.target_pdb:
            self.target = iu.Target(conf.scaffoldguided, conf.ppi.hotspot_res)
            self.target_pdb = self.target.get_target()
            if conf.scaffoldguided.target_ss is not None:
                self.target_ss = torch.load(conf.scaffoldguided.target_ss).long()
                self.target_ss = torch.nn.functional.one_hot(self.target_ss, num_classes=4)
            if conf.scaffoldguided.target_adj is not None:
                self.target_adj = torch.load(conf.scaffoldguided.target_adj).long()
                self.target_adj=torch.nn.functional.one_hot(self.target_adj, num_classes=3)
        else:
            self.target = None


    def note_checkpointed_design(self):
        self.blockadjacency.increment_scaffold()

    def sample_init(self):
        """
        Wrapper method for taking ss + adj, and outputting xt, seq_t
        """

        self.L, self.ss, self.adj, scaff_name, ba_contig = self.blockadjacency.get_scaffold()
        hax.hacks().add_scores_to_run(None, dict(scaff_name=scaff_name))
        self.adj = nn.one_hot(self.adj.long(), num_classes=3)
        xT = torch.full((self.L, 27,3), np.nan)
        xT = get_init_xyz(xT[None,None]).squeeze()
        seq_T = torch.full((self.L,),21)
        self.diffusion_mask = torch.full((self.L,),False)
        atom_mask = torch.full((self.L,27), False)
        self.binderlen=self.L
        # for ppi
        preserve_center = hax.hacks().preserve_center
        self.binder_L = np.copy(self.L)
        if self.target:
            is_target_mask = torch.ones(len(self.target_pdb['xyz']), dtype=bool)
            if ( hax.hacks().force_contig ):
                for idx, (chain, seqpos) in enumerate(self.target_pdb['pdb_idx']):
                    is_target_mask[idx] = chain == "B"
            target_L = is_target_mask.sum()
            hax.hacks().center_target(self.target_pdb, self.L + target_L)
            target_xyz = torch.full((target_L, 27, 3), np.nan)
            target_xyz[:,:14,:] = torch.from_numpy(self.target_pdb['xyz'][is_target_mask])
            xT = torch.cat((xT, target_xyz), dim=0)
            seq_T = torch.cat((seq_T, torch.from_numpy(self.target_pdb['seq'][is_target_mask])), dim=0)
            self.diffusion_mask = torch.cat((self.diffusion_mask, torch.full((target_L,), True)),dim=0)
            mask_27 = torch.full((target_L, 27), False)
            mask_27[:,:14] = torch.from_numpy(self.target_pdb['mask'][is_target_mask])
            atom_mask = torch.cat((atom_mask, mask_27), dim=0)
            self.L += target_L
            
        # make contigmap object
        if self.target:
            contig = []
            for idx,i in enumerate(self.target_pdb['pdb_idx'][:-1]):
                if idx==0:
                    start=i[1]               
                if i[1] + 1 != self.target_pdb['pdb_idx'][idx+1][1] or i[0] != self.target_pdb['pdb_idx'][idx+1][0]:
                    contig.append(f'{i[0]}{start}-{i[1]},0 ')
                    start = self.target_pdb['pdb_idx'][idx+1][1]
            contig.append(f"{self.target_pdb['pdb_idx'][-1][0]}{start}-{self.target_pdb['pdb_idx'][-1][1]},0 ")
            contig.append(f"{self.binderlen}-{self.binderlen}")
            contig = ["".join(contig)]
        else:
            contig = [f"{self.binderlen}-{self.binderlen}"]


        if ( hax.hacks().force_contig ):
            contig = self.contig_conf.contigs
            if ( ba_contig is not None ):
                contig = ba_contig
            print("Using contig:", contig)
            self.contig_map=ContigMap(self.target_pdb, contig)

            full_xyz = torch.full((len(self.target_pdb['xyz']), 27, 3), np.nan)
            full_xyz[:,:14,:] = torch.from_numpy(self.target_pdb['xyz'])
            assert(torch.allclose(xT[self.contig_map.hal_idx0][is_target_mask], full_xyz[self.contig_map.ref_idx0][is_target_mask], equal_nan=True))
            xT[self.contig_map.hal_idx0] = full_xyz[self.contig_map.ref_idx0]

            full_seq = torch.from_numpy(self.target_pdb['seq'])
            assert(torch.allclose(seq_T[self.contig_map.hal_idx0][is_target_mask], full_seq[self.contig_map.ref_idx0][is_target_mask], equal_nan=True ))
            seq_T[self.contig_map.hal_idx0] = full_seq[self.contig_map.ref_idx0]

            full_mask_27 = torch.full((len(self.target_pdb['xyz']), 27), False)
            full_mask_27[:,:14] = torch.from_numpy(self.target_pdb['mask'])
            assert(torch.all(atom_mask[self.contig_map.hal_idx0][is_target_mask] == full_mask_27[self.contig_map.ref_idx0][is_target_mask]))
            atom_mask[self.contig_map.hal_idx0] = full_mask_27[self.contig_map.ref_idx0]


            self.diffusion_mask = torch.from_numpy(self.contig_map.inpaint_str)[None,:]

        else:
            self.contig_map=ContigMap(self.target_pdb, contig)


        self.mappings = self.contig_map.get_mappings()                
        self.hotspot_0idx=iu.get_idx0_hotspots(self.mappings, self.ppi_conf, self.binderlen)
        # Now initialise potential manager here. This allows variable-length binder design
        self.potential_manager = PotentialManager(self.potential_conf,
                                                  self.ppi_conf,
                                                  self.diffuser_conf,
                                                  self.inf_conf,
                                                  self.hotspot_0idx,
                                                  self.binderlen)


        self.mask_seq = torch.clone(self.diffusion_mask)
        self.mask_str = torch.clone(self.diffusion_mask)


        self.chain_idx=['A' if i < self.binderlen else 'B' for i in range(self.L)]
        # Diffuse the contig-mapped coordinates 
        if self.diffuser_conf.partial_T:
            assert self.diffuser_conf.partial_T <= self.diffuser_conf.T
            self.t_step_input = int(self.diffuser_conf.partial_T)
        else:
            self.t_step_input = int(self.diffuser_conf.T)
        t_list = np.arange(1, self.t_step_input+1)
        fa_stack, aa_masks, xyz_true = self.diffuser.diffuse_pose(
            xT,
            torch.clone(seq_T),  # TODO: Check if copy is needed.
            atom_mask.squeeze(),
            diffusion_mask=self.diffusion_mask.squeeze(),
            t_list=t_list,
            diffuse_sidechains=self.preprocess_conf.sidechain_input,
            include_motif_sidechains=self.preprocess_conf.motif_sidechain_input,
            preserve_center=preserve_center)





        # Setup denoiser 
        is_motif = self.mask_seq.squeeze()
        is_shown_at_t = torch.tensor(aa_masks[-1])
        if ( hax.hacks().dont_show_any_input ):
            is_shown_at_t[:] = False
        visible = is_motif | is_shown_at_t
        if self.diffuser_conf.partial_T:
            seq_t[visible] = seq_orig[visible]

        self.denoiser = self.construct_denoiser(self.L, visible=visible)


        xT = torch.clone(fa_stack[-1].squeeze()[:,:14,:])


        hax.hacks().init_motifs_closer(xT, self.binderlen, self.diffusion_mask.squeeze())


        return xT, seq_T
    
    def _preprocess(self, seq, xyz_t, t):
        msa_masked, msa_full, seq, xyz_prev, idx_pdb, t1d, t2d, xyz_t, alpha_t = super()._preprocess(seq, xyz_t, t, repack=False)
        
        # Now just need to tack on ss/adj
        assert self.preprocess_conf.d_t1d == 28, "The checkpoint you're using hasn't been trained with SS/block adjacency features"
        assert self.preprocess_conf.d_t2d == 47, "The checkpoint you're using hasn't been trained with SS/block adjacency features"
        
        if self.target:
            blank_ss = torch.nn.functional.one_hot(torch.full((self.L-self.binderlen,), 3), num_classes=4)
            full_ss = torch.cat((self.ss, blank_ss), dim=0)
            if self._conf.scaffoldguided.target_ss is not None:
                full_ss[self.binderlen:] = self.target_ss
        else:
            full_ss = self.ss


        ### t2d ###
        ###########

        if self.d_t2d == 47:
            if self.target:
                full_adj = torch.zeros((self.L, self.L, 3))
                full_adj[:,:,-1] = 1. #set to mask
                full_adj[:self.binderlen, :self.binderlen] = self.adj
                if self._conf.scaffoldguided.target_adj is not None:
                    full_adj[self.binderlen:,self.binderlen:] = self.target_adj
            else:
                full_adj = self.adj


        hax.hacks().do_constraints( t, full_ss, full_adj, self.binderlen, self )

        # with open("full_adj.pt", "wb") as f:
        #     torch.save(full_adj, f)

        t1d=torch.cat((t1d, full_ss[None,None].to(self.device)), dim=-1)
        t1d = t1d.float()
        t2d=torch.cat((t2d, full_adj[None,None].to(self.device)),dim=-1)

        ### idx ###
        ###########
        if self.target:
            idx_pdb[:,self.binderlen:] += 200

        ### msa N/C ###
        ###############
        msa_masked[...,-2:] = 0
        msa_masked[...,0,-2] = 1 # N ter token
        msa_masked[...,self.binderlen-1,-1] = 1 # C ter token

        msa_full[...,-2:] = 0
        msa_full[...,0,-2] = 1 # N ter token
        msa_full[...,self.binderlen-1,-1] = 1 # C ter token



        return msa_masked, msa_full, seq, xyz_prev, idx_pdb, t1d, t2d, xyz_t, alpha_t
