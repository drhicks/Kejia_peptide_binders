#!/bin/bash
#SBATCH -p gpu
#SBATCH -J self_cond_contseq_str_diff
#SBATCH --cpus-per-task=4
#SBATCH --mem=400g
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=4
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate /home/nrbennet/miniconda3/envs/SE3nv

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12345

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Parameters from DJ and JW's train script
crop=384
max_len=384
max_complex_chain=250
wandb_pref='ContSeqSelfCond_plus_L2_losses'
so3_type='igso3'
euclidean_schedule='linear'
b0=0.01 # 1e-2
bT=0.07 # 7e-2 

chi_type='circular'
uncond_diff=0.2
# End parameters from DJ and JW's train script

prob_self_cond=0.5

seqdiff='continuous'

if [ $seqdiff == 'none' ]
then
        echo Doing Autoregressive sequence decoding
        srun python -u train_multi_deep.py -p_drop 0.15 -accum 8 -crop $crop -w_disp 0.5 -w_frame_dist 1.0 -w_aa 3 -w_blen 0.0 -w_bang 0.0 -w_lj 0.0 -w_hb 0.0 -w_str 0.0 -maxlat 256 -maxseq 1024 -num_epochs 200 -lr 0.0005 -n_main_block 32 -seed 42 -seqid 150.0 -mintplt 1 -use_H -max_length $max_len -max_complex_chain $max_complex_chain -task_names diff,seq2str -task_p 1.0,0.0 -diff_T 200 -aa_decode_steps 40 -wandb_prefix $wandb_pref -diff_so3_type $so3_type -diff_chi_type $chi_type -use_tschedule -maxcycle 1 -prob_self_cond 0.5 -diff_b0 $b0 -diff_bT $bT -diff_schedule_type $euclidean_schedule -prob_self_cond $prob_self_cond

elif [ $seqdiff == 'uniform' ]
then
        echo Doing uniform discrete sequence diffusion

        seqdiff_b0=0.001
        seqdiff_bT=0.2  # The slow schedule Brian T and I agreed upon

        # Schedule is one of [linear, cosine, exponential]
        seqdiff_schedule='linear'

        seqdiff_lambda=1

        srun python -u train_multi_deep.py -p_drop 0.15 -accum 8 -crop $crop -w_disp 0.5 -w_frame_dist 1.0 -w_aa 3 -w_blen 0.0 -w_bang 0.0 -w_lj 0.0 -w_hb 0.0 -w_str 0.0 -maxlat 256 -maxseq 1024 -num_epochs 200 -lr 0.0005 -n_main_block 32 -seed 42 -seqid 150.0 -mintplt 1 -use_H -max_length $max_len -max_complex_chain $max_complex_chain -task_names diff,seq2str -task_p 1.0,0.0 -diff_T 200 -aa_decode_steps 40 -wandb_prefix $wandb_pref -diff_so3_type $so3_type -diff_chi_type $chi_type -use_tschedule -maxcycle 1 -prob_self_cond 0.5 -diff_b0 $b0 -diff_bT $bT -diff_schedule_type $euclidean_schedule -seqdiff $seqdiff -seqdiff_b0 $seqdiff_b0 -seqdiff_bT $seqdiff_bT -seqdiff_schedule $seqdiff_schedule -seqdiff_lambda $seqdiff_lambda -prob_self_cond $prob_self_cond

elif [ $seqdiff == 'continuous' ]
then
        echo Doing continuous analog bit sequence diffusion

        seqdiff_b0=0.01
        seqdiff_bT=0.07 # Same as for Euclidean diffusion

        w_aa=3 # I have no idea what a good value for this is - NRB 

        # Schedule is one of [linear, cosine, exponential]
        seqdiff_schedule='linear'

        srun python -u train_multi_deep.py -p_drop 0.15 -accum 8 -crop $crop -w_disp 0.5 -w_frame_dist 1.0 -w_aa $w_aa -w_blen 0.0 -w_bang 0.0 -w_lj 0.0 -w_hb 0.0 -w_str 0.0 -maxlat 256 -maxseq 1024 -num_epochs 200 -lr 0.0005 -n_main_block 32 -seed 42 -seqid 150.0 -mintplt 1 -use_H -max_length $max_len -max_complex_chain $max_complex_chain -task_names diff,seq2str -task_p 1.0,0.0 -diff_T 200 -aa_decode_steps 40 -wandb_prefix $wandb_pref -diff_so3_type $so3_type -diff_chi_type $chi_type -use_tschedule -maxcycle 1 -prob_self_cond 0.5 -diff_b0 $b0 -diff_bT $bT -diff_schedule_type $euclidean_schedule -seqdiff $seqdiff -seqdiff_b0 $seqdiff_b0 -seqdiff_bT $seqdiff_bT -seqdiff_schedule $seqdiff_schedule -prob_self_cond $prob_self_cond

fi
