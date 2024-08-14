#!/bin/bash 

complex_pdb=${1}
base=$(basename $complex_pdb ".pdb")

contigA=${2}
contigB=${3}

#TODO

#NOTES
#full diffusion is really slow, in order to decrease the cpu seconds per pdb,
#we take each diffusion output and generate an ensemble
#we do this by:
#1. we generate 10 fast partial diffusion outputs

# i/o 
ckpt=$4
BFF=$(basename $ckpt ".pt")
# script='/home/bcov/sc/diffusion/st/23_01_04_time_warp/rf_diffusion/run_inference.py'
script='/home/bcov/sc/diffusion/st/23_01_20_really_slow_mpnn/rf_diffusion/run_inference.py'


#reduce these values to sample less
num_tries=500 

out_folder="diffusion_outputs/${base}_${BFF}_"

apptainer exec -B /home/bcov -B /home/drhicks1 --nv \
     /software/containers/SE3nv.sif python \
     $script --config-name=base \
     inference.output_prefix=$out_folder \
     inference.input_pdb=$complex_pdb \
     ppi.no_hotspots=True \
     inference.final_step=20 \
     inference.ckpt_path=$ckpt \
     inference.num_designs=$num_tries \
     inference.stop_after_x_successes=$num_tries \
     denoiser.noise_scale_ca=0 \
     denoiser.noise_scale_frame=0 \
     contigmap.contigs=[\'$contigA,0\ $contigB\'] \
     bcov_hacks.force_contig=True \
     bcov_hacks.rf2_frameskip='"(t==198_or_t==196)_or_((t_<_195)_and_(t_>_185)_and_t%5!=0)_or_((t_<_185)_and_(t_>_0)_and_t%20!=10)"' \
     bcov_hacks.output_style=mpnn-insta \
     bcov_hacks.no_regular_output=False \
     bcov_hacks.silent=True \
     +inference.write_trajectory=False \
     bcov_hacks.timewarp_subtrajectories=[[20,70],[20,50],[20,30]] \
     bcov_hacks.really_slow_mpnn_nstruct=1

#bcov_hacks.timewarp_subtrajectories=[[20,70],[20,70,20],[20,50],[20,50,20],[20,30],[20,30,20]] \
