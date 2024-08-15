import pyrosetta
import sys
import os

pyrosetta.init("-beta -mute all")

pdb_file = sys.argv[1]
pdb_base = os.path.basename(pdb_file).replace(".pdb", "")

pose=pyrosetta.pose_from_pdb(pdb_file)

script='/home/drhicks1/scripts/Kejia_peptide_binders/bcov_rf_diffusion_24_04_12_tied_mpnn/run_inference.py'
#each trajectory will produce 10 outputs due to the 9 timewarp_subtrajectories
#also we need to stop at T=20 given our timewarp_subtrajectories
num_tries=50
#becasue of our frameskips, our partial_T needs to satisfy these criteria
#_or_((t_<_185)_and_(t_>_0)_and_t%20!=10)
#ie we only call RF to make prediction if partial_T % 20 = 10, so 30,50,70,90,etc..
partial_Ts = [30, 50, 70, 90]
noises = [0.5, 1]

contigA = f"{pose.chain_end(1)}-{pose.chain_end(1)}"
contigB = f"B{pose.chain_begin(2)}-{pose.chain_end(2)}"

ckpts = ["/net/databases/diffusion/models/hotspot_models/BFF_4.pt",
          "/net/databases/diffusion/models/hotspot_models/BFF_6.pt",
          "/net/databases/diffusion/models/hotspot_models/BFF_9.pt"]

for partial_T in partial_Ts:
     for noise in noises:
          for ckpt in ckpts:
               BFF=os.path.basename(ckpt).replace(".pt", "")
               out_folder=f"diffusion_outputs/{pdb_base}_{BFF}_{partial_T}_{str(noise).replace('.', '')}"

               print(f"apptainer exec -B /home/drhicks1 "\
                    +f"/software/containers/SE3nv.sif python "\
                    +f"{script} --config-name=base "\
                    +f"inference.output_prefix={out_folder} "\
                    +f"inference.input_pdb={pdb_file} "\
                    +f"ppi.no_hotspots=True "\
                    +f"inference.ckpt_path={ckpt} "\
                    +f"inference.num_designs={num_tries} "\
                    +f"inference.stop_after_x_successes={num_tries} "\
                    +f"denoiser.noise_scale_ca={noise} "\
                    +f"denoiser.noise_scale_frame={noise} "\
                    +f"contigmap.contigs=[\\'{contigA},0\\ {contigB}\\'] "\
                    +f"bcov_hacks.force_contig=True "\
                    +f"bcov_hacks.output_style=mpnn-insta "\
                    +f"bcov_hacks.no_regular_output=False "\
                    +f"bcov_hacks.silent=True "\
                    +f"+inference.write_trajectory=False "\
                    +f"bcov_hacks.rf2_frameskip=\"'(t==198_or_t==196)_or_((t_<_195)_and_(t_>_185)_and_t%5!=0)_or_((t_<_185)_and_(t_>_0)_and_t%20!=10)'\" "\
                    +f"bcov_hacks.timewarp_subtrajectories=[[20,70],[20,70,20],[20,50],[20,50,20],[20,30],[20,30,20]] "\
                    +f"bcov_hacks.really_slow_mpnn_nstruct=1 "\
                    +f"inference.final_step=20 "\
                    +f"diffuser.partial_T={partial_T}")

#apptainer exec -B /home/drhicks1 --nv
#needed to run on gpu?
