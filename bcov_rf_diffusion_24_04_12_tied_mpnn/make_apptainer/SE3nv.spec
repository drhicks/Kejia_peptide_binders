Bootstrap: docker
From: ubuntu
IncludeCmd: yes

%setup
rsync -a --no-g --no-o /home/drhicks1/scripts/Kejia_peptide_binders/bcov_rf_diffusion_24_04_12_tied_mpnn/SE3Transformer/ $APPTAINER_ROOTFS/SE3Transformer/

%files
/etc/localtime
/etc/hosts
/etc/apt/sources.list
/archive/software/Miniconda3-latest-Linux-x86_64.sh /opt/miniconda.sh

%post
# Switch shell to bash
rm /bin/sh; ln -s /bin/bash /bin/sh

# Common symlinks
ln -s /net/databases /databases
ln -s /net/software /software
ln -s /home /mnt/home
ln -s /projects /mnt/projects
ln -s /net /mnt/net

apt-get update
# required X libs
apt-get install -y libx11-6 libxau6 libxext6 libxrender1
#  git
apt-get install -y git
apt-get clean

# Install conda
bash /opt/miniconda.sh -b -u -p /usr

# Install conda/pip packages
# conda update conda

# open-babel
conda install \
   -c conda-forge \
   openbabel


# pytorch + dependancies
conda install \
   -c nvidia \
   -c pytorch \
   -c pyg \
   -c dglteam/label/cu117 \
   -c https://conda.rosettacommons.org \
   pip \
   ipython \
   ipykernel \
   numpy \
   pandas \
   seaborn \
   matplotlib \
   jupyterlab \
   pytorch \
   pytorch-cuda=11.7 \
   dgl \
   pyg \
   pyrosetta

# pip extras
pip install e3nn \
   omegaconf \
   hydra-core \
   pyrsistent \
   opt_einsum \
   sympy \
   omegaconf \
   icecream \
   wandb \
   deepdiff \
   assertpy

# SE3 transformer
pip install /SE3Transformer/

# Clean up
conda clean -a -y
apt-get -y purge build-essential git wget
apt-get -y autoremove
apt-get clean
rm /opt/miniconda.sh

%environment
export PATH=$PATH:/usr/local/cuda/bin

%runscript
exec python "$@"

%help
SE3nv environment for running RF-diffusion, etc.
