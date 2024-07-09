Bootstrap: docker
From: ubuntu
IncludeCmd: yes

%setup
#copy in our modified alphafold and colabfold
mkdir $APPTAINER_ROOTFS/apptainer_python_packages
mkdir $APPTAINER_ROOTFS/apptainer_python_packages/colabfold_bak
mkdir $APPTAINER_ROOTFS/apptainer_python_packages/alphafold_bak
cp -r /home/drhicks1/ColabFold_dev/colabfold/* $APPTAINER_ROOTFS/apptainer_python_packages/colabfold_bak/
cp -r /home/drhicks1/ColabFold_dev/alphafold/* $APPTAINER_ROOTFS/apptainer_python_packages/alphafold_bak/
cp /home/drhicks1/ColabFold_dev/AlphaFold2_initial_guess.py $APPTAINER_ROOTFS/apptainer_python_packages/
cp -r /home/drhicks1/silent_tools/ $APPTAINER_ROOTFS/apptainer_python_packages/

rsync -a --no-o --no-g /usr/local/cuda-11.7/lib/ $APPTAINER_ROOTFS/usr/lib/x86_64-linux-gnu/
rsync -a --no-o --no-g /usr/local/cuda-11.7/lib64/ $APPTAINER_ROOTFS/usr/lib/x86_64-linux-gnu/
rsync -a --no-o --no-g /usr/local/cuda-11.7/bin/ $APPTAINER_ROOTFS/usr/bin/

#only Brians hacked/old pyrosetta seems compatible
cp -r /home/drhicks1/dev_SE3nv_22_12_11/ $APPTAINER_ROOTFS/apptainer_python_packages/

cp /usr/local/cuda/bin/ptxas $APPTAINER_ROOTFS/apptainer_python_packages/

%files
/etc/localtime
/etc/apt/sources.list
/archive/software/Miniconda3-latest-Linux-x86_64.sh /opt/miniconda.sh

%post
# Switch shell to bash
rm /bin/sh; ln -s /bin/bash /bin/sh

# Common symlinks for IPD
ln -s /net/databases /databases
ln -s /net/software /software
ln -s /home /mnt/home
ln -s /projects /mnt/projects
ln -s /net /mnt/net

# apt
apt-get update
# apt-get install -y git
# apt-get clean

bash /opt/miniconda.sh -b -u -p /usr
rm /opt/miniconda.sh                 


conda install python=3.9 pip git

#install all required packages from colabfold
pip install 'colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold'

#This was needed, but pip complained about package discrepancies
pip install --upgrade dm-haiku

#install some other packages we need
conda install -y -c conda-forge -c bioconda kalign2=2.04 hhsuite=3.3.0 mmseqs2=14.7 openmm=7.7.0 pdbfixer

pip uninstall -y jaxlib
pip uninstall -y jax
#install correct jaxlib
pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.20+cuda11.cudnn86-cp39-cp39-manylinux2014_x86_64.whl
#install/update biopython
pip install biopython==1.81
#install correct jax
#pip install --upgrade jax
pip install jax[cuda11_pip]==0.4.20

#only Brians hacked/old pyrosetta seems compatible
pip install $APPTAINER_ROOTFS/apptainer_python_packages/dev_SE3nv_22_12_11/setup/
rm -r $APPTAINER_ROOTFS/apptainer_python_packages/dev_SE3nv_22_12_11/

ls 
ls $APPTAINER_ROOTFS
ls $APPTAINER_ROOTFS/apptainer_python_packages/

site_packages=$(python -c 'import site; print(site.getsitepackages()[0])')
#sed -i 's/fastmath=1)(func)/fastmath=True)(func.py_func)/g' $site_packages/homog/util.py

ls $site_packages

rm -r $site_packages/alphafold
rm -r $site_packages/colabfold

mv $APPTAINER_ROOTFS/apptainer_python_packages/alphafold_bak $site_packages/alphafold
mv $APPTAINER_ROOTFS/apptainer_python_packages/colabfold_bak $site_packages/colabfold
mv $APPTAINER_ROOTFS/apptainer_python_packages/silent_tools $site_packages/

mv $APPTAINER_ROOTFS/apptainer_python_packages/ptxas $APPTAINER_ROOTFS/usr/bin/ptxas

# Clean up
apt-get clean
pip cache purge

%runscript
exec python "$@"

%help
ColabFold_initial_guess env.
Author: Derrick Hicks
