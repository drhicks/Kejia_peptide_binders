Bootstrap: docker
From: ubuntu
IncludeCmd: yes

%setup
# Copy in our modified alphafold and colabfold
mkdir -p $APPTAINER_ROOTFS/apptainer_python_packages/colabfold_bak
mkdir -p $APPTAINER_ROOTFS/apptainer_python_packages/alphafold_bak
cp -r ../colabfold/* $APPTAINER_ROOTFS/apptainer_python_packages/colabfold_bak/
cp -r ../alphafold/* $APPTAINER_ROOTFS/apptainer_python_packages/alphafold_bak/
cp -r ../../silent_tools/ $APPTAINER_ROOTFS/apptainer_python_packages/

rsync -a --no-o --no-g /usr/local/cuda-11.7/lib/ $APPTAINER_ROOTFS/usr/lib/x86_64-linux-gnu/
rsync -a --no-o --no-g /usr/local/cuda-11.7/lib64/ $APPTAINER_ROOTFS/usr/lib/x86_64-linux-gnu/
rsync -a --no-o --no-g /usr/local/cuda-11.7/bin/ $APPTAINER_ROOTFS/usr/bin/
cp /usr/local/cuda/bin/ptxas $APPTAINER_ROOTFS/apptainer_python_packages/

#only this old pyrosetta seems compatible
cp -r /home/drhicks1/dev_SE3nv_22_12_11/ $APPTAINER_ROOTFS/apptainer_python_packages/

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

bash /opt/miniconda.sh -b -u -p /usr
rm /opt/miniconda.sh

conda install -y -n base python=3.9 pip git
conda install -y -n base libarchive conda-libmamba-solver libmamba libmambapy -c main --force-reinstall
#conda install -y --solver=classic conda-forge::conda-libmamba-solver conda-forge::libmamba conda-forge::libmambapy conda-forge::libarchive

#    pyrosetta=2023.49+release.9891f2c=py39_0 \
# Install packages with conda where possible
conda install -y -n base -c conda-forge -c bioconda -c https://west.rosettacommons.org/pyrosetta/conda/release/ \
    kalign2=2.04 \
    hhsuite=3.3.0 \
    mmseqs2=14.7 \
    openmm=7.7.0 \
    pdbfixer \
    appdirs \
    absl-py \
    chex \
    contextlib2 \
    contourpy \
    cryptography \
    cycler \
    dm-haiku \
    dm-tree \
    etils \
    flatbuffers \
    flax \
    fonttools \
    fsspec \
    gast \
    google-pasta \
    grpcio \
    h5py \
    immutabledict \
    importlib-metadata \
    importlib_resources \
    keras \
    kiwisolver \
    libclang \
    markdown \
    markdown-it-py \
    markupsafe \
    matplotlib \
    mdurl \
    ml-collections \
    namex \
    nest-asyncio \
    numpy=2.0.0 \
    opt-einsum \
    optree \
    packaging \
    pandas \
    pillow \
    pluggy \
    protobuf \
    py3Dmol \
    python-dateutil \
    pytz \
    pyyaml \
    requests \
    rich \
    ruamel.yaml \
    scipy \
    six \
    tabulate \
    tensorboard \
    termcolor \
    toolz \
    tqdm \
    typing_extensions \
    urllib3 \
    werkzeug \
    wrapt \
    zipp \
    libstdcxx-ng=14.1.0 \
    libgcc-ng=14.1.0

# Fallback to pip for remaining packages
echo "Installing remaining packages with pip"
pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.20+cuda11.cudnn86-cp39-cp39-manylinux2014_x86_64.whl
pip install jax[cuda11_pip]==0.4.20
pip install ml-dtypes
pip install msgpack
pip install optax
pip install orbax-checkpoint
pip install tensorboard-data-server
pip install tensorflow-io-gcs-filesystem
pip install tensorstore
pip install biopython==1.81

#only this old pyrosetta seems compatible
pip install $APPTAINER_ROOTFS/apptainer_python_packages/dev_SE3nv_22_12_11/setup/

site_packages=$(python -c 'import site; print(site.getsitepackages()[0])')

mv $APPTAINER_ROOTFS/apptainer_python_packages/alphafold_bak $site_packages/alphafold
mv $APPTAINER_ROOTFS/apptainer_python_packages/colabfold_bak $site_packages/colabfold
mv $APPTAINER_ROOTFS/apptainer_python_packages/silent_tools $site_packages/

mv $APPTAINER_ROOTFS/apptainer_python_packages/ptxas $APPTAINER_ROOTFS/usr/bin/ptxas

ls $site_packages

conda list 

# Clean up
apt-get clean
pip cache purge

%runscript
exec python "$@"

%help
ColabFold_initial_guess env.
Author: Derrick Hicks

