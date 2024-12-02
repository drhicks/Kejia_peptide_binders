Bootstrap: docker
From: ubuntu

IncludeCmd: yes

%setup
    # Create directories for Python packages
    mkdir -p $APPTAINER_ROOTFS/apptainer_python_packages/
    cp -r /home/$USER/silent_tools/ $APPTAINER_ROOTFS/apptainer_python_packages/

    # Rsync CUDA 12.0 files (handle existing directories)
    rsync -a --ignore-existing --no-o --no-g /usr/local/cuda-12.0/lib64/ $APPTAINER_ROOTFS/usr/lib/x86_64-linux-gnu/
    rsync -a --ignore-existing --no-o --no-g /usr/local/cuda-12.0/bin/ $APPTAINER_ROOTFS/usr/bin/

    cp /usr/local/cuda/bin/ptxas $APPTAINER_ROOTFS/apptainer_python_packages/

%files
    /etc/localtime
    /etc/apt/sources.list
    /archive/software/Miniconda3-latest-Linux-x86_64.sh /opt/miniconda.sh

%post
    # Update the package list and install basic dependencies
    apt-get update && apt-get install -y wget git build-essential

    # Install Miniconda
    bash /opt/miniconda.sh -b -u -p /usr
    rm /opt/miniconda.sh

    # Install Mamba (via conda-forge channel)
    /usr/bin/conda install -n base -c conda-forge mamba -y

    # Create and activate the Mamba environment
    echo -e "Creating Mamba environment with GPU-enabled JAX and Pyrosetta for CUDA 12.0\n"
    CONDA_OVERRIDE_CUDA='12.0' /usr/bin/mamba create --name colabfold_ig python=3.11 \
        pip pandas matplotlib numpy'<2.0.0' biopython=1.81 scipy pdbfixer seaborn \
        kalign2=2.04 hhsuite=3.3.0 mmseqs2=14.7 openmm=8.0.0 \
        libgfortran5 tqdm jupyter ffmpeg pyrosetta fsspec py3dmol chex \
        dm-haiku flax'<0.10.0' dm-tree joblib ml-collections immutabledict optax \
        jaxlib=0.4.23=cuda120py311h5a49573_202 jax[cuda] cuda-nvcc cudnn appdirs tensorflow \
        -c conda-forge -c nvidia -c bioconda --channel https://conda.graylab.jhu.edu -y

    # Clean up
    apt-get clean
    mamba clean -y --all
    pip cache purge

%environment
    # Add CUDA 12.0 and Mamba environment to the PATH
    export PATH=/usr/local/cuda-12.0/bin:/usr/envs/colabfold_ig/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH
    export CUDA_HOME=/usr/local/cuda-12.0
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
    export TF_FORCE_GPU_ALLOW_GROWTH=true

%runscript
    python "$@"

%help
    Environment for running colabfold_ig with CUDA 12.0
