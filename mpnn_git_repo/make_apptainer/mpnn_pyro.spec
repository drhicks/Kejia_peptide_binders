Bootstrap: docker
From: ubuntu
IncludeCmd: yes

%setup

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

  /usr/bin/conda install -c conda-forge mamba -y
  mamba install python=3.11 pip tensorflow pytorch numpy scipy numba biopython ipython pandas matplotlib ipykernel seaborn pyrosetta -c pytorch -c conda-forge --channel https://conda.graylab.jhu.edu -y

  # # PyRosetta
  # pip install pyrosetta -f https://west.rosettacommons.org/pyrosetta/release/release/PyRosetta4.Release.python310.linux.wheel/


  # Clean up
  apt-get clean
  mamba clean -y --all
  conda clean -y --all
  pip cache purge

%runscript
  exec python "$@"

%help
  Tensorflow environment for running mpnn with pyrosetta