sbatch -a 1-$(cat jobs | wc -l) ${1}
