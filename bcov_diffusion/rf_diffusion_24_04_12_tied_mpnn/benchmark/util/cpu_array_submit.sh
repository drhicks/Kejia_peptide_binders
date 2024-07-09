#!/bin/bash
if [ -z "$2" ]; then
    jobname=$1
else
    jobname=$2
fi

sbatch -a 1-$(cat $1 | wc -l) -p short -J $jobname \
       -c 2 --mem=12g \
       --wrap="eval \`sed -n \${SLURM_ARRAY_TASK_ID}p $1\`" \
       -o /dev/null -e /dev/null # comment this out to see slurm logs
