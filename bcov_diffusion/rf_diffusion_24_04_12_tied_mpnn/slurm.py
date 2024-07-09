import os
import sys

# TODO: make this path relative
sys.path.append('/home/ahern/projects/BFF/rf_diffusion/benchmark/util')
import slurm_tools

def array_submit(top_dir, script, arg_sets, submit=False, gres='', env='woody-SE3-nvidia'):
    jobslist_path = os.path.join(top_dir, 'jobs.list')
    #log_dir = os.path.join'../logs')
    output_dir = os.path.join(top_dir, 'output')
    if submit:
        os.makedirs(output_dir)
        #os.path.makedirs(log_dir)

    fh = open(jobslist_path, 'w') if submit else sys.stdout
    for job_i, args in enumerate(arg_sets):
        cmd = " ".join([script] + args)
        job_dir = os.path.join(top_dir, 'jobs', str(job_i))
        if submit:
            os.makedirs(job_dir)
        log_path = 'log.txt'
        print(f'cd {job_dir}; source activate {env}; python {cmd} >> {log_path}', file=fh)

    if submit:
        fh.close()
        job_name = os.path.basename(top_dir)
        slurm_job, proc = slurm_tools.array_submit(jobslist_path, gres=gres, log=True, J=job_name)
        print(f'Submitted array job {slurm_job} with {len(arg_sets)} jobs')


        
