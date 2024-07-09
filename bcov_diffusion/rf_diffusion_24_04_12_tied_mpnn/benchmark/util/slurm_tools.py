import subprocess, re, os

def slurm_submit(cmd, p='cpu', c=1, mem=2, gres=None, J=None, wait_for=[], hold_until_finished=False, log=False, **kwargs):
    '''
    wait_for = wait for these slurm jobids to exist okay
    hold_until_finished =  if True, don't return command line control until the slurm job is done
    '''
    job_name = J if J else os.environ["USER"]+'_auto_submit'
    log_file = f'%A_%a_{J}.log' if log else '/dev/null'
    cmd_sbatch = f'sbatch --wrap "{cmd}" -p {p} -c {c} --mem {mem}g '\
        f'-J {job_name} '\
        f'{f"--gres {gres}" if gres else ""} '\
        f'{"-W" if hold_until_finished else ""} '\
        f'{"--dependency afterok:" + ":".join(map(str, wait_for)) if wait_for else ""} '\
        f'-o {log_file} '
    cmd_sbatch += ' '.join([f'{"--"+k if len(k)>1 else "-"+k} {v}' for k,v in kwargs.items() if v is not None])

    proc = subprocess.run(cmd_sbatch, shell=True, stdout=subprocess.PIPE)
    slurm_job = re.findall(r'\d+', str(proc.stdout))[0]

    return slurm_job, proc

def array_submit(job_list_file, p='gpu', gres='gpu:rtx2080:1', wait_for=None, log=False, **kwargs):
    return slurm_submit(
        cmd = 'eval \\`sed -n \\${SLURM_ARRAY_TASK_ID}p '+job_list_file+'\\`',
        a = f'1-$(cat {job_list_file} | wc -l)',
        p = p,
        c = 2,
        mem = 12,
        gres = gres,
        wait_for = wait_for,
        log = log,
        **kwargs
    )
