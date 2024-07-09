import sys, os, argparse, subprocess, re, shutil, datetime, string, random
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('list', help='List of pairs of PDBs')
args = p.parse_args()
args.out = args.list+'.out'

with open(args.list) as f:
    pairs = [l.strip().split() for l in f.readlines()]

# copy files into memory, to avoid excessive disk reading
filenames = np.unique([fn for fnlist in pairs for fn in fnlist])

datestr = str(datetime.datetime.now()).replace(':','').replace(' ','_') # YYYY-MM-DD_HHMMSS.xxxxxx
randstr = ''.join(random.choice(string.ascii_lowercase) for i in range(10)) # 10-letter random string
tmp_dir = '/dev/shm/tmalign_'+datestr+'_'+randstr+'/'

os.makedirs(tmp_dir)
for fn in filenames:
    shutil.copy(fn, tmp_dir+os.path.basename(fn))

fn_map = dict(zip(filenames, [tmp_dir+os.path.basename(fn) for fn in filenames]))

# perform all pairwise TM aligns
try: 
    with open(args.out,'w') as outfile:
        for fn1,fn2 in pairs:
            fn1 = fn_map[fn1]
            fn2 = fn_map[fn2]

            cmd = f'/home/aivan/prog/TMalign {fn1} {fn2} -a'
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out,err = proc.communicate()

            m = re.search('TM\-score\= ((0|1)\.\d+).*average',out.decode('ascii'))
            score = float(m.groups()[0])

            output = '%s %s %f' % (os.path.basename(fn1).replace('.pdb',''), \
                                os.path.basename(fn2).replace('.pdb',''), \
                                score)
            print(output)
            print(output, file=outfile)
    shutil.rmtree(tmp_dir)

except Exception as e:
    # upon any error above, remove copied files from /dev/shm/
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    raise e

