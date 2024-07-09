import sys, os, argparse, subprocess, re

p = argparse.ArgumentParser()
p.add_argument('list', help='List of pairs of fastas')
args = p.parse_args()
args.out = args.list+'.out'

with open(args.list) as f:
    pairs = [l.strip().split() for l in f.readlines()]

with open(args.out,'w') as outfile:
    for fn1,fn2 in pairs:
        cmd = f'blastp -query {fn1} -subject {fn2} -max_hsps 1 -subject_besthit -outfmt "10 std nident qlen slen"'
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out,err = proc.communicate()
        output = out.decode().strip()
        print(output)
        print(output, file=outfile)

