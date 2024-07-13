import sys

#multi line fasta
#>header
#peptide_seq
#assumes no blank lines

python_env = "/home/drhicks1/.conda/envs/pyro/bin/python"
threading_script = "/home/drhicks1/scripts/Kejia_peptide_binders/threading/thread_peptide_sequence_new.py"

peptide_fasta = sys.argv[1]
template_list = sys.argv[2]

with open(template_list, "r") as f:
    templates = f.readlines()
    templates = [x.strip() for x in templates]

with open(peptide_fasta, "r") as f:
    lines = f.readlines()
    lines = [x.strip() for x in lines]

pairs = []
for i, j in zip(lines[0:-1:2], lines[1::2]):
    i = i.replace(">", "")
    pairs.append([i,j])

for template in scaffolds:
    for pair in pairs:
        print(f"{python_env} {threading_script} --peptide_header {pair[0]} --peptide_seq {pair[1]} --template {template}")
