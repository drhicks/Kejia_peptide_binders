import pandas as pd
import sys, os
import re
import argparse
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import molecular_weight
import subprocess
import numpy as np

try:
    from silent_tools import silent_tools
except ImportError:
    print("silent_tools not in path; adding directory from drhicks1")
    sys.path.append("/home/drhicks1/")
    try:
        from silent_tools import silent_tools
    except ImportError:
        print("Failed to import silent_tools even after modifying sys.path")
        sys.exit(1)

def run_mmseqs2_easy_cluster(fasta_file, output_prefix, min_id=0.5, max_id=0.99, goal_min=40, goal_max=70, max_attempts=100):
    rep_file = f"{output_prefix}_rep_seq.fasta"
    clusters = 0
    attempts = 0
    cluster_value = (min_id+max_id)/2

    while (clusters < goal_min or clusters > goal_max) and attempts < max_attempts and min_id < cluster_value < max_id:
        subprocess.run(['/software/mmseqs2/bin/mmseqs', 'easy-cluster', fasta_file, output_prefix, 'tmp', '--min-seq-id', str(cluster_value), '-s', '5.7'])
        
        result = subprocess.run(['wc', '-l', rep_file], stdout=subprocess.PIPE, text=True)
        line_count = int(result.stdout.split()[0])
        clusters = line_count / 2

        if clusters < goal_min:
            cluster_value += 0.01
        else:
            cluster_value -= 0.003
    return f"{output_prefix}_cluster.tsv"

def map_pdb_to_cluster(mmseqs2_outfile, df):
    cluster_to_pdb = {}
    with open(mmseqs2_outfile, 'r') as f:
        for line in f:
            cluster_id, pdb_id = line.strip().split('\t')
            if cluster_id in cluster_to_pdb:
                cluster_to_pdb[cluster_id].append(pdb_id)
            else:
                cluster_to_pdb[cluster_id] = [pdb_id]

    cluster_id_to_pdb = {}
    cluster_index = 0
    for cluster_id, pdb_ids in cluster_to_pdb.items():
        cluster_id_to_pdb[cluster_index] = pdb_ids
        cluster_index += 1

    cluster_id_column = []
    for index, row in df.iterrows():
        pdb_file_name_substr = row['description']
        for cluster_id, pdb_ids in cluster_id_to_pdb.items():
            for pdb_id in pdb_ids:
                if pdb_file_name_substr in pdb_id:
                    cluster_id_column.append(cluster_id)
                    break
            else:
                continue
            break
        else:
            cluster_id_column.append(None)

    df['cluster_id'] = cluster_id_column
    return df

def add_sequence_column(df, sequences):
    df['sequence'] = sequences
    return df

def output_fasta(df, output_file):
    with open(output_file, 'w') as f:
        for index, row in df.iterrows():
            f.write(f">{row['description']}\n")
            sequence = row['sequence']
            for i in range(0, len(sequence), 80):
                f.write(sequence[i:i+80] + "\n")

def load_data(filepath):
    return pd.read_csv(filepath, delim_whitespace=True)

def dynamic_filtering(group, thresholds, max_thresholds, N_examples=1, max_stagnant_cycles=1000, max_cycles=10000, not_initial_guess=False):
    if not_initial_guess:
        filtered = group[
            (group['iptm'] >= thresholds['iptm']) &
            (group['plddt_binder'] >= thresholds['plddt_binder'])
        ]
        
    else:
        filtered = group[
            (group['iptm'] >= thresholds['iptm']) &
            (group['interface_rmsd'] <= thresholds['interface_rmsd']) &
            (group['plddt_binder'] >= thresholds['plddt_binder'])
        ]

    previous_len = len(filtered)
    stagnant_count = 0
    cycles = 0
    while len(filtered) != N_examples and stagnant_count < max_stagnant_cycles and cycles < max_cycles:
        if not_initial_guess:
            if len(filtered) > N_examples:
                new_thresholds = {
                    'iptm': max(thresholds['iptm'] * 1.0003, max_thresholds['iptm']),
                    'plddt_binder': max(thresholds['plddt_binder'] * 1.0002, max_thresholds['plddt_binder'])
                }
            else:
                new_thresholds = {
                    'iptm': max(thresholds['iptm'] * 0.9998, max_thresholds['iptm']),
                    'plddt_binder': max(thresholds['plddt_binder'] * 0.9997, max_thresholds['plddt_binder'])
                }
            
            filtered = group[
                (group['iptm'] >= new_thresholds['iptm']) &
                (group['plddt_binder'] >= new_thresholds['plddt_binder'])
            ]

        else:
            if len(filtered) > N_examples:
                new_thresholds = {
                    'iptm': max(thresholds['iptm'] * 1.0003, max_thresholds['iptm']),
                    'interface_rmsd': min(thresholds['interface_rmsd'] * 0.9998, max_thresholds['interface_rmsd']),
                    'plddt_binder': max(thresholds['plddt_binder'] * 1.0002, max_thresholds['plddt_binder'])
                }
            else:
                new_thresholds = {
                    'iptm': max(thresholds['iptm'] * 0.9998, max_thresholds['iptm']),
                    'interface_rmsd': min(thresholds['interface_rmsd'] * 1.0003, max_thresholds['interface_rmsd']),
                    'plddt_binder': max(thresholds['plddt_binder'] * 0.9997, max_thresholds['plddt_binder'])
                }
            
            filtered = group[
                (group['iptm'] >= new_thresholds['iptm']) &
                (group['interface_rmsd'] <= new_thresholds['interface_rmsd']) &
                (group['plddt_binder'] >= new_thresholds['plddt_binder'])
            ]

        if len(filtered) == previous_len:
            stagnant_count += 1
            thresholds = new_thresholds
        else:
            previous_len = len(filtered)
            thresholds = new_thresholds
            stagnant_count = 0

        cycles += 1

    if len(filtered) > N_examples:
        filtered = filtered.sample(n=N_examples, random_state=42)

    if len(filtered) == 0:
        if not_initial_guess:
            filtered = group[
                (group['iptm'] >= max_thresholds['iptm']) &
                (group['plddt_binder'] >= max_thresholds['plddt_binder'])
            ]
        
        else:
            filtered = group[
                (group['iptm'] >= max_thresholds['iptm']) &
                (group['interface_rmsd'] <= max_thresholds['interface_rmsd']) &
                (group['plddt_binder'] >= max_thresholds['plddt_binder'])
            ]

        if len(filtered) > N_examples:
            filtered = filtered.sample(n=N_examples, random_state=42)
        
        thresholds = max_thresholds

    return filtered, thresholds

def print_full(x):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')

class TrieNode:
    def __init__(self):
        self.children = {}
        self.strings = []

def insert(root, string):
    current = root
    for char in string:
        if char not in current.children:
            current.children[char] = TrieNode()
        current = current.children[char]
    current.strings.append(string)

def map_substrings_to_strings(substrings, strings):
    # Sort the substrings and strings
    substrings.sort()
    strings.sort()

    root = TrieNode()
    for string in strings:
        insert(root, string)

    substring_map = {}
    for substr in substrings:
        current = root
        for char in substr:
            if char in current.children:
                current = current.children[char]
            else:
                current = None
                break
        if current:
            # Collect up to 5 strings and remove them
            matches = []
            for s in current.strings[:5]:
                matches.append(s)
            substring_map[substr] = matches
            # Remove matched strings from trie
            for match in matches:
                remove(root, match)

    return substring_map

def remove(root, string):
    def remove_rec(node, string, depth):
        if depth == len(string):
            if node.strings:
                node.strings.remove(string)
            return len(node.strings) == 0 and not node.children

        char = string[depth]
        if char in node.children:
            can_delete = remove_rec(node.children[char], string, depth + 1)
            if can_delete:
                del node.children[char]
                return len(node.strings) == 0 and not node.children

        return False

    remove_rec(root, string, 0)

def binder_seqs_from_silent(silentfile, tags):
    silent_index = silent_tools.get_silent_index(silentfile)
    full_tags = silent_index['tags']

    substring_to_strings = map_substrings_to_strings(tags, full_tags)

    sequences = []
    with open(silentfile, errors='ignore') as sf:
        for subtag in tags:
            tag = substring_to_strings[subtag][0]

            structure = silent_tools.get_silent_structure_file_open(sf, silent_index, tag)

            sequence_chunks = silent_tools.get_sequence_chunks(structure)
            sequences.append(sequence_chunks[0])

    return sequences

def add_protein_info(df, sequence_column):
    def calculate_protein_properties(seq):
        analysis = ProteinAnalysis(seq)
        molar_ext_coeff = analysis.molar_extinction_coefficient()[1]
        mw = molecular_weight(seq, seq_type="protein", monoisotopic=False)
        abs_01_percent = molar_ext_coeff / mw
        isoelectric_point = analysis.isoelectric_point()
        mw_average = mw
        return abs_01_percent, isoelectric_point, mw_average

    df['Abs_0.1%'], df['isoelectric_point'], df['molecular_weight_average'] = zip(*df[sequence_column].apply(calculate_protein_properties))
    
    return df

def filter_duplicate_seqs(df, sequence_column):
    df_sorted = df.sort_values(by=[sequence_column, 'iptm'], ascending=[True, False])
    df_filtered = df_sorted.drop_duplicates(subset=[sequence_column], keep='first')
    return df_filtered

def main():
    parser = argparse.ArgumentParser(description="Process protein sequences and filter based on properties.")
    parser.add_argument("scorefile", type=str, help="Path to the score file.")
    parser.add_argument("silentfile", type=str, help="Path to the silent file.")
    parser.add_argument("--output_file", type=str, default="for_filtering.fasta", help="Output file name for the FASTA file.")
    parser.add_argument("--mmseqs_prefix", type=str, default="mmseqs_out", help="Prefix for MMseqs2 output files.")
    parser.add_argument("--min_id", type=float, default=0.5, help="Minimum sequence identity for clustering.")
    parser.add_argument("--max_id", type=float, default=0.99, help="Maximum sequence identity for clustering.")
    parser.add_argument("--goal_min", type=int, default=40, help="Minimum number of clusters to aim for.")
    parser.add_argument("--goal_max", type=int, default=70, help="Maximum number of clusters to aim for.")
    parser.add_argument("--max_attempts", type=int, default=100, help="Maximum number of attempts for clustering.")
    parser.add_argument("--initial_iptm", type=float, default=0.89, help="Initial minimum threshold for iptm.")
    parser.add_argument("--initial_interface_rmsd", type=float, default=1.0, help="Initial maximum threshold for interface_rmsd.")
    parser.add_argument("--initial_plddt_binder", type=float, default=92.0, help="Initial minimum threshold for plddt_binder.")
    parser.add_argument("--max_iptm", type=float, default=0.85, help="Maximum allowable threshold for iptm.")
    parser.add_argument("--max_interface_rmsd", type=float, default=1.5, help="Maximum allowable threshold for interface_rmsd.")
    parser.add_argument("--max_plddt_binder", type=float, default=88.0, help="Minimum allowable threshold for plddt_binder.")
    parser.add_argument("--n_examples", type=int, default=1, help="Desired number of output per cluster")
    parser.add_argument("--abs_cut", type=float, default=0, help="minimum acceptable Abs_0.1 value so you can use A280")
    parser.add_argument("--isoelectric_point_cut", type=float, default=5.5, help="maximum acceptable isoelectric_point")
    parser.add_argument('--not_initial_guess', action='store_true', help='Set this flag for af2 predictions not using initial guess')

    args = parser.parse_args()

    if args.not_initial_guess:
        initial_thresholds = {
            'iptm': args.initial_iptm,
            'plddt_binder': args.initial_plddt_binder
        }

        max_thresholds = {
            'iptm': args.max_iptm,
            'plddt_binder': args.max_plddt_binder
        }

    else:
         initial_thresholds = {
            'iptm': args.initial_iptm,
            'interface_rmsd': args.initial_interface_rmsd,
            'plddt_binder': args.initial_plddt_binder
        }

        max_thresholds = {
            'iptm': args.max_iptm,
            'interface_rmsd': args.max_interface_rmsd,
            'plddt_binder': args.max_plddt_binder
        }

    df = load_data(args.scorefile)
    print(f"df len: {len(df)}")
    
    if args.not_initial_guess:
        df = df[(df["iptm"] >= max_thresholds["iptm"]) & 
                (df["plddt_binder"] >= max_thresholds["plddt_binder"])]
    else:
        df = df[(df["iptm"] >= max_thresholds["iptm"]) & 
            (df["interface_rmsd"] <= max_thresholds["interface_rmsd"]) & 
            (df["plddt_binder"] >= max_thresholds["plddt_binder"])]

    print(f"df len: {len(df)}", flush=True)

    tags = df["description"].values
    sequences = binder_seqs_from_silent(args.silentfile, tags)
    print(f"number of sequences: {len(sequences)}")

    df = add_sequence_column(df, sequences)
    print("added sequences to df")
    df = filter_duplicate_seqs(df, 'sequence')
    print("filtered duplicates sequences")
    df = add_protein_info(df, 'sequence')
    print("added protein info")

    df = df[(df["Abs_0.1%"] > args.abs_cut) & 
            (df["isoelectric_point"] <= args.isoelectric_point_cut)]
    print(f"df len: {len(df)}", flush=True)

    output_fasta(df, args.output_file)

    clusters_file = run_mmseqs2_easy_cluster(args.output_file, args.mmseqs_prefix, args.min_id, args.max_id, args.goal_min, args.goal_max, args.max_attempts)

    df = map_pdb_to_cluster(clusters_file, df)

    grouped = df.groupby("cluster_id")
    df['cluster_size'] = grouped['cluster_id'].transform('size')
    final_results = {}

    for key, group in grouped:
        filtered_group, thresholds = dynamic_filtering(group, initial_thresholds, max_thresholds, N_examples=args.n_examples, not_initial_guess=args.not_initial_guess)
        final_results[key] = filtered_group
        print(f"Cluster ID: {key}, Final Length: {len(filtered_group)}, Final Thresholds: {thresholds}")

    for key, group in final_results.items():
        print_full(group)

if __name__ == "__main__":
    main()
