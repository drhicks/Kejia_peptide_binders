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

def run_mmseqs2_easy_cluster(fasta_file, output_prefix, min_id=0.5, max_id=0.99, desired_goal=100, threshold=10, max_attempts=20):
    rep_file = f"{output_prefix}_rep_seq.fasta"
    clusters = 0
    attempts = 0
    best_cluster_value = min_id
    best_diff = float('inf')
    best_clusters = clusters

    while attempts < max_attempts and min_id <= max_id:
        cluster_value = (min_id + max_id) / 2
        subprocess.run(['/software/mmseqs2/bin/mmseqs', 'easy-cluster', fasta_file, output_prefix, 'tmp', '--min-seq-id', str(cluster_value), '-s', '5.7'])
        
        result = subprocess.run(['wc', '-l', rep_file], stdout=subprocess.PIPE, text=True)
        line_count = int(result.stdout.split()[0])
        clusters = line_count / 2

        diff = abs(clusters - desired_goal)
        if diff < best_diff:
            best_diff = diff
            best_cluster_value = cluster_value
            best_clusters = clusters

        if diff <= threshold:
            break  # Close enough to the desired goal
        elif clusters < desired_goal:
            min_id = cluster_value + 0.0001  # Small adjustment to avoid infinite loop
        else:
            max_id = cluster_value - 0.0001  # Small adjustment to avoid infinite loop

        attempts += 1

    # In case the loop exits without reaching the threshold
    if best_diff > threshold:
        print(f"Could not reach desired goal. Best result was {best_clusters} clusters with cluster_value {best_cluster_value}")

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

def dynamic_filtering(group, thresholds, worst_thresholds, N_examples=1, max_stagnant_cycles=1000, max_cycles=10000, not_initial_guess=False):
    def apply_filters(group, thresholds, not_initial_guess):
        if not_initial_guess:
            return group[
                (group['iptm'] >= thresholds['iptm']) &
                (group['plddt_binder'] >= thresholds['plddt_binder'])
            ]
        else:
            return group[
                (group['iptm'] >= thresholds['iptm']) &
                (group['interface_rmsd'] <= thresholds['interface_rmsd']) &
                (group['plddt_binder'] >= thresholds['plddt_binder'])
            ]

    filtered = apply_filters(group, thresholds, not_initial_guess)
    previous_len = len(filtered)
    stagnant_count = 0
    cycles = 0

    while len(filtered) != N_examples and stagnant_count < max_stagnant_cycles and cycles < max_cycles:
        adjust_factor_up_iptm = 1.0003
        adjust_factor_down_iptm = 0.9998
        adjust_factor_up_other = 1.0002
        adjust_factor_down_other = 0.9997

        new_thresholds = {
            'iptm': max(thresholds['iptm'] * (adjust_factor_down_iptm if len(filtered) < N_examples else adjust_factor_up_iptm), worst_thresholds['iptm']),
            'plddt_binder': max(thresholds['plddt_binder'] * (adjust_factor_down_other if len(filtered) < N_examples else adjust_factor_up_other), worst_thresholds['plddt_binder'])
        }

        if not not_initial_guess:
            new_thresholds['interface_rmsd'] = min(thresholds['interface_rmsd'] * (adjust_factor_down_other if len(filtered) > N_examples else adjust_factor_up_other), worst_thresholds['interface_rmsd'])

        filtered = apply_filters(group, new_thresholds, not_initial_guess)

        if len(filtered) == previous_len:
            stagnant_count += 1
        else:
            previous_len = len(filtered)
            stagnant_count = 0

        thresholds = new_thresholds
        cycles += 1

    if len(filtered) > N_examples:
        filtered = filtered.sample(n=N_examples, random_state=42)

    if len(filtered) == 0:
        filtered = apply_filters(group, worst_thresholds, not_initial_guess)
        if len(filtered) > N_examples:
            filtered = filtered.sample(n=N_examples, random_state=42)
        thresholds = worst_thresholds

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
    parser.add_argument("--min_id", type=float, default=0.4, help="Minimum sequence identity for clustering.")
    parser.add_argument("--max_id", type=float, default=0.9, help="Maximum sequence identity for clustering.")
    parser.add_argument("--desired_goal", type=int, default=105, help="Desired number of clusters to aim for.")
    parser.add_argument("--goal_threshold", type=int, default=5, help="Devation from desired number of clusters that is ok.")
    parser.add_argument("--max_attempts", type=int, default=20, help="Maximum number of attempts for clustering.")
    parser.add_argument("--initial_iptm", type=float, default=0.88, help="Initial minimum threshold for iptm.")
    parser.add_argument("--initial_interface_rmsd", type=float, default=1.5, help="Initial maximum threshold for interface_rmsd.")
    parser.add_argument("--initial_plddt_binder", type=float, default=90.0, help="Initial minimum threshold for plddt_binder.")
    parser.add_argument("--worst_iptm", type=float, default=0.83, help="Maximum allowable threshold for iptm.")
    parser.add_argument("--worst_interface_rmsd", type=float, default=2.0, help="Maximum allowable threshold for interface_rmsd.")
    parser.add_argument("--worst_plddt_binder", type=float, default=85.0, help="Minimum allowable threshold for plddt_binder.")
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

        worst_thresholds = {
            'iptm': args.worst_iptm,
            'plddt_binder': args.worst_plddt_binder
        }

    else:
        initial_thresholds = {
            'iptm': args.initial_iptm,
            'interface_rmsd': args.initial_interface_rmsd,
            'plddt_binder': args.initial_plddt_binder
        }

        worst_thresholds = {
            'iptm': args.worst_iptm,
            'interface_rmsd': args.worst_interface_rmsd,
            'plddt_binder': args.worst_plddt_binder
        }

    df = load_data(args.scorefile)
    print(f"df len: {len(df)}")
    
    if args.not_initial_guess:
        df = df[(df["iptm"] >= worst_thresholds["iptm"]) & 
                (df["plddt_binder"] >= worst_thresholds["plddt_binder"])]
    else:
        df = df[(df["iptm"] >= worst_thresholds["iptm"]) & 
            (df["interface_rmsd"] <= worst_thresholds["interface_rmsd"]) & 
            (df["plddt_binder"] >= worst_thresholds["plddt_binder"])]

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

    clusters_file = run_mmseqs2_easy_cluster(args.output_file, args.mmseqs_prefix, args.min_id, args.max_id, args.desired_goal, args.goal_threshold, args.max_attempts)

    df = map_pdb_to_cluster(clusters_file, df)

    grouped = df.groupby("cluster_id")
    df['cluster_size'] = grouped['cluster_id'].transform('size')
    final_results = {}

    for key, group in grouped:
        filtered_group, thresholds = dynamic_filtering(group, initial_thresholds, worst_thresholds, N_examples=args.n_examples, not_initial_guess=args.not_initial_guess)
        final_results[key] = filtered_group
        print(f"Cluster ID: {key}, Final Length: {len(filtered_group)}, Final Thresholds: {thresholds}")

    for key, group in final_results.items():
        print_full(group)

if __name__ == "__main__":
    main()
