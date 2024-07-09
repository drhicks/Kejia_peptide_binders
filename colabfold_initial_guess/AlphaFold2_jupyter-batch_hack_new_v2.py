import argparse
import sys
import os
import glob
import re
from pathlib import Path

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from colabfold.download import download_alphafold_params, default_data_dir
from colabfold.utils import setup_logging
from colabfold.batch_hack import run, set_model_type

from colabfold.colabfold import plot_protein
from pathlib import Path
import matplotlib.pyplot as plt

def read_fasta(fasta_path):
    if not os.path.isfile(fasta_path):
        raise FileNotFoundError(f"The specified FASTA file was not found: {fasta_path}")
    with open(fasta_path, "r") as f:
        sequences = []
        sequence = ""
        header = None
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            if line.startswith(">"):
                if header:  # Save previous sequence
                    sequences.append((header, sequence))
                header = line  # This is a new header
                sequence = ""  # Reset sequence
            else:
                sequence += line  # Continue accumulating sequence lines
        if header:  # Save the last sequence
            sequences.append((header, sequence))
    return sequences

def pair_sequences(sequences, target_seq):
    # returns sorted pairs from smallest to largest sequence by overall length,
    # and alphabetically by the part after ":" within each length group
    pairs = []
    for header, seq in sequences:
        seq = seq if target_seq == '' else f"{seq}:{target_seq}"
        pairs.append((header, seq))
    # Sort by the length of the sequence first, then by the substring after ":"
    # Assumes that every sequence will have a ":" if target_seq is not empty
    pairs.sort(key=lambda x: (len(x[1]), x[1].split(":")[1] if ":" in x[1] else x[1]))
    return pairs

def create_query_list(pairs):
    queries = []
    for query_sequence in pairs:
        name = re.sub(r"\s+", "", query_sequence[0].strip().replace(">", ""))
        query_sequence = "".join(query_sequence[1].split())
        queries.append((name, query_sequence.upper().split(":"), None))
    return queries

def setup_argparse():
    parser = argparse.ArgumentParser(description='Run AlphaFold predictions with customizable parameters.')
    parser.add_argument('--fasta', type=str, required=True, help='Path to the FASTA file containing query sequences.')
    parser.add_argument('--result_dir', type=str, default='./', help='Directory where results will be saved.')
    parser.add_argument('--msa_mode', type=str, default='single_sequence', choices=['MMseqs2 (UniRef+Environmental)', 'MMseqs2 (UniRef only)', 'single_sequence', 'custom'], help='Mode for multiple sequence alignment.')
    parser.add_argument('--pair_mode', type=str, default='unpaired+paired', choices=['unpaired+paired', 'paired', 'unpaired'], help='Pairing mode for sequences.')
    parser.add_argument('--model_type', type=str, default='auto', choices=['AlphaFold2-ptm', 'AlphaFold2-multimer-v3', 'auto'], help='Type of AlphaFold model to use.')
    parser.add_argument('--num_recycles', type=int, default=3, help='Number of recycles to use in AlphaFold prediction.')
    parser.add_argument('--is_monomer', action='store_false', dest="is_complex", help='Set this flag if predicting a monomer.')
    parser.add_argument('--overwrite_results', action='store_false', dest="do_not_overwrite_results", help='Set this flag to overwrite existing results.')
    parser.add_argument('--dpi', type=int, default=200, help='DPI for image resolution.')
    parser.add_argument('--target_seq', type=str, default='', help='You can omit target from fasta, and append this target seq to all input binder seqs.')
    parser.add_argument('--template_mode', type=str, default='none', choices=['none', 'pdb70', 'custom'], help='Template mode for structure prediction; currently not implemented')
    parser.add_argument('--use_templates', action='store_true', help='Set this flag if you want to use templates; currently not implemented')
    parser.add_argument('--custom_template_path', type=str, default=None, help='path to custom templates; currently not implemented')
    return parser.parse_args()

def run_prediction(queries, result_dir, use_templates, custom_template_path, msa_mode, model_type, num_recycles, is_complex, do_not_overwrite_results, pair_mode, dpi):
    run(
        queries=queries,
        result_dir=result_dir,
        use_templates=use_templates,
        custom_template_path=custom_template_path,
        use_amber=False,
        msa_mode=msa_mode,    
        model_type=model_type,
        num_models=5,
        num_recycles=num_recycles,
        model_order=[1, 2, 3, 4, 5],
        is_complex=is_complex,
        data_dir=Path("/home/drhicks1/ColabFold/"),
        keep_existing_results=do_not_overwrite_results,
        recompile_padding=1.0,
        rank_by="auto",
        pair_mode=pair_mode,
        stop_at_score=float(100),
        prediction_callback=None,
        dpi=dpi,
        recycle_early_stop_tolerance=0.05,
        max_seq=256,
        max_extra_seq=512
    )

def main():
    setup_logging(Path(".").joinpath("log.txt"))

    args = setup_argparse()

    try:
        query_sequences = read_fasta(args.fasta)
        sorted_pairs = pair_sequences(query_sequences, args.target_seq)
        queries = create_query_list(sorted_pairs)
        model_type = set_model_type(args.is_complex, args.model_type)
        
        run_prediction(
                        queries, 
                        args.result_dir, 
                        args.use_templates, 
                        args.custom_template_path, 
                        args.msa_mode, 
                        model_type, 
                        args.num_recycles, 
                        args.is_complex,
                        args.do_not_overwrite_results,
                        args.pair_mode,
                        args.dpi
                    )

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
