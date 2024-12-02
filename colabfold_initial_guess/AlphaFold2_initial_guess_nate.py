import argparse
import sys
import os
import re
from pathlib import Path

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from colabfold.utils import setup_logging
from colabfold.batch_ig import run_ig, set_model_type

from pathlib import Path


"""
###suggested running modes to try, could adjust num_recycles...
###I would not suggest other combinations of arguments

#nate_ig_mode_both_models
python /home/drhicks1/ColabFold_dev/AlphaFold2_initial_guess.py --silent your.silent --num_recycles 2 --msa_mode single_sequence --pair_mode unpaired --model_type AlphaFold2-ptm --num_models 2 --model_order 1,2

#single_seq_classic
python /home/drhicks1/ColabFold_dev/AlphaFold2_initial_guess.py --silent your.silent --num_recycles 2 --msa_mode single_sequence --pair_mode unpaired+paired --model_type AlphaFold2-ptm --num_models 2 --model_order 1,2

#single_seq_multimer
python /home/drhicks1/ColabFold_dev/AlphaFold2_initial_guess.py --silent your.silent --num_recycles 2 --msa_mode single_sequence --pair_mode unpaired+paired --model_type AlphaFold2-multimer-v3 --num_models 5 --model_order 1,2,3,4,5

#msa_classic
python /home/drhicks1/ColabFold_dev/AlphaFold2_initial_guess.py --silent your.silent --num_recycles 2 --msa_mode 'MMseqs2 (UniRef+Environmental)' --pair_mode unpaired+paired --model_type AlphaFold2-ptm --num_models 2 --model_order 1,2

#msa_multimer
python /home/drhicks1/ColabFold_dev/AlphaFold2_initial_guess.py --silent your.silent --num_recycles 2 --msa_mode 'MMseqs2 (UniRef+Environmental)' --pair_mode unpaired+paired --model_type AlphaFold2-multimer-v3 --num_models 5 --model_order 1,2,3,4,5
"""


def validate_args(args):
    if args.model_type == 'AlphaFold2-ptm' and any(m not in [1, 2] for m in args.model_order):
        raise ValueError("For AlphaFold2-ptm, model_order can only include models 1 and 2.")

    if args.model_type == 'AlphaFold2-multimer-v3' and any(m not in range(1, 6) for m in args.model_order):
        raise ValueError("For AlphaFold2-multimer-v3, model_order must be in the range 1 to 5.")

    if len(args.model_order) > args.num_models:
        raise ValueError("model_order cannot contain more models than num_models.")

def setup_argparse():
    parser = argparse.ArgumentParser(description='Run AlphaFold predictions with customizable parameters.')
    parser.add_argument('-silent', '--silent', type=str, required=True, help='Path to a silent file for initial guess predictions')
    parser.add_argument('--result_dir', type=str, default='./', help='Directory where results will be saved.')
    parser.add_argument('--outname', type=str, default='out', help='name prefix for score file, silent file, and check point file')
    parser.add_argument('--msa_mode', type=str, default='single_sequence', choices=['MMseqs2 (UniRef+Environmental)', 'MMseqs2 (UniRef only)', 'single_sequence', 'custom'], help='Mode for multiple sequence alignment. Absolutely do not submit many jobs with MSA. It would overload colabfold server. Should implement MSA on digs...')
    parser.add_argument('--pair_mode', type=str, default='unpaired', choices=['unpaired+paired', 'paired', 'unpaired'], help='Pairing mode for sequences.')
    parser.add_argument('--pairing_strategy', type=str, default='greedy', choices=["greedy", "complete"], help='Pairing strategy for sequences.')
    parser.add_argument('--model_type', type=str, default='AlphaFold2-ptm', choices=['AlphaFold2-ptm', 'AlphaFold2-multimer-v3', 'auto'], help='Type of AlphaFold model to use.')
    parser.add_argument('--num_models', type=int, default=2, help='Number of models to run. Initial guess/AlphaFold2-ptm could do 2, Initial guess/AAlphaFold2-multimer-v3 could do all 5,')
    parser.add_argument('--model_order', type=lambda s: [int(item) for item in s.split(',')], default=[1,2], help='Which models to run, from 1 to 5 (comma-separated list)')    
    parser.add_argument('--num_recycles', type=int, default=2, help='Number of recycles to use in AlphaFold prediction (default of 2 = 3 total cycles).')
    parser.add_argument('--is_monomer', action='store_false', dest="is_complex", help='Set this flag if predicting a monomer. This may not be properly implemented')
    parser.add_argument('--template_mode', type=str, default='none', choices=['none', 'pdb100', 'custom'], help='Template mode for structure prediction; dont use, initial guess would override this.')
    parser.add_argument('--use_templates', action='store_true', help='Set this flag if you want to use templates; dont use, initial guess would override this.')
    parser.add_argument('--custom_template_path', type=str, default=None, help='path to custom templates; currently not implemented')
    parser.add_argument('--recycle_early_stop_tolerance', type=float, default=0.05, help='early stop if distograms converge')
    parser.add_argument('--pae_interaction_cut', type=float, default=29, help='stop predictions early if pae_interaction is this bad')
    parser.add_argument('--interface_rmsd_cut', type=float, default=20, help='stop predictions early if interface_rmsd is this bad')
    parser.add_argument('--do_not_template_chain_2_plus', action='store_false', dest="template_chain_2_plus", help='trun off templating for chain 2 or higher, ie target')
    parser.add_argument('--template_chain_1', action='store_true', help='Set this flag if you want to turn on templating for chain 1, ie binder')

    return parser.parse_args()

def main():
    setup_logging(Path(".").joinpath("log.txt"))

    args = setup_argparse()
    validate_args(args)

    try:
        model_type = set_model_type(args.is_complex, args.model_type)
        
        run_ig(
            queries=[],
            result_dir=args.result_dir,
            use_templates=args.use_templates,
            custom_template_path=args.custom_template_path,
            msa_mode=args.msa_mode,
            model_type=args.model_type,
            num_models=args.num_models,
            num_recycles=args.num_recycles,
            recycle_early_stop_tolerance=args.recycle_early_stop_tolerance,
            num_seeds=1, #fixed as 1 prediction/seed per
            use_dropout=False, #only useful with more seeds
            model_order=args.model_order,
            is_complex=args.is_complex,
            data_dir=Path("/net/databases/alphafold/"),
            pair_mode=args.pair_mode,
            pairing_strategy=args.pairing_strategy,
            max_msa=None,
            use_cluster_profile=True,
            save_recycles=False,
            user_agent="colabfold/google-colab-main",
            initial_guess=args.silent,
            template_chain_1=args.template_chain_1,
            template_chain_2_plus=args.template_chain_2_plus,
            pae_interaction_cut=args.pae_interaction_cut,
            interface_rmsd_cut=args.interface_rmsd_cut,
            recompile_padding=0,
            outname=args.outname,
        )

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
