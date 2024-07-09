import argparse

def main(args):
    import glob
    import random
    import numpy as np
    import json
    
    mpnn_alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    
    mpnn_alphabet_dict = {'A': 0,'C': 1,'D': 2,'E': 3,'F': 4,'G': 5,'H': 6,'I': 7,'K': 8,'L': 9,'M': 10,'N': 11,'P': 12,'Q': 13,'R': 14,'S': 15,'T': 16,'V': 17,'W': 18,'Y': 19,'X': 20}
    my_dict = {}
    my_dict["srtRe1657"] = {}
    my_bias_array = 10* np.random.normal(0, 1, [109,21])
    my_dict["srtRe1657"]["A"] = my_bias_array.tolist()

    with open(args.output_path, 'w') as f:
        f.write(json.dumps(my_dict) + '\n')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--output_path", type=str, help="Path to the output dictionary")

    args = argparser.parse_args()
    main(args) 
