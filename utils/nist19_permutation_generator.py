import argparse
import numpy as np
from random import shuffle
from utils.prepare_nist19_dataset import generate_paths


def argument_parser():
    arg_pars = argparse.ArgumentParser(
        description='NIST19 data permutation generator'
    )
    arg_pars.add_argument('--dataset',
                          required=True,
                          type=str
                          )
    arg_pars.add_argument('--save-npy',
                          required=True,
                          type=str
                          )
    return arg_pars.parse_args()


def main(args):
    data = generate_paths(args.dataset)
    permutation = np.array(list(range(len(data))))
    shuffle(permutation)
    np.save(args.save_npy, permutation)


if __name__ == '__main__':
    args = argument_parser()
    main(args)
