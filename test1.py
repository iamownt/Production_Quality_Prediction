import numpy as np

abc_file = np.load("abc.npy", allow_pickle=True)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("xx", type=float, default=0)

args = parser.pa