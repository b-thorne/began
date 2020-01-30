import tensorflow as tf
import tensorflow_probability as tfp 
import began
import argparse

parser = argparse.ArgumentParser(description='Prepare Planck map for training.')
parser.add_argument('planck_path', metavar='path', type=str, help='the path to list')
args = parser.parse_args()

if __name__ == '__main__':
    print(args)