import argparse

import numpy as np

from model import Network
from model_spec import ModelSpec

#install nasbench
#!git clone https://github.com/google-research/nasbench
#!pip install ./nasbench

from nasbench import api

nasbench = api.NASBench('nasbench_only108.tfrecord')


#random specs constants
ALLOWED_EDGES = [0, 1] #binary adj matrix
MAXPOOL3X3 = 'maxpool3x3'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAX_VERTICES = 7



def get_matrix():
    mat = np.triu(np.random.randint(2, size=(7, 7)), k=1)

    f = open('matrices.txt', 'a+')
    lines = list(f.read().split())

    if mat.flatten() in lines:
        get_matrix()

    f.write(f'{"".join(str(m) for m in mat.flatten())}' + '\n')
    f.close()
    return mat


def get_random_matrix():
    while True:
        matrix = np.random.choice(ALLOWED_EDGES, size=(MAX_VERTICES, MAX_VERTICES))
        matrix = np.triu(matrix, 1)
        ops = np.random.choice([CONV3X3, CONV1X1, MAXPOOL3X3], size=(MAX_VERTICES)).tolist()
        ops[0] = 'input'
        ops[-1] = 'output'
        spec = api.ModelSpec(matrix=matrix, ops=ops)
        if nasbench.is_valid(spec): #if matrix is valid, check if this was already generated?
            f = open('matrices.txt', 'a+')
            lines = list(f.read().split())
            if matrix.flatten() in lines: # if already generated, call get_random_matrix again to generate a new one.
                get_random_matrix()
            f.write(f'{"".join(str(m) for m in matrix.flatten())}' + '\n') #if it was not generated before, save it in matrices.txt and return the matrix and ops to original caller
            f.close()
            print("Returning a matrix of shape: "+str(matrix.shape)+", and ops of size: "+str(len(ops)))
            return matrix, ops


def get_model_specs():



    # matrix = [[0, 1, 1, 0, 1, 1, 1],
    #            [0, 0, 0, 0, 1, 1, 0],
    #            [0, 0, 0, 1, 1, 1, 1],
    #            [0, 0, 0, 0, 0, 1, 1],
    #            [0, 0, 0, 0, 0, 0, 1],
    #            [0, 0, 0, 0, 0, 0, 0],
    #            [0, 0, 0, 0, 0, 0, 0]]
    # ops = ['input', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3', 'output']

    matrix, ops = get_random_matrix()
#    ops = ['input', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3', 'output']
    return ModelSpec(matrix, ops)


def get_model(**kwargs):
    args = argparse.Namespace(num_labels=10, num_modules_per_stack=1, num_stacks=1, stem_out_channels=12)
    specs = get_model_specs()
    return Network(specs, args)