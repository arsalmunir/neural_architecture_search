from model_spec import ModelSpec
from model import Network

import argparse

def get_model_specs():
    matrix = [[0, 1, 1, 0, 1, 1, 1],
               [0, 0, 0, 0, 1, 1, 0],
               [0, 0, 0, 1, 1, 1, 1],
               [0, 0, 0, 0, 0, 1, 1],
               [0, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0]]
    ops = ['input', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3', 'output']

    return ModelSpec(matrix, ops)


def get_model(**kwargs):
    """
    For simplicity, I have reduced the number of stacks, number of modules per stacks and output channels of each stem.
    """
    args = argparse.Namespace(num_labels=10, num_modules_per_stack=1, num_stacks=1, stem_out_channels=12)

    specs = get_model_specs()

    return Network(specs, args)

if __name__ == '__main__':
    model = get_model(output_size=10)

    print('--- Model info ---')
    print(model)
    