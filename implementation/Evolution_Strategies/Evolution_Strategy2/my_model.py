import argparse

import numpy as np
import pickle
import time
import pandas as pd

from model import Network
from model_spec import ModelSpec


import copy
import random
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
OP_SPOTS = MAX_VERTICES - 2   # Input/output vertices are fixed





# nasbench source start



# def random_spec():
#   """Returns a random valid spec."""
#   while True:
#     matrix = np.random.choice(ALLOWED_EDGES, size=(NUM_VERTICES, NUM_VERTICES))
#     matrix = np.triu(matrix, 1)
#     ops = np.random.choice(ALLOWED_OPS, size=(NUM_VERTICES)).tolist()
#     ops[0] = INPUT
#     ops[-1] = OUTPUT
#     spec = api.ModelSpec(matrix=matrix, ops=ops)
#     if nasbench.is_valid(spec):
#       return spec

# def mutate_spec(old_spec, mutation_rate=1.0):
#   """Computes a valid mutated spec from the old_spec."""
#   while True:
#     new_matrix = copy.deepcopy(old_spec.original_matrix)
#     new_ops = copy.deepcopy(old_spec.original_ops)

#     # In expectation, V edges flipped (note that most end up being pruned).
#     edge_mutation_prob = mutation_rate / NUM_VERTICES
#     for src in range(0, NUM_VERTICES - 1):
#       for dst in range(src + 1, NUM_VERTICES):
#         if random.random() < edge_mutation_prob:
#           new_matrix[src, dst] = 1 - new_matrix[src, dst]
          
#     # In expectation, one op is resampled.
#     op_mutation_prob = mutation_rate / OP_SPOTS
#     for ind in range(1, NUM_VERTICES - 1):
#       if random.random() < op_mutation_prob:
#         available = [o for o in nasbench.config['available_ops'] if o != new_ops[ind]]
#         new_ops[ind] = random.choice(available)
        
#     new_spec = api.ModelSpec(new_matrix, new_ops)
#     if nasbench.is_valid(new_spec):
#       return new_spec

# def random_combination(iterable, sample_size):
#   """Random selection from itertools.combinations(iterable, r)."""
#   pool = tuple(iterable)
#   n = len(pool)
#   indices = sorted(random.sample(range(n), sample_size))
#   return tuple(pool[i] for i in indices)

# def run_random_search(max_time_budget=5e6):
#   """Run a single roll-out of random search to a fixed time budget."""
#   nasbench.reset_budget_counters()
#   times, best_valids, best_tests = [0.0], [0.0], [0.0]
#   while True:
#     spec = random_spec()
#     data = nasbench.query(spec)

#     # It's important to select models only based on validation accuracy, test
#     # accuracy is used only for comparing different search trajectories.
#     if data['validation_accuracy'] > best_valids[-1]:
#       best_valids.append(data['validation_accuracy'])
#       best_tests.append(data['test_accuracy'])
#     else:
#       best_valids.append(best_valids[-1])
#       best_tests.append(best_tests[-1])

#     time_spent, _ = nasbench.get_budget_counters()
#     times.append(time_spent)
#     if time_spent > max_time_budget:
#       # Break the first time we exceed the budget.
#       break

#   return times, best_valids, best_tests

# def run_evolution_search(max_time_budget=5e6,
#                          population_size=50,
#                          tournament_size=10,
#                          mutation_rate=1.0):
#   """Run a single roll-out of regularized evolution to a fixed time budget."""
#   nasbench.reset_budget_counters()
#   times, best_valids, best_tests = [0.0], [0.0], [0.0]
#   population = []   # (validation, spec) tuples

#   # For the first population_size individuals, seed the population with randomly
#   # generated cells.
#   for _ in range(population_size):
#     spec = random_spec()
#     data = nasbench.query(spec)
#     time_spent, _ = nasbench.get_budget_counters()
#     times.append(time_spent)
#     population.append((data['validation_accuracy'], spec))

#     if data['validation_accuracy'] > best_valids[-1]:
#       best_valids.append(data['validation_accuracy'])
#       best_tests.append(data['test_accuracy'])
#     else:
#       best_valids.append(best_valids[-1])
#       best_tests.append(best_tests[-1])

#     if time_spent > max_time_budget:
#       break

#   # After the population is seeded, proceed with evolving the population.
#   while True:
#     sample = random_combination(population, tournament_size)
#     best_spec = sorted(sample, key=lambda i:i[0])[-1][1]
#     new_spec = mutate_spec(best_spec, mutation_rate)

#     data = nasbench.query(new_spec)
#     time_spent, _ = nasbench.get_budget_counters()
#     times.append(time_spent)

#     # In regularized evolution, we kill the oldest individual in the population.
#     population.append((data['validation_accuracy'], new_spec))
#     population.pop(0)

#     if data['validation_accuracy'] > best_valids[-1]:
#       best_valids.append(data['validation_accuracy'])
#       best_tests.append(data['test_accuracy'])
#     else:
#       best_valids.append(best_valids[-1])
#       best_tests.append(best_tests[-1])

#     if time_spent > max_time_budget:
#       break

#   return times, best_valids, best_tests
  



# nasbench source end


























def get_matrix():
    mat = np.triu(np.random.randint(2, size=(7, 7)), k=1)

    f = open('mutated-matrices.txt', 'a+')
    lines = list(f.read().split())

    if mat.flatten() in lines:
        get_matrix()

    f.write(f'{"".join(str(m) for m in mat.flatten())}' + '\n')
    f.close()
    return mat


# def get_random_matrix():
#     while True:
#         matrix = np.random.choice(ALLOWED_EDGES, size=(MAX_VERTICES, MAX_VERTICES))
#         matrix = np.triu(matrix, 1)
#         ops = np.random.choice([CONV3X3, CONV1X1, MAXPOOL3X3], size=(MAX_VERTICES)).tolist()
#         ops[0] = 'input'
#         ops[-1] = 'output'
#         spec = api.ModelSpec(matrix=matrix, ops=ops)
#         if nasbench.is_valid(spec):
#             # print("Returning a matrix of shape: "+str(matrix.shape)+", and ops of size: "+str(len(ops)))
#             return matrix, ops

def get_random_matrix():
  """Returns a random valid spec."""
  print("Getting random matrix")
  
  while True:
    matrix = np.random.choice(ALLOWED_EDGES, size=(MAX_VERTICES, MAX_VERTICES))
    matrix = np.triu(matrix, 1)
    ops = np.random.choice([CONV3X3, CONV1X1, MAXPOOL3X3], size=(MAX_VERTICES)).tolist()
    ops[0] = 'input'
    ops[-1] = 'output'
    spec = api.ModelSpec(matrix=matrix, ops=ops)
    if nasbench.is_valid(spec):
      print("get_random_matrix: Returning a matrix of shape: "+str(spec.matrix.shape)+", and ops of size: "+str(len(spec.ops)))
      return spec


def mutate_spec1(old_spec, mutation_rate=1.0):
  """Computes a valid mutated spec from the old_spec."""
  while True:
    new_matrix = copy.deepcopy(old_spec.original_matrix)
    new_ops = copy.deepcopy(old_spec.original_ops)

    # In expectation, V edges flipped (note that most end up being pruned).
    edge_mutation_prob = mutation_rate / MAX_VERTICES
    for src in range(0, MAX_VERTICES - 1):
      for dst in range(src + 1, MAX_VERTICES):
        if random.random() < edge_mutation_prob:
          new_matrix[src, dst] = 1 - new_matrix[src, dst]
          
    # In expectation, one op is resampled.
    op_mutation_prob = mutation_rate / OP_SPOTS
    for ind in range(1, MAX_VERTICES - 1):
      if random.random() < op_mutation_prob:
        available = [o for o in nasbench.config['available_ops'] if o != new_ops[ind]]
        new_ops[ind] = random.choice(available)
        
    new_spec = api.ModelSpec(new_matrix, new_ops)
    if nasbench.is_valid(new_spec):
        print("mutate_spec1: Returning a matrix of shape: "+str(new_spec.matrix.shape)+", and ops of size: "+str(len(new_spec.ops)))
        return new_spec

def get_model_specs(generation, rep):

    if generation == 0:
      rand_spec = get_random_matrix()

      f = open('random-matrices.txt', 'a+')
      lines = list(f.read().split())

      if rand_spec.matrix.flatten() in lines:
          f.close()
          get_model_specs()

      f.write(f'{"".join(str(m) for m in rand_spec.matrix.flatten())}' + '\n')
      f.close()

      f_name = str(time.time()).replace('.', '')
      with open(f"./specs/{f_name}.spec", 'wb') as f_spec:
        pickle.dump(rand_spec, f_spec)

      with open('specs.txt', 'a+') as fn:
        fn.write(f_name + '\n')
      
      return rand_spec
    else:
      generation -= 1
      data = pd.read_csv('results.csv')
      grouped = data.groupby(['generation'])
      data = grouped.get_group(generation)
      data = data.sort_values(['accuracy'], ascending=False).reset_index(drop=True)
      row = data.iloc[rep]

      spec = int(row[1])

      with open(f'./specs/{spec}.spec', 'rb') as f:
        rand_spec = pickle.load(f)

      rand_spec = mutate_spec1(rand_spec)

      f_name = str(time.time()).replace('.', '')
      with open(f"./specs/{f_name}.spec", 'wb') as f_spec:
        pickle.dump(rand_spec, f_spec)

      with open('specs.txt', 'a+') as fn:
        fn.write(f_name + '\n')

      return rand_spec


def get_model(**kwargs):
    args = argparse.Namespace(num_labels=10, num_modules_per_stack=1, num_stacks=1, stem_out_channels=12)
    specs = get_model_specs(generation=int(kwargs['generation']), rep=int(kwargs['rep']))
    return Network(specs, args)
