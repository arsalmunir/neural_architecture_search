import argparse
import json

parser = argparse.ArgumentParser(description='Just to get run and generation number!')
parser.add_argument('--run', required=True)
parser.add_argument('--gen', required=True)
args = parser.parse_args()
run = int(args.run)
gen = int(args.gen)

with open(f'train/train_{run}/eval_{run}/config.json') as f:
    data = json.load(f)

specs = open('specs.txt', 'r')
lines = list(specs.read().split('\n'))
specs.close()

acc = data['evaluation'][0]['evaluation_accuracy']
f1 = data['evaluation'][0]['evaluation_f1']
loss = data['evaluation'][0]['evaluation_loss']
precision = data['evaluation'][0]['evaluation_precision']
recall = data['evaluation'][0]['evaluation_recall']

with open('results.csv', 'a') as f:
    f.write(f'{gen},{lines[run]},{acc},{f1},{loss},{precision},{recall}\n')
