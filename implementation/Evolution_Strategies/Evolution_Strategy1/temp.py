import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--generation')
parser.add_argument('--reps')
args = parser.parse_args()

generation = int(args.generation)
reps = int(args.reps)

data = pd.read_csv('results.csv')
grouped = data.groupby(['generation'])
data = grouped.get_group(generation-1)
data = data.sort_values(['accuracy'], ascending=False).reset_index(drop=True)

print(data.shape[0], reps)
for i in range(0, data.shape[0]-reps-1):
    row = data.iloc[i]
    with open('results.csv', 'a') as f:
        f.write(f"{generation},{int(row[1])},{row[2]},{row[3]},{row[4]},{row[5]},{row[6]}\n")
