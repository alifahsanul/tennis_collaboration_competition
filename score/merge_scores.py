import csv
import numpy as np

with open('merged_scores.csv', 'r') as f:
    reader = csv.reader(f)
    scores1 = list(reader)
    scores1 = [float(x[0]) for x in scores1]


with open('scores.csv', 'r') as f:
    reader = csv.reader(f)
    scores2 = list(reader)
    scores2 = [float(x[0]) for x in scores2]

merged_scores = [x for x in (scores1 + scores2)]

np.savetxt('merged_scores.csv', merged_scores, delimiter = ',')
