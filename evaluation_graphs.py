import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas

with open('eval_metrics.csv', 'r') as f:
  df = pandas.read_csv(f)
  fig = plt.figure()

  r = np.arange(len(df))
  for col in df:
    if col == 'User #':
      continue
    plt.plot(r, list(df[col]), label=col)
  plt.xlabel('Iteration')
  plt.ylabel('Evaluation Value')
  plt.legend()
  plt.show()