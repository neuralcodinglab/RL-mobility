
import numpy as np
import pandas as pd
import os

DATA_PATH = "D:\\RLHallwayData"

data = pd.read_csv(os.path.join(DATA_PATH,"_labels.csv"))

print(data.columns)


data['idx_by_file'] = data.index.values
data = data.sort_values(['env_idx', 'y_pos', 'x_pos'])
print(data.index)


print(data[['y_pos', 'x_pos', 'pos_in_cycle']].head(15))


print(data.sort_values(['env_idx', 'y_pos', 'x_pos']).head(15).index)
