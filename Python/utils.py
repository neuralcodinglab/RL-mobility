import numpy as np
import cv2
import torch
import itertools
import re
import pandas as pd
import yaml
import os
from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

def load_train_configs(yaml_file):
    """Loads a configuration from yaml file, returns DataFrame with all models and their training parameters"""

    # Load yaml file as dictionary
    with open(yaml_file) as file:
        raw_content = yaml.load(file,Loader=yaml.FullLoader) # nested dictionary
        unpacked = {k:v for params in raw_content.values() for k,v in params.items()}
        assert not any(isinstance(v,dict) for v in unpacked.values()), "Only single-valued parameters or lists of single-valued parameters are accepted"

    # single-valued (fixed) training parameters
    fxd_params = {k:v for k,v in unpacked.items() if type(v) is not list}

    # variable training parameters. All combinations are used in training (grid-search)
    var_params = {k:v for k,v in unpacked.items() if type(v) is list}
    combinations = list(itertools.product(*var_params.values()))

    # Generate model names based on the training combinations
    model_names = []
    for c in combinations:
        exp_condition = '-'.join(['{:.4}_{}'.format(k,v) for k,v in zip(var_params.keys(),c)])
        model_name = '_'.join([fxd_params['NAME'],exp_condition])
        model_name = re.sub(r'[^\w\s-]', '', model_name.lower()) # Remove illegal characters
        model_name = re.sub(r'[-\s]+', '-', model_name).strip('-_') # Replace spaces with '_'
        model_names.append(model_name)

    # return as Pandas DataFrame
    out = pd.DataFrame(combinations, columns=[k.lower() for k in var_params.keys()])
    out.insert(0,'status','todo')
    out.insert(0,'model_name',model_names)
    for k,v in fxd_params.items():
        out[k.lower()]=v
    return out.replace(['None'],[None]).replace([np.nan],[None]).set_index('model_name')

## Write replay memory to output videos
def save_replay(replay_memory, filename, size):
    out = cv2.VideoWriter(filename,
                          cv2.VideoWriter_fourcc('M','J','P','G'),
                          2, size)
    for i, (state, action, next_state, reward) in enumerate(replay_memory):
        frame = (255*state[0,-1].detach().cpu().numpy()).astype('uint8')
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR )
        frame = cv2.putText(frame,'reward: {:0.1f}'.format(reward.item()),(0,20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.35,color=(0,0,255))
        frame = cv2.putText(frame,'action: {}'.format(action.item()),(0,10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.35,color=(0,0,255))
        out.write(frame)
    out.release()
