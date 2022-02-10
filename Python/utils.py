import numpy as np
import cv2
import torch
import itertools
import re
import pandas as pd
import yaml


def load_train_configs(yaml_file):
    """Loads a configuration from yaml file, returns DataFrame with all models and their training parameters"""

    # Load yaml file as dictionary
    with open(yaml_file) as file:
        raw_content = yaml.load(file) # nested dictionary
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
        model_name = '-'.join(['{:.4}_{}'.format(k,v) for k,v in zip(var_params.keys(),c)])
        model_name = re.sub(r'[^\w\s-]', '', model_name.lower()) # Remove illegal characters
        model_name = re.sub(r'[-\s]+', '-', model_name).strip('-_') # Replace spaces with '_'
        model_names.append(model_name)

    # return as Pandas DataFrame
    out = pd.DataFrame(combinations, columns=var_params.keys())
    out.insert(0,'status','todo')
    out.insert(0,'model name',model_names)
    for k,v in fxd_params.items():
        out[k]=v
    return out.replace(['None'],[None]).replace([np.nan],[None])

## Write replay memory to output videos
def save_replay():
    out = cv2.VideoWriter(os.path.join(OUT_PATH,'{}.avi'.format(model_name)),
                          cv2.VideoWriter_fourcc('M','J','P','G'),
                          2, (IMSIZE,IMSIZE))
    for i, (state, action, next_state, reward) in enumerate(agent.memory.memory):
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
