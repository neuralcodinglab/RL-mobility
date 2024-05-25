# Numpy, OpenCV, os, sys
import cv2
import os, sys
import pyClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Connect to Unity environment
ip = "127.0.0.1"  # Ip address that the TCP/IP interface listens to
port = 12000  # Port number that the TCP/IP interface listens to
size = 128
screen_height = screen_width = size
channels = 16
environment = pyClient.Environment(ip=ip, port=port, size=size, channels=channels)
assert (environment.client is not None), "Please start Unity server environment first!"

# Start variables
N_RUNS = 100
N_CYCLES_PER_RUN = 50
save_path = "D:\\RLHallwayData"
env_type_names = {0: 'plain', 1: 'textured'}
env_type = 0
file_idx = 0
env_data = pd.DataFrame()

# reset the environment
def reset_env(env_type):
    environment.reset(env_type)
    environment.step(1)  # start in left position
    for _ in range(4):  # move 4 places forward
        environment.step(0)
    return environment.step(0)

def save_files(save_path, file_idx, state_raw, pos_in_cycle):
    # Write files
    cv2.imwrite(os.path.join(save_path, f"{file_idx:06d}_colors.png"), state_raw['colors'][:, :, ::-1])
    cv2.imwrite(os.path.join(save_path, f"{file_idx:06d}_depth.png"), state_raw['depth'])
    cv2.imwrite(os.path.join(save_path, f"{file_idx:06d}_objseg.png"), state_raw['objseg'])
    cv2.imwrite(os.path.join(save_path, f"{file_idx:06d}_semseg.png"), state_raw['semseg'])

def update_env_data(file_idx,**kwargs):
    for col_name, col_val in kwargs.items():
        env_data.loc[file_idx, col_name] = col_val


# Do 50 runs in the plain and 50 runs in the complex environment
envs = [0, ] * (N_RUNS//2) + [1, ] * (N_RUNS//2)

# In each run, perform the following cycle of actions:
action_cycle = [2, 2, 0, 1, 1] # r r f l l (returning at leftmost pos, one y-pos ahead in env)
x_positions = [0, 1, 2, 2, 1, 0]
n_cycles = N_CYCLES_PER_RUN

for env_idx, env_type in enumerate(envs):
    end, reward, state_raw = reset_env(env_type)

    for cycle_idx in range(n_cycles):
        pos_in_cycle = 0
        for a in action_cycle:
            save_files(save_path, file_idx, state_raw, pos_in_cycle)
            update_env_data(file_idx,
                            env_idx=env_idx,
                            cycle_idx=cycle_idx,
                            pos_in_cycle=pos_in_cycle,
                            x_pos=x_positions[pos_in_cycle],
                            y_pos=cycle_idx * 2 + pos_in_cycle // 3,
                            env_complexity=env_type_names[env_type],
                            reward=reward,
                            end=end,
                            img_file=f"{file_idx:06d}")
            end, reward, state_raw = environment.step(a)
            pos_in_cycle += 1
            file_idx += 1
        env_data.to_csv(os.path.join(save_path, '_labels.csv'))
        print(f"env_idx: {env_idx}, file_idx: {file_idx}")


