import math
import datetime
import os, sys
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from IPython.display import Audio
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchsummary import summary

import cv2

# local files
sys.path.insert(0, '../')
import pyClient
import utils
import model
from model import Transition
import imgproc

ENDPOINTS = ['wall_collisions','box_collisions','endless_loops',
             'step_count','cumulative_reward']

def test(agent, environment, img_processing, cfg, store_replay=False, store_Q_predictions=False):

    if store_Q_predictions:
        # Forward hook to store Q_predictions
        agent.Q_predictions = []
        def Q_hook(m, i, o):
            agent.Q_predictions.append(o)
        agent.policy_net.head.register_forward_hook(Q_hook)
    
    # Counters
    wall_collisions = 0
    box_collisions = 0
    endless_loops = 0
    step_count = 0
    cumulative_reward = 0
    side_steps = 0 # Side-step counter (to prevent endless loops)


    # Reset environment at start of episode
    _, _, _ = environment.reset(cfg['training_condition'])

    # Create an empty frame stack and fill it with frames
    frame_stack = imgproc.FrameStack(stack_size=cfg['stack_size'] )
    for _ in range(cfg['stack_size'] ):
        _, _, frame_raw = environment.step(0)
        frame = img_processing(frame_raw).to(agent.device)
        state = frame_stack.update_with(frame)

    # Episode starts here:
    agent_finished = False
    while not agent_finished:

        # 1. Agent performs a step (based on the current state) and obtains next state
        action = agent.select_action(state, validation=True)
        side_steps = side_steps + 1  if action != 0 else 0
        if side_steps > cfg['reset_after_nr_sidesteps']:
            endless_loops +=1
            action = torch.zeros_like(action) # force forward step
            side_steps = 0

        end, reward, frame_raw = environment.step(action.item())
#         agent_finished = cfg['reset_end_is_{}'.format(end)]
        agent_finished = end == 3 # Only finished after the entire test-hallway is walked
        frame = img_processing(frame_raw).to(agent.device)
        next_state = frame_stack.update_with(frame) if not agent_finished else None

        # 2. Interpret reward signal
        if reward > 100:
            reward = -(reward -100)
            reward = reward * cfg['reward_multiplier']

        # 3. Store performance and training measures
        step_count += 1
        cumulative_reward += reward;
        if end == 1:
            box_collisions += 1
        if end == 2:
            wall_collisions +=1
            
        if store_replay:
            # 4. Push the transition to replay memory (in the right format & shape)
            reward = torch.tensor([reward], device=agent.device,dtype=torch.float)
            action = action.unsqueeze(0)
            agent.memory.push(state, action, next_state, reward)
        
        # The new state becomes the current state
        if next_state is not None:
            state = next_state.clone()

    return {'wall_collisions':wall_collisions,
            'box_collisions' :box_collisions,
            'endless_loops' :endless_loops,
            'step_count' :step_count,
            'cumulative_reward' :cumulative_reward}

# def test(agent, environment, img_processing, cfg):

#     # Counters
#     wall_collisions = 0
#     box_collisions = 0
#     endless_loops = 0
#     step_count = 0
#     cumulative_reward = 0
#     side_steps = 0 # Side-step counter (to prevent endless loops)


#     # Reset environment at start of episode
#     _, _, _ = environment.reset(cfg['training_condition'])

#     # Create an empty frame stack and fill it with frames
#     frame_stack = imgproc.FrameStack(stack_size=cfg['stack_size'] )
#     for _ in range(cfg['stack_size'] ):
#         _, _, frame_raw = environment.step(0)
#         frame = img_processing(frame_raw).to(agent.device)
#         state = frame_stack.update_with(frame)

#     # Episode starts here:
#     agent_finished = False
#     while not agent_finished:

#         # 1. Agent performs a step (based on the current state) and obtains next state
#         action = agent.select_action(state, validation=True)
#         side_steps = side_steps + 1  if action != 0 else 0
#         if side_steps > cfg['reset_after_nr_sidesteps']:
#             endless_loops +=1
#             action = torch.zeros_like(action) # force forward step
#             side_steps = 0

#         end, reward, frame_raw = environment.step(action.item())
#         agent_finished = cfg['reset_end_is_{}'.format(end)]
#         frame = img_processing(frame_raw).to(agent.device)
#         state = frame_stack.update_with(frame) if not agent_finished else None

#         # 2. Interpret reward signal
#         if reward > 100:
#             reward = -(reward -100)

#         # 3. Store performance and training measures
#         step_count += 1
#         cumulative_reward += reward;
#         if end == 1:
#             box_collisions += 1
#         if end == 2:
#             wall_collisions +=1

#     return {'wall_collisions':wall_collisions,
#             'box_collisions' :box_collisions,
#             'endless_loops' :endless_loops,
#             'step_count' :step_count,
#             'cumulative_reward' :cumulative_reward}


# TODO add argparser and main script (to test list of models at once) 
