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


import itertools
import re
import pandas as pd
import numpy as np


# local files
sys.path.insert(0, '../')
import pyClient
import utils
import model
import imgproc
from model import Transition





def validation_loop(agent,environment,img_processing, cfg, val_seeds=[251,252,253,254,255]):
    # How to handle the different end signals
    RESET_UPON_END_SIGNAL = {0:False,  # Nothing happened
                             1:False,   # Box collision
                             2:False,   # Wall collision
                             3:True}    # Reached step target

    # Set nn.module to evaluation mode
    agent.policy_net.eval()

    # Reset counters
    wall_collisions = 0
    box_collisions = 0
    total_reward = 0
    endless_loops = 0
    step_count = 0


    for seed in val_seeds:

        # Reset environment at start of episode
        _, _, _ = environment.setRandomSeed(seed)
        _, _, _ = environment.reset(cfg['training_condition'])


        # Create an empty frame stack and fill it with frames
        frame_stack = imgproc.FrameStack(stack_size=cfg['stack_size'] )
        for _ in range(cfg['stack_size'] ):
            _, _, state_raw = environment.step(0)
            frame = img_processing(state_raw).to(agent.device)
            state = frame_stack.update_with(frame)

        side_steps = 0

        # Episode starts here:
        for t in count():

            # 1. Agent performs a step (based on the current state) and obtains next state
            action = agent.select_action(state,validation=True)
            end, reward, next_state_raw = environment.step(action.item())
            frame = img_processing(next_state_raw).to(agent.device)
            next_state = frame_stack.update_with(frame)
            side_steps = side_steps + 1  if action != 0 else 0

            # 2. interpret reward
            if reward > 100:
                reward = -(reward -100)
            reward *= cfg['reward_multiplier']


            # 3. Store performance and training measures
            total_reward += reward;
            if end == 1:
                box_collisions += 1
            if end == 2:
                wall_collisions +=1

            # 4. the episode ends here after reaching step target or too many side steps
            if side_steps>cfg['reset_after_nr_sidesteps']:
                endless_loops += 1
                step_count += t
                break
            elif RESET_UPON_END_SIGNAL[end]:
                step_count += t
                break
            else:
                state = next_state

    print('step count {} wall_collisions: {}, box_collisions: {}, endless_loops: {}, total_reward: {}'.format(step_count, wall_collisions, box_collisions, endless_loops, total_reward))
    return step_count, wall_collisions, box_collisions, endless_loops, total_reward




def train(agent, environment, img_processing, optimizer, cfg):

    # For reproducability, reset the RNG seed
    torch.manual_seed(cfg['seed'])

    # Write header to logfile
    with open(cfg['logfile'], 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['episode','step_count',
                         'wall_collisions', 'box_collisions',
                         'endless_loops','reward', 'epsilon', 'train_loss', 'validation'])

    # Counters
    wall_collisions = 0
    box_collisions = 0
    episode_reward = 0
    endless_loops = 0
    total_loss = 0
    step_count = 0
    best_reward = 0

    for episode in range(cfg['max_episodes']):

        # Valdation loop
        if episode % 50 == 0:
            val_performance = validation_loop(agent,environment,img_processing,cfg)
            val_reward = val_performance[-1]

            # Save best model
            if val_reward > best_reward:
                print("new best model")
                best_reward = val_reward
                torch.save(agent.policy_net.state_dict(), cfg['model_path'])

            # Write validation performance to log file
            with open(cfg['logfile'], 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([episode, *val_performance,0, 0, 1])

        # Write training performance to log file
        with open(cfg['logfile'], 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([episode,step_count,
                             wall_collisions, box_collisions,
                             endless_loops, episode_reward,agent.eps_threshold,total_loss, 0])

        # Reset counters
        total_loss = 0 # COMMENT OUT TO REGISTER CUMULATIVE LOSS
        wall_collisions = 0
        box_collisions = 0
        episode_reward = 0
        endless_loops = 0
        step_count = 0
        side_steps = 0 # Side-step counter (to prevent endless loops)


        # Stop training after (either maximum number of steps or maximum number of episodes)
        if agent.step_count > cfg['max_steps']:
            break


        # Target net is updated once in a few episodes (double Q-learning)
        if episode % cfg['target_update']  == 0:  #episodes
            print('episode {}, target net updated'.format(episode))
            agent.update_target_net()


        # Reset environment at start of episode
        seed = torch.randint(250,(1,)).item()
        _, _, _ = environment.setRandomSeed(seed)
        _, _, _ = environment.reset(cfg['training_condition'])

        # Create an empty frame stack and fill it with frames
        frame_stack = imgproc.FrameStack(stack_size=cfg['stack_size'] )
        for _ in range(cfg['stack_size'] ):
            _, _, frame_raw = environment.step(0)
            frame = img_processing(frame_raw).to(agent.device)
            state = frame_stack.update_with(frame)

        # Episode starts here:
        for t in count():

            # 1. Agent performs a step (based on the current state) and obtains next state
            agent.policy_net.eval()
            action = agent.select_action(state)
            side_steps = side_steps + 1  if action != 0 else 0
            end, reward, frame_raw = environment.step(action.item())
            agent_died = cfg['reset_upon_end_signal'][end] or side_steps > cfg['reset_after_nr_sidesteps']
            frame = img_processing(frame_raw).to(agent.device)
            next_state = frame_stack.update_with(frame) if not agent_died else None

            # 2. Interpret reward signal
            if reward > 100:
                reward = -(reward -100)
            reward *= cfg['reward_multiplier']

            # 3. Push the transition to replay memory (in the right format & shape)
            reward = torch.tensor([reward], device=agent.device,dtype=torch.float)
            action = action.unsqueeze(0)
            agent.memory.push(state, action, next_state, reward)


            # 4. optimize model
            agent.policy_net.train()
            if len(agent.memory) > cfg['replay_start_size']:

                state_action_values, expected_state_action_values = agent.forward()

                # Compute Huber loss
                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
                total_loss += loss.item()

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(agent.policy_net.parameters(), 1)

                # Update the model parameters
                optimizer.step()

            else:
                # Do not count as optimization loop
                agent.step_count = 0

            # 5. Store performance and training measures
            step_count += 1
            episode_reward += reward.item();
            if end == 1:
                box_collisions += 1
            if end == 2:
                wall_collisions +=1
            if side_steps > cfg['reset_after_nr_sidesteps']:
                endless_loops +=1

            # 6. the episode ends here if agent performed any 'lethal' action (specified in RESET_UPON_END_SIGNAL)
            if agent_died:
                break
            else:
                state = next_state

if __name__ == "__main__":
    ### TODO: ARGPARSER ###

    ## Environment
    IMSIZE = 128
    STACK_SIZE = 4
    N_ACTIONS = 3
    IP  = "127.0.0.1" # Ip address that the TCP/IP interface listens to
    PORT = 13000       # Port number that the TCP/IP interface listens to
    environment =  pyClient.Environment(ip = IP, port = PORT, size = IMSIZE) # or choose # DummyEnvironment()

    ## Image processing
    PHOSPHENE_RESOLUTION = 50
    img_processing = imgproc.ImageProcessor(phosphene_resolution = PHOSPHENE_RESOLUTION)

    ## DQN Agent
    BATCH_SIZE = 128 #original 128
    GAMMA = 0.5
    EPS_START = 0.95
    EPS_END = 0.05
    EPS_DECAY_steps = 4000
    EPS_DECAY = (EPS_START - EPS_END)/EPS_DECAY_steps
    REPLAY_START_SIZE =  1500
    TARGET_UPDATE = 10 #episodes
    DEVICE = 'cpu'
    MEMORY_CAPACITY = 12000
    agent = model.DoubleDQNAgent(imsize=IMSIZE,
                     in_channels=STACK_SIZE,
                     n_actions=N_ACTIONS,
                     memory_capacity=MEMORY_CAPACITY,
                     eps_start=EPS_START,
                     eps_end=EPS_END,
                     eps_delta=EPS_DECAY,
                     gamma_discount = GAMMA,
                     batch_size = BATCH_SIZE,
                     device=DEVICE)

    ## Optimizer
    LR_DQN = 0.01
    optimizer = optim.Adam(agent.policy_net.parameters(), lr = LR_DQN)

    ## Training parameters
    OUT_PATH = './DemoTraining'
    MODEL_PATH = os.path.join(OUT_PATH,'demo.pth')
    LOGFILE = os.path.join(OUT_PATH,'train_stats.csv')
    SEED = 0
    TRAINING_CONDITION = 0
    MAX_EPISODES = 1000 # number of episodes (an episode ends after agent hits a box)
    MAX_STEPS  = 1e6 # number of optimization steps (each time step the model parameters are updated)
    RESET_AFTER_NR_SIDESTEPS = 5
    RESET_UPON_END_SIGNAL = {0:False,  # Nothing happened
                             1:True,   # Box collision
                             2:False,   # Wall collision
                             3:True}  # Reached step target
    REWARD_MULTIPLIER        = 1.



    ## Start training
    cfg = dict()
    cfg['model_path']               = MODEL_PATH # Save path for model
    cfg['logfile']                  = LOGFILE # To save the optimizaiton stats
    cfg['seed']                     = SEED # for reproducability of random factors
    cfg['training_condition']       = TRAINING_CONDITION # 0: plain training, 1: complex training, 2: plain testing 3: complex testing
    cfg['max_episodes']             = MAX_EPISODES
    cfg['max_steps']                = MAX_STEPS # Training stops after either max episodes is reached, or max optimization steps
    cfg['stack_size']               = STACK_SIZE # For frame stacking
    cfg['target_update']            = TARGET_UPDATE #Number of episodes after which DQN target net is updated
    cfg['reset_after_nr_sidesteps'] = RESET_AFTER_NR_SIDESTEPS # Training is stopped when model keeps side stepping (i.e. it stops reveicing positive rewards)
    cfg['reset_upon_end_signal']    = RESET_UPON_END_SIGNAL # Decide whether to consider different end signals as final state (i.e. box collision, wall collision, step target reached)
    cfg['replay_start_size']        = REPLAY_START_SIZE # Start optimizing when replay memory contains this number of transitions
    cfg['reward_multiplier']        = REWARD_MULTIPLIER # Multiplies the reward signal with this value
    print('training...')
    if not os.path.isdir(OUT_PATH):
        os.makedirs(OUT_PATH)
    train(agent, environment, img_processing, optimizer, cfg)
    print('finished training')
    utils.save_replay()
