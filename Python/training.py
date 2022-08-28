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
import argparse

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

def validation_loop(agent,environment,img_processing, cfg, val_seeds=[246,247,248,249,250,251,252,253,254,255]):
    # # How to handle the different end signals
    # 16 feb 2021 previously:
    # RESET_UPON_END_SIGNAL = {0:False,  # Nothing happened
    #                          1:False,   # Box collision
    #                          2:False,   # Wall collision
    #                          3:True}    # Reached step target

    # Set nn.module to evaluation mode
    agent.policy_net.eval()

    # Reset counters
    wall_collisions = 0
    box_collisions = 0
    total_reward = 0
    endless_loops = 0
    step_count = 0
    fwd_steps = 0


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

        if cfg['dist_feedback']: # Additional channel encodes distance from start
            state = torch.cat([state, torch.zeros(1,1,cfg['imsize'],cfg['imsize'],device=cfg['device'])], dim=1)

        side_steps = 0

        # Episode starts here:
        for t in count():

            # 1. Agent performs a step (based on the current state) and obtains next state
            action = agent.select_action(state,validation=True)
            end, reward, next_state_raw = environment.step(action.item())
            frame = img_processing(next_state_raw).to(agent.device)
            next_state = frame_stack.update_with(frame)
            side_steps = side_steps + 1  if action != 0 else 0

            if action == 0:
                fwd_steps += 1

            if cfg['dist_feedback']: # Additional channel encodes distance from start
                next_state = torch.cat([next_state,fwd_steps*torch.ones(1,1,cfg['imsize'],cfg['imsize'],device=cfg['device'])], dim=1)/cfg['n_target_steps']

            # 2. interpret reward
            if reward > 100:
                reward = -(reward -100)
            reward *= cfg['reward_multiplier']

            if side_steps>cfg['reset_after_nr_sidesteps']:
                reward = cfg['early_stop_reward']

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
            elif end == 3: # Reached step target. Was: elif cfg['reset_end_is_{}'.format(end))
                step_count += t
                break
            else:
                state = next_state

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
    best_reward = np.NINF
    best_tr_reward = np.NINF
    optimizer.optimization_count = 0
    target_net_update_count = 0

    for episode in range(cfg['max_episodes']):

        # Valdation loop
        if episode_reward > best_tr_reward:
            best_tr_reward = episode_reward
            best_episode_so_far = True

        if (episode % cfg['validate_every'] == 0) or best_episode_so_far:
            best_episode_so_far = False
            val_performance = validation_loop(agent,environment,img_processing,cfg)
            val_reward = val_performance[-1]
            print('episode {}, step count: {} wall_collisions: {}, box_collisions: {}, endless_loops: {}, total_reward: {}'.format(episode,*val_performance))

            # Save best model
            if val_reward > best_reward:
                print("model improved! Saving to {}".format(cfg['model_path']))
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
        fwd_steps = 0 # Total forward step counter (for dist from start feedback)

        # Stop training after (either maximum number of steps or maximum number of episodes)
        if optimizer.optimization_count > cfg['max_optim_steps']:
            break


        # Target net is updated once in a few steps (double Q-learning)
        if optimizer.optimization_count / cfg['target_update'] >= target_net_update_count:  #steps
            print('Target net updated!')
            agent.update_target_net()
            target_net_update_count += 1


        # Reset environment at start of episode
        # seed = torch.randint(250,(1,)).item()
        # _, _, _ = environment.setRandomSeed(seed)
        _, _, _ = environment.reset(cfg['training_condition'])

        # Create an empty frame stack and fill it with frames
        frame_stack = imgproc.FrameStack(stack_size=cfg['stack_size'] )
        for _ in range(cfg['stack_size'] ):
            _, _, frame_raw = environment.step(0)
            frame = img_processing(frame_raw).to(agent.device)
            state = frame_stack.update_with(frame)

        if cfg['dist_feedback']: # Additional channel encodes distance from start
            state = torch.cat([state, torch.zeros(1,1,cfg['imsize'],cfg['imsize'],device=cfg['device'])], dim=1)

        # Episode starts here:
        for t in count():

            # 1. Agent performs a step (based on the current state) and obtains next state
            agent.policy_net.eval()
            action = agent.select_action(state)
            side_steps = side_steps + 1  if action != 0 else 0
            end, reward, frame_raw = environment.step(action.item())
            agent_died = cfg['reset_end_is_{}'.format(end)] or side_steps > cfg['reset_after_nr_sidesteps']
            frame = img_processing(frame_raw).to(agent.device)
            next_state = frame_stack.update_with(frame) if not agent_died else None

            if action == 0:
                fwd_steps += 1

            if cfg['dist_feedback'] and next_state is not None: # Additional channel encodes distance from start
                next_state = torch.cat([next_state,fwd_steps*torch.ones(1,1,cfg['imsize'],cfg['imsize'],device=cfg['device'])], dim=1)/cfg['n_target_steps']


            # 2. Interpret reward signal
            if reward > 100:
                reward = -(reward -100)
            reward *= cfg['reward_multiplier']

            if side_steps > cfg['reset_after_nr_sidesteps']:
                reward = cfg['early_stop_reward']

            # 3. Push the transition to replay memory (in the right format & shape)
            reward = torch.tensor([reward], device=agent.device,dtype=torch.float)
            action = action.unsqueeze(0)
            agent.memory.push(state, action, next_state, reward)


            # 4. optimize model
            agent.policy_net.train()
            if len(agent.memory) > cfg['replay_start_size']:

                for _ in range(cfg['optimizations_per_step']):
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
                    optimizer.optimization_count += 1

            else:
                # Do not count steps, as optimization has not started yet (delay epsilon decay)
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

def main(config_file=None, specs_file=None):
    """ Either config_file (yaml) with configurations or specs_file (csv) with a
    list of models and specifications must be specified.  This script calls the
    train loop once for each of the specified models /configurations.

    Usage:
    run python training.py -c <filename> to specify yaml file with the training configuration"
    run python training.py -s <filename> to specify csv file with list of model specifications"
    """

    if config_file is not None:
        # Create train_specs (pandas DataFrame) from specified configurations (yaml file)
        print("creating specs list from configuration file..")
        train_specs = utils.load_train_configs(config_file)

        # Create directory, or continue from previous training session.
        # train_specs (DataFrame) are written to a csv file in the save directory
        assert train_specs.savedir.nunique() == 1
        savedir = train_specs.savedir[0]
        specs_file  = os.path.join(savedir,'_specs.csv')
        if not os.path.isdir(savedir):
            print('creating directory: {}'. format(savedir))
            os.makedirs(savedir)
        else:
            print("Found existing directory. resuming previous training session..")
            assert os.path.exists(specs_file), "Cannot resume training session: no specs file was found. Please specify new directory."
            existing_specs = pd.read_csv(specs_file).set_index('model_name').fillna(np.nan).replace([np.nan], [None])
            if not all(train_specs.index.isin(existing_specs.index)):
                new_specs = train_specs[~train_specs.index.isin(existing_specs.index)]
                train_specs = pd.concat([existing_specs, new_specs])
                print("adding new models to the specs list: \n{}".format('\n'.join(new_specs.index.tolist())))
            else:
                train_specs = existing_specs
                print("Using the already existing specs list (instead of the provided config file)")

    else:
        # Load training specficiations from the provided specs list
        train_specs = pd.read_csv(specs_file).set_index('model_name').fillna(np.nan).replace([np.nan], [None])
        savedir = train_specs.savedir[0]
        print("resuming previous training session..")

    # Run a training loop for each specified model (i.e. each row in the train_configs / specs_file)
    environment_connected = False
    for _ , cfg in train_specs.iterrows():
        current_model = cfg.name

        # Additional training settings (inferred from specified settings)
        cfg['training_condition'] = {'plain': 0, 'complex': 1}[cfg['complexity']]
        cfg['model_path']         = os.path.join(savedir,'{}.pth'.format(current_model)) # Save path for model
        cfg['logfile']            = os.path.join(savedir,'{}_train_stats.csv'.format(current_model)) # To save the training stats

        # Write train_specs to csvfile
        if train_specs.loc[current_model, 'status'] == 'finished':
            print('skipping.. already finished in previous training: {}'.format(current_model))
            continue
        train_specs.loc[current_model, 'status'] = 'training'
        train_specs.to_csv(specs_file)
        print(train_specs)

        # Initialize model components
        torch.manual_seed(cfg['seed'])
        agent = model.DoubleDQNAgent(**cfg)
        img_processing = imgproc.ImageProcessor(**cfg)
        optimizer = optim.Adam(agent.policy_net.parameters(), lr = cfg['lr_dqn'])
        environment =  pyClient.Environment(**cfg) if not environment_connected else environment # Only initialized on first run

        # # Training
        assert environment.client is not None, "Error: could not connect to env. Make sure to start Unity server first!"
        environment_connected = True
        train(agent, environment, img_processing, optimizer, cfg)

        # Write specs to training file
        train_specs.loc[current_model, 'status'] = 'finished'
        train_specs.to_csv(specs_file)
        print('finished training')

	# Testing 
        results = testing.test(agent, environment, img_processing, cfg)
        for metric, result  in results.items():
            train_specs.loc[current_model,metric] = result # add each of the result metrics to the train_specs_dataframe
        train_specs.to_csv(specs_file)
        print('finished testing')
        print(f'results are saved in {specs_file}')

        # write replay memory to video
        #videopath = os.path.join(savedir,'{}.avi'.format(current_model))
        #utils.save_replay(agent.memory.memory, videopath,(cfg['imsize'], cfg['imsize']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-c", "--config", type=str, default=None,
                    help="filename of config file (yaml) with the training configurations: e.g. '_config.yaml' ")
    group.add_argument("-s", "--specs", type=str, default=None,
                        help="filename of specs file (csv) with the list of model specifications")
    args = parser.parse_args()
    main(args.config, args.specs)
