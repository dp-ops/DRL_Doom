from turtle import done
from vizdoom import *
import os
import random 
import time
# For identity matrix
import numpy as np
# Import env base class from OpneAI gym
import gymnasium as gym
# Import spaces from gym
from gymnasium.spaces import Discrete, Box
# Import opencv
import cv2
#import callback class from sb3
from stable_baselines3.common.callbacks import BaseCallback

#Directories
CHECKPOINT_DIR_BASE = './train/train_basic'
CHECKPOINT_DIR_DC = './train/train_dc'
CHECKPOINT_DIR_DEADLY = './train/train_deadly_dqn'
# LOG_DIR = './logs/logs_basic'

class VizDoomGym(gym.Env):
    def __init__(self, render = False, config = "github/ViZDoom/scenarios/basic.cfg"):

        super().__init__()

        # Setup the game
        self.game = DoomGame()
        self.game.load_config(config)

        if render == False:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)

        # Init the game
        self.game.init()

        self.observation_space = Box(0, 255, (100, 160, 1), dtype=np.uint8)
        self.action_space = Discrete(3)

    # How to take a step in the environment
    def step(self, action):
        actions = np.identity(3, dtype=int).tolist()
        reward = self.game.make_action(actions[action], 4)
        
        if self.game.get_state() :
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
            ammo = self.game.get_state().game_variables[0]
            info = {'ammo': ammo}
        else: 
            state = np.zeros(self.observation_space.shape)
            info = {'ammo': 0}

        terminated = self.game.is_episode_finished()
        truncated = False

        return state, reward, terminated, truncated, info
    # Define how to render env
    def render(self, mode='human'):
        pass
    # Define how to reset the environment
    def reset(self, seed=None, options=None):
        # Optionally set seeds, if needed
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self.grayscale(state), {}
    
    # Grayscale and resize
    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100, 160, 1))

        return state


    # Close down the game
    def close(self):
        self.game.close()


class VizDoomGym_DC(gym.Env):
    def __init__(self, render = False, config = "github/ViZDoom/scenarios/defend_the_center.cfg"):

        super().__init__()

        # Setup the game
        self.game = DoomGame()
        self.game.load_config(config)

        if render == False:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)

        # Init the game
        self.game.init()

        self.observation_space = Box(0, 255, (100, 160, 1), dtype=np.uint8)
        self.action_space = Discrete(3)

    # How to take a step in the environment
    def step(self, action):
        actions = np.identity(3, dtype=int).tolist()
        reward = self.game.make_action(actions[action], 4)
        
        if self.game.get_state() :
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
            ammo = self.game.get_state().game_variables[0]
            info = {'ammo': ammo}
        else: 
            state = np.zeros(self.observation_space.shape)
            info = {'ammo': 0}

        terminated = self.game.is_episode_finished()
        truncated = False

        return state, reward, terminated, truncated, info
    # Define how to render env
    def render(self, mode='human'):
        pass
    # Define how to reset the environment
    def reset(self, seed=None, options=None):
        # Optionally set seeds, if needed
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self.grayscale(state), {}
    
    # Grayscale and resize
    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100, 160, 1))

        return state


    # Close down the game
    def close(self):
        self.game.close()

class VizDoomGym_DeadlyCorridor(gym.Env):
    def __init__(self, render = False, config = "github/ViZDoom/scenarios/deadly_corridor_s1.cfg"):

        super().__init__()

        # Setup the game
        self.game = DoomGame()
        self.game.load_config(config)

        if render == False:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)

        # Init the game
        self.game.init()

        self.observation_space = Box(0, 255, (100, 160, 1), dtype=np.uint8)
        self.action_space = Discrete(7)

        self.damage_taken = 0
        self.damage_count = 0
        self.ammo = 52

    # How to take a step in the environment
    def step(self, action):
        actions = np.identity(7, dtype=int).tolist()
        movement_reward = self.game.make_action(actions[action], 4)

        reward = 0
        if self.game.get_state() :
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)

            #reward shaping
            game_variabels = self.game.get_state().game_variables
            health, damage_taken, damage_count, ammo = game_variabels

            #reward deltas calc
            damage_taken_delta = -damage_taken + self.damage_taken  
            self.damage_taken = damage_taken
            damage_count_delta = damage_count - self.damage_count
            self.damage_count = damage_count
            ammo_delta = ammo - self.ammo
            self.ammo = ammo  

            reward = movement_reward + damage_taken_delta*100 + damage_count_delta*200 + ammo_delta*3

            info = {'ammo': ammo}
        else: 
            state = np.zeros(self.observation_space.shape)
            info = {'ammo': 0}

        terminated = self.game.is_episode_finished()
        truncated = False

        return state, reward, terminated, truncated, info
    # Define how to render env
    def render(self, mode='human'):
        pass
    # Define how to reset the environment
    def reset(self, seed=None, options=None):
        # Optionally set seeds, if needed
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self.grayscale(state), {}
    
    # Grayscale and resize
    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100, 160, 1))

        return state


    # Close down the game
    def close(self):
        self.game.close()


class TrainAndLogging(BaseCallback):

    def __init__(self, check_freq, mode='base', verbose=1):
        super(TrainAndLogging, self).__init__(verbose)
        self.check_freq = check_freq
        if mode == 'base':
            self.save_path = CHECKPOINT_DIR_BASE
        elif mode == 'DC':
            self.save_path = CHECKPOINT_DIR_DC
        elif mode == 'deadly':
            self.save_path = CHECKPOINT_DIR_DEADLY
        else:
            raise ValueError("Invalid mode. Use 'base' or 'DC'.")
    
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True