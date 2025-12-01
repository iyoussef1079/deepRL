import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import argparse
import cv2
import ale_py

gym.register_envs(ale_py)

IMAGE_SIZE=64

class InputWrapper(gym.ObservationWrapper):
    def __init__(self, *args):
        super(InputWrapper, self).__init__(*args)
        assert isinstance(self.observation_space, gym.spaces.Box)
        old_space = self.observation_space
        self.observation_space = gym.spaces.Box(
        self.observation(old_space.low),
        self.observation(old_space.high),
        dtype=np.float32)

    def observation(self, observation):
        new_obs = cv2.resize(
        observation, (IMAGE_SIZE, IMAGE_SIZE))
        new_obs = np.moveaxis(new_obs, 2, 0)
        return new_obs.astype(np.float32)

class Discriminator(nn.module):
    def __init__(self, *args):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)

class Generator(nn.module):
    def __init__(self, *args):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--cuda", default=True, action='store_true',
    help="Enable cuda computation")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    envs = [
    InputWrapper(gym.make(name))
    for name in ('ALE/Breakout-v5', 'ALE/Berzerk-v5', 'ALE/Atlantis2-v5')
    ]
    input_shape = envs[0].observation_space.shape