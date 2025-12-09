#!/usr/bin/env python3
import gymnasium as gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter
import argparse
import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70
MODEL_FILE = "acrobot.pth"

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs, _ = env.reset()
    sm = nn.Softmax(dim=1)
    
    while True:
        obs_v = torch.FloatTensor(np.array([obs]))
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        
        action = np.random.choice(len(act_probs), p=act_probs)
        
        next_obs, reward, is_terminated, is_truncated, _ = env.step(action)
        
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))
        
        if is_terminated or is_truncated:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs, _ = env.reset()
            
            if len(batch) == batch_size:
                yield batch
                batch = []
        
        obs = next_obs


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

    train_obs_v = torch.FloatTensor(np.array(train_obs))
    train_act_v = torch.LongTensor(np.array(train_act))
    return train_obs_v, train_act_v, reward_bound, reward_mean


def play_model(file_path):
    """Loads the model and plays it in human render mode"""
    if not os.path.exists(file_path):
        print(f"Error: Model file '{file_path}' not found. Train first!")
        return

    # Force render mode to human for playback
    env = gym.make("Acrobot-v1", render_mode="human")
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Load model architecture and weights
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    net.load_state_dict(torch.load(file_path))
    net.eval() # Switch to evaluation mode

    sm = nn.Softmax(dim=1)
    
    print(f"Playing model from {file_path}...")
    
    for _ in range(5): # Play 5 episodes
        obs, _ = env.reset()
        total_reward = 0.0
        
        while True:
            obs_v = torch.FloatTensor(np.array([obs]))
            with torch.no_grad():
                act_probs = sm(net(obs_v)).numpy()[0]
            
            # Select action (Greedy or Stochastic)
            # For demonstration, greedy (argmax) usually looks 'smarter', 
            # but stochastic (random.choice) is how we trained.
            action = np.argmax(act_probs) 
            
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                print(f"Episode finished. Total Reward: {total_reward}")
                break
                
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--play", action="store_true", help="Play the trained model instead of training")
    args = parser.parse_args()

    if args.play:
        play_model(MODEL_FILE)
        sys.exit(0)

    # --- Training Logic ---
    env = gym.make("Acrobot-v1") # No render during training (faster)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-acrobot")

    print("Training started...")
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b))
        
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        
        # Changed threshold to -62 based on your previous success
        if reward_m > -70: 
            print(f"Solved! Saving model to {MODEL_FILE}")
            torch.save(net.state_dict(), MODEL_FILE)
            break
            
    writer.close()