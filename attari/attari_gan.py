import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import argparse
import cv2
import ale_py
import torch.optim as optim
from tensorboardX import SummaryWriter

import torchvision.utils as vutils
import logging

gym.register_envs(ale_py)
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

LATENT_VECTOR_SIZE = 100
IMAGE_SIZE=64
DISCR_FILTERS = 64
GENER_FILTERS = 64
BATCH_SIZE = 16

LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1000

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

class Discriminator(nn.Module):  # capital M
    def __init__(self, input_shape):
        super().__init__()
        
        self.conv_pipe = nn.Sequential(
            # Layer 1: (3, 64, 64) → (64, 32, 32)
            nn.Conv2d(input_shape[0], DISCR_FILTERS, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            # Layer 2: (64, 32, 32) → (128, 16, 16)
            nn.Conv2d(DISCR_FILTERS, DISCR_FILTERS * 2, 4, 2, 1),
            nn.BatchNorm2d(DISCR_FILTERS * 2),
            nn.LeakyReLU(0.2),
            
            # Layer 3: (128, 16, 16) → (256, 8, 8)
            nn.Conv2d(DISCR_FILTERS * 2, DISCR_FILTERS * 4, 4, 2, 1),
            nn.BatchNorm2d(DISCR_FILTERS * 4),
            nn.LeakyReLU(0.2),
            
            # Layer 4: (256, 8, 8) → (512, 4, 4)
            nn.Conv2d(DISCR_FILTERS * 4, DISCR_FILTERS * 8, 4, 2, 1),
            nn.BatchNorm2d(DISCR_FILTERS * 8),
            nn.LeakyReLU(0.2),
            
            # Layer 5: (512, 4, 4) → (1, 1, 1)
            nn.Conv2d(DISCR_FILTERS * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.conv_pipe(x).view(-1)

class Generator(nn.Module):
    def __init__(self, output_shape):
        super(Generator, self).__init__()
        # pipe deconvolves input vector into (3, 64, 64) image
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LATENT_VECTOR_SIZE, out_channels=GENER_FILTERS * 8,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(GENER_FILTERS * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 8, out_channels=GENER_FILTERS * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 4, out_channels=GENER_FILTERS * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 2, out_channels=GENER_FILTERS,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS, out_channels=output_shape[0],
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.pipe(x)
    
def iterate_batches(envs, batch_size=BATCH_SIZE):
    batch = [e.reset()[0] for e in envs]
    env_gen = iter(lambda: random.choice(envs), None)

    while True:
        e = next(env_gen)
        sample_action = e.action_space.sample()
        obs, reward, is_terminated, is_truncated, info  = e.step(sample_action)
        if np.mean(obs) > 0.01:
            batch.append(obs)
        if len(batch) == batch_size:
            # Normalising input between -1 to 1
            batch_np = np.array(batch, dtype=np.float32) * 2.0 / 255.0 - 1.0
            yield torch.tensor(batch_np)
            batch.clear()
        if is_terminated or is_truncated:
            e.reset()

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

    net_discr = Discriminator(input_shape=input_shape).to(device)
    net_gener = Generator(output_shape=input_shape).to(device)

    objective = nn.BCELoss()
    gen_optimizer = optim.Adam(params=net_gener.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    dis_optimizer = optim.Adam(params=net_discr.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    writer = SummaryWriter()

    gen_losses = []
    dis_losses = []
    iter_no = 0

    true_labels_v = torch.ones(BATCH_SIZE, dtype=torch.float32, device=device)
    fake_labels_v = torch.zeros(BATCH_SIZE, dtype=torch.float32, device=device)

    for batch_v in iterate_batches(envs):
        # generate extra fake samples, input is 4D: batch, filters, x, y
        gen_input_v = torch.FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1).normal_(0, 1).to(device)
        batch_v = batch_v.to(device)
        gen_output_v = net_gener(gen_input_v)

        # train discriminator
        dis_optimizer.zero_grad()
        dis_output_true_v = net_discr(batch_v)
        dis_output_fake_v = net_discr(gen_output_v.detach())
        dis_loss = objective(dis_output_true_v, true_labels_v) + objective(dis_output_fake_v, fake_labels_v)
        dis_loss.backward()
        dis_optimizer.step()
        dis_losses.append(dis_loss.item())

        # train generator
        gen_optimizer.zero_grad()
        dis_output_v = net_discr(gen_output_v)
        gen_loss_v = objective(dis_output_v, true_labels_v)
        gen_loss_v.backward()
        gen_optimizer.step()
        gen_losses.append(gen_loss_v.item())

        iter_no += 1
        if iter_no % REPORT_EVERY_ITER == 0:
            log.info("Iter %d: gen_loss=%.3e, dis_loss=%.3e", iter_no, np.mean(gen_losses), np.mean(dis_losses))
            writer.add_scalar("gen_loss", np.mean(gen_losses), iter_no)
            writer.add_scalar("dis_loss", np.mean(dis_losses), iter_no)
            gen_losses = []
            dis_losses = []
        if iter_no % SAVE_IMAGE_EVERY_ITER == 0:
            writer.add_image("fake", vutils.make_grid(gen_output_v.data[:64], normalize=True), iter_no)
            writer.add_image("real", vutils.make_grid(batch_v.data[:64], normalize=True), iter_no)