# Python module to collect data for the Perception net training
import click
import numpy as np
import os

from world import World
from tqdm import tqdm


@click.option('--gui', default=True)
@click.option('--collect-path', default=os.cwdir())
@click.option('--num-episodes', default=1)
@click.command()
def collect(collect_path=os.cwdir(), gui=True, num_episodes=1):
    """Generates data required for training the Perception LEC

    Args:
        gui (bool, optional): [Enable GUI when collecting data]. Defaults to True.
        collect_path ([type], optional): [Directory in which the collected data will be stored]. Defaults to os.cwdir().
        num_episodes (int, optional): [Number of episodes over which to collect the data]. Defaults to 1.
    """
    # Create the world
    world = World(gui=gui, collect=collect, collect_path=collect_path)

    for episode in tqdm(range(num_episodes)):
        print(f'Running episode:{episode + 1}')
        # Sample random distance and velocity values
        initial_distance = np.random.normal(100, 1)
        initial_velocity = np.random.uniform(25, 28)

        # Sample a random precipitation parameter
        precipitation = np.random.uniform(0, 20)

        # Initialize the world with the sampled params
        world.init(initial_velocity, initial_distance, precipitation)
        while True:
            episode_status = world.step()
            if episode_status == "DONE":
                break
