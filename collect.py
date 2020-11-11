# Python module to collect data for the Perception net training
import click
import numpy as np
import os

from world import World
from tqdm import tqdm


@click.option('--gui', is_flag=True, default=False)
@click.option('--collect-path', default=os.getcwd())
@click.option('--num-episodes', type=int, default=1)
@click.command()
def collect(collect_path=os.getcwd(), gui=False, num_episodes=1):
    """Generates data required for training the Perception LEC

    Args:
        gui (bool, optional): [Enable GUI when collecting data]. Defaults to True.
        collect_path ([type], optional): [Directory in which the collected data will be stored]. Defaults to os.cwdir().
        num_episodes (int, optional): [Number of episodes over which to collect the data]. Defaults to 1.
    """
    # Create the world
    world = World(gui=gui, collect=collect, collect_path=collect_path)
    world.episode = 26

    for episode in range(num_episodes):
        print(f'Running episode:{episode + 1}')
        # Sample random distance and velocity values
        initial_distance = np.random.normal(100, 1)
        initial_velocity = np.random.uniform(25, 28)

        # Sample a random precipitation parameter
        precipitation = np.random.uniform(0, 20)
        print(f'Precipitation: {precipitation}')

        # Initialize the world with the sampled params
        dist, vel, status = world.init(initial_velocity, initial_distance, precipitation)
        if status == 'FAILED':
            print(f'Reset failed. Stopping episode {episode + 1} and continuing!')
            continue
        while True:
            episode_status = world.step()
            if episode_status == "DONE":
                break


if __name__ == '__main__':
    collect()
